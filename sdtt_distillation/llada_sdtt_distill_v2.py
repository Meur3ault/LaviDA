import os
import sys
import copy
import lightning as L

# Torch utilities
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import LaViDa components
from llava.model.language_model.llava_llada import (
    LlavaLladaForMaskedDiffusion,
    forward_process,
    sample_t
)
from llava.model.builder import load_pretrained_model
from llava.train.train import preprocess_multimodal, preprocess
from llava.mm_utils import process_images
from llava.model.language_model.llada.generate import generate as llada_generate

class OneRoundSDTTDistiller(L.LightningModule):
    """
    Distiller class for one-round SDTT distillation on LLaVa-LLaDA.
    
    Key insight: LLaVa-LLaDA already has time sampling in forward_process(),
    but does NOT use time conditioning (model doesn't receive sigma).
    
    The student learns to match teacher's predictions over num_distill_steps (e.g., 2 steps),
    allowing the student to generate in fewer steps (e.g., half the steps).
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Model configuration
        self.checkpoint_path = config['checkpoint_path']
        self.student_checkpoint_path = config.get('student_checkpoint_path', None)  # Optional student checkpoint path
        self.model_name = config['model_name']
        self.device_str = config['device']
        self.device_map = config.get('device_map', 'cuda:0')
        self.torch_dtype = config.get('torch_dtype', 'bfloat16')
        self.use_fim = config.get('use_fim', False)
        
        # Distillation parameters
        self.teacher_num_inference_steps = config.get('teacher_num_inference_steps', 64)
        self.max_new_tokens = config.get('max_new_tokens', 64)
        self.num_distill_steps = config.get('num_distill_steps', 2)  # Teacher runs 2 steps
        self.sampling_eps = config.get('sampling_eps', 1e-3)
        self.teacher_num_tokens_per_step = self.max_new_tokens // self.teacher_num_inference_steps
        self.student_num_tokens_per_step = self.teacher_num_tokens_per_step * self.num_distill_steps
        
        # Calculate dt for teacher - how much timestep decreases per step
        self.dt = (1 - self.sampling_eps) / self.teacher_num_inference_steps
        
        # Loss configuration
        self.distill_mode = config.get('distill_mode', 'kl-fwd')  # kl-fwd, kl-bwd, mse, tvd
        self.loss_precision = config.get('loss_precision', '32')  # 32 or 64

        # Token IDs for LLaVa-LLaDA
        self.eos_id = 126081
        self.mask_id = 126336
        self.fim_id = 126085

        # Load the pretrained model
        vision_kwargs = dict(
            mm_vision_tower="google/siglip-so400m-patch14-384",
            mm_resampler_type=None,
            mm_projector_type='mlp2x_gelu',
            mm_hidden_size=1152,
            use_mm_proj=True
        )
        self.tokenizer, self.teacher_model, self.image_processor, self.max_length = load_pretrained_model(
            self.checkpoint_path,
            None,
            self.model_name,
            device_map=self.device_map,
            vision_kwargs=vision_kwargs,
            torch_dtype=self.torch_dtype
        )

        # ✨ FIX VOCAB SIZE MISMATCH
        print(f"\nChecking vocab size compatibility...")
        tokenizer_vocab_size = len(self.tokenizer)
        model_vocab_size = self.teacher_model.get_model().transformer.wte.weight.shape[0]
        
        print(f"  Tokenizer vocab size: {tokenizer_vocab_size}")
        print(f"  Teacher model embedding size: {model_vocab_size}")
        
        if tokenizer_vocab_size != model_vocab_size:
            print(f"  ⚠️  Vocab size mismatch detected!")
            print(f"  Resizing teacher model embeddings to match tokenizer...")
            self.teacher_model.resize_token_embeddings(tokenizer_vocab_size)
            print(f"  ✓ Teacher model embeddings resized to {tokenizer_vocab_size}")
        else:
            print(f"  ✓ Teacher vocab size matches tokenizer")
        
        # Load student model from checkpoint path if provided, otherwise create as copy of teacher
        if self.student_checkpoint_path is not None:
            print(f"\nLoading student model from checkpoint: {self.student_checkpoint_path}")
            _, self.student_model, _, _ = load_pretrained_model(
                self.student_checkpoint_path,
                None,
                self.model_name,
                device_map=self.device_map,
                vision_kwargs=vision_kwargs,
                torch_dtype=self.torch_dtype
            )
            
            # Check and fix vocab size for student model
            student_model_vocab_size = self.student_model.get_model().transformer.wte.weight.shape[0]
            if student_model_vocab_size != tokenizer_vocab_size:
                print(f"  Student model vocab size: {student_model_vocab_size}, resizing to {tokenizer_vocab_size}")
                self.student_model.resize_token_embeddings(tokenizer_vocab_size)
            print(f"  ✓ Student model loaded from checkpoint")
        else:
            # Create student as a copy of teacher
            print(f"\nCreating student model as copy of teacher")
            self.student_model = copy.deepcopy(self.teacher_model)

        # Freeze the teacher model
        self.teacher_model.eval()
        self.student_model.train()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        # Move both models to device
        self.teacher_model.to(self.device_str)
        self.student_model.to(self.device_str)

        print(self.teacher_model.get_model().transformer.wte.weight.shape[0])
        print(self.student_model.get_model().transformer.wte.weight.shape[0])

        # Setup loss function
        self._setup_loss_fn()

    def _setup_loss_fn(self):
        """Setup loss function based on distill_mode"""
        mode = self.distill_mode
        if mode == "mse":
            self._loss_fn = self._mse
        elif mode == "tvd":
            self._loss_fn = self._tvd
        elif mode == "kl-fwd":
            self._loss_fn = self._fwd_kl
        elif mode == "kl-bwd":
            self._loss_fn = self._bwd_kl
        else:
            raise ValueError(f"Unknown distill_mode: {mode}")

    def _mse(self, preds, target):
        """Mean Squared Error loss"""
        return F.mse_loss(preds.exp(), target.exp())  # MSE on probabilities

    def _tvd(self, preds, target):
        """Total Variation Distance loss"""
        return (preds.exp() - target.exp()).abs().sum(-1).mean()

    def _fwd_kl(self, preds, target):
        """Forward KL divergence: KL(target || preds)"""
        return F.kl_div(preds, target, log_target=True, reduction="batchmean")

    def _bwd_kl(self, preds, target):
        """Backward KL divergence: KL(preds || target)"""
        return F.kl_div(target, preds, log_target=True, reduction="batchmean")

    def forward_teacher(self, inputs_embeds, attention_mask=None):
        """
        Teacher forward pass.
        Note: LLaVa-LLaDA does NOT use time conditioning.
        Call model.model directly to avoid labels processing in forward().
        """
        # Call the underlying LLaDAModel directly (bypasses the forward() that needs labels)
        # LLaDAModel.forward() returns LLaDAOutput which has .logits attribute
        print("DEBUGGING: Model vocab size")
        print(self.teacher_model.get_model().transformer.wte.weight.shape[0])
        outputs = self.teacher_model.model(
            input_ids=None,
            input_embeddings=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False
        )
        return outputs.logits

    def forward_student(self, inputs_embeds, attention_mask=None):
        """
        Student forward pass.
        Call model.model directly to avoid labels processing in forward().
        """
        # Call the underlying LLaDAModel directly (bypasses the forward() that needs labels)
        # LLaDAModel.forward() returns LLaDAOutput which has .logits attribute
        outputs = self.student_model.model(
            input_ids=None,
            input_embeddings=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False
        )
        return outputs.logits

    @torch.no_grad()
    def _teacher_logprobs_on_mask(self, inputs_embeds, masked_indices, attention_mask, t_start):
        """
        Collect teacher predictions for ALL masked tokens over num_distill_steps.
        
        This is the CORE of SDTT distillation:
        - Teacher runs for num_distill_steps (e.g., 2 steps)
        - At each step, teacher predicts and unmasks some tokens
        - We collect predictions for each token at the step it was unmasked
        
        Args:
            inputs_embeds: Embeddings with masked positions [B, L, D]
            masked_indices: Boolean mask indicating which positions are masked [B, L]
            attention_mask: Attention mask [B, L]
            t_start: Starting timestep [B, 1]
        
        Returns:
            teacher_predictions: Log probabilities for all masked tokens [B, L, vocab_size]
        """
        dt = self.dt
        device = inputs_embeds.device
        bsz, seq_len, hidden_dim = inputs_embeds.shape

        # Create evenly-spaced timesteps from t_start to t_end
        space = torch.linspace(1, 0, self.num_distill_steps, device=device).double()[:, None]
        t_start = t_start[None, :].double()
        t_end = t_start - dt * self.num_distill_steps
        ts = t_start * space + (1 - space) * t_end
        ts = torch.maximum(ts, torch.tensor(self.sampling_eps, device=device))

        # Get vocab_size from first forward pass
        #logits = self.forward_teacher(inputs_embeds, attention_mask)
        vocab_size = 126464 # Fixed value to skip inferencing
        teacher_predictions = torch.zeros((bsz, seq_len, vocab_size), device=device)
        unmasked_tokens = torch.zeros((bsz, seq_len), device=device, dtype=torch.bool)
        curr_embeds = inputs_embeds.clone()

        # Run teacher for num_distill_steps
        for idx in range(len(ts)):
            # Get teacher predictions on current state
            logits = self.forward_teacher(curr_embeds, attention_mask)
            log_p_x0 = F.log_softmax(logits, dim=-1)
            
            # Sample which tokens to unmask based on confidence
            predicted_tokens = torch.argmax(log_p_x0, dim=-1)
            
            # Calculate confidence for masked positions
            p = log_p_x0.exp()
            confidence = torch.gather(p, dim=-1, index=predicted_tokens.unsqueeze(-1)).squeeze(-1)
            
            # Only consider currently masked positions (not yet unmasked)
            currently_masked = masked_indices & (~unmasked_tokens)
            confidence[~currently_masked] = float('-inf')
            
            # Determine how many tokens to unmask this step
            num_masked = currently_masked.sum().item()
            if num_masked == 0:
                break
            
            # IMPORTANT: Unmask exactly teacher_num_tokens_per_step tokens per step
            # This is the core of SDTT - teacher unmasks a fixed number of tokens each step
            tokens_per_step = self.teacher_num_tokens_per_step
            
            # Select top-k confident tokens to unmask
            update_mask = torch.zeros_like(currently_masked, dtype=torch.bool)
            for j in range(bsz):
                if currently_masked[j].sum() > 0:
                    # Unmask exactly teacher_num_tokens_per_step (or remaining if less)
                    k = min(tokens_per_step, currently_masked[j].sum().item())
                    _, select_index = torch.topk(confidence[j], k)
                    update_mask[j, select_index] = True
            
            # DEBUG: Print teacher predictions for first batch at each step
            if idx < 2:  # Only print first 2 steps
                print(f"\n{'='*80}")
                print(f"TEACHER STEP {idx+1}/{self.num_distill_steps}")
                print(f"{'='*80}")
                print(f"Timestep t: {ts[idx, 0].item():.4f}")
                print(f"Currently masked positions: {currently_masked[0].sum().item()}")
                print(f"Tokens to unmask this step: {update_mask[0].sum().item()}")
                
                # Show positions being unmasked
                unmasked_positions = update_mask[0].nonzero(as_tuple=True)[0].tolist()
                print(f"\nPositions being unmasked: {unmasked_positions[:10]}{'...' if len(unmasked_positions) > 10 else ''}")
                
                # Show predicted tokens at these positions
                predicted_at_positions = predicted_tokens[0][update_mask[0]].tolist()
                print(f"Predicted token IDs: {predicted_at_positions[:10]}{'...' if len(predicted_at_positions) > 10 else ''}")
                
                # Show confidence scores
                confidence_at_positions = confidence[0][update_mask[0]].tolist()
                print(f"Confidence scores: {[f'{c:.4f}' for c in confidence_at_positions[:10]]}{'...' if len(confidence_at_positions) > 10 else ''}")
                
                # Show top-5 logits for first unmasked position
                if len(unmasked_positions) > 0:
                    first_pos = unmasked_positions[0]
                    logits_at_pos = logits[0, first_pos]
                    top_logits, top_indices = torch.topk(logits_at_pos, 5)
                    print(f"\nFirst unmasked position {first_pos}:")
                    print(f"  Top-5 logits: {top_logits.tolist()}")
                    print(f"  Top-5 token IDs: {top_indices.tolist()}")
                    
                    # Decode predicted token
                    pred_token = predicted_tokens[0, first_pos].item()
                    if pred_token >= 0 and pred_token < len(self.tokenizer):
                        pred_text = self.tokenizer.decode([pred_token])
                        print(f"  Predicted token: {pred_token} -> '{pred_text}'")
                
                print(f"{'='*80}\n")
            
            # Store predictions for newly unmasked tokens
            teacher_predictions[update_mask] = log_p_x0[update_mask]
            unmasked_tokens[update_mask] = True
            
            # Update embeddings: replace masked positions with predicted token embeddings
            predicted_embeds = self.teacher_model.get_model().transformer.wte(predicted_tokens)
            curr_embeds = torch.where(
                update_mask.unsqueeze(-1),
                predicted_embeds,
                curr_embeds
            )

        # Handle any remaining masked tokens with last predictions
        remaining_masked = masked_indices & (~unmasked_tokens)
        if remaining_masked.any():
            logits = self.forward_teacher(curr_embeds, attention_mask)
            log_p_final = F.log_softmax(logits, dim=-1)
            teacher_predictions[remaining_masked] = log_p_final[remaining_masked]

        print("Teacher logits")
        print(teacher_predictions[0, masked_indices[0]])
        return teacher_predictions


    def training_step(self, batch, batch_idx):
        """
        Training step for SDTT distillation on LLaVa-LLaDA.
        
        Key process:
        1. Use LLaVa's prepare_inputs_labels_for_multimodal to get embeddings
        2. Use forward_process() to sample t and create masked version
        3. Teacher runs num_distill_steps to predict all masked tokens
        4. Student runs 1 step to predict all masked tokens
        5. Minimize divergence between teacher and student predictions
        """
        # Ensure the student model is in train mode
        self.student_model.train()
        
        # ============================================================================
        # STEP 1: PREPARE INPUTS
        # ============================================================================
        input_ids = batch['input_ids']
        images = batch['images']
        labels = batch['labels']
        attention_mask = batch['attention_mask']
        
        bsz = input_ids.shape[0]
        
        # Validate labels
        assert labels is not None, "Labels should not be None for distillation"
        

        # ============================================================================
        # DEBUG: Decode input_ids and labels
        # ============================================================================
        if batch_idx % 100 == 0:  # Only print every 100 batches to avoid spam
            print(f"\n{'='*80}")
            print(f"TRAINING STEP DEBUG - Batch {batch_idx}")
            print(f"{'='*80}")
            
            # Decode first sample in batch
            sample_input_ids = input_ids[0]
            sample_labels = labels[0]
            sample_attention_mask = attention_mask[0]

            print("RAW input_ids and labels:")
            print(f"input_ids: {sample_input_ids.tolist()}")
            print(f"labels: {sample_labels.tolist()}")
            
            # Get actual length (non-padded)
            actual_length = sample_attention_mask.sum().item()
            
            print(f"\n--- INPUT_IDS (length={actual_length}) ---")
            print(f"Raw token IDs (first 20): {sample_input_ids[:20].tolist()}")
            print(f"Raw token IDs (last 20): {sample_input_ids[max(0, actual_length-20):actual_length].tolist()}")
            
            # Decode full input_ids to text, filtering out special negative tokens
            # (like -200 for IMAGE_TOKEN_INDEX which can't be decoded)
            valid_tokens = sample_input_ids[:actual_length].clone()
            # Replace negative tokens with a valid token ID (use pad token or 0)
            replacement_token = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            valid_tokens[valid_tokens < 0] = replacement_token
            full_input_text = self.tokenizer.decode(valid_tokens, skip_special_tokens=False)
            print(f"\nDecoded text:\n{full_input_text}")
            
            print(f"\n--- LABELS (length={actual_length}) ---")
            # Find where labels start (first non -100 token)
            labels_mask = sample_labels != -100
            if labels_mask.any():
                first_label_pos = labels_mask.nonzero(as_tuple=True)[0][0].item()
                print(f"First label position: {first_label_pos}")
                
                # Decode prompt part (labels == -100)
                prompt_ids = sample_input_ids[:first_label_pos].clone()
                replacement_token = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
                prompt_ids[prompt_ids < 0] = replacement_token
                prompt_text = self.tokenizer.decode(prompt_ids, skip_special_tokens=False)
                print(f"\n[PROMPT - labels=-100] (tokens 0 to {first_label_pos-1}):")
                print(f"{prompt_text}")
                
                # Decode answer part (labels != -100)
                # Use input_ids from the answer region (should match labels)
                answer_ids_from_input = sample_input_ids[first_label_pos:actual_length].clone()
                answer_ids_from_input[answer_ids_from_input < 0] = replacement_token
                answer_text_from_input = self.tokenizer.decode(answer_ids_from_input, skip_special_tokens=False)
                print(f"\n[ANSWER from input_ids] (tokens {first_label_pos} to {actual_length-1}):")
                print(f"{answer_text_from_input}")
                
                # Also decode from labels to verify they match
                answer_ids_from_labels = sample_labels[first_label_pos:actual_length].clone()
                # Replace -100 and any negative tokens
                answer_ids_from_labels[answer_ids_from_labels < 0] = replacement_token
                answer_text_from_labels = self.tokenizer.decode(answer_ids_from_labels, skip_special_tokens=False)
                print(f"\n[ANSWER from labels] (tokens {first_label_pos} to {actual_length-1}):")
                print(f"{answer_text_from_labels}")
                
                # Show token-level comparison
                print(f"\n--- TOKEN COMPARISON (positions {first_label_pos} to {first_label_pos+9}) ---")
                print(f"Input_ids: {sample_input_ids[first_label_pos:first_label_pos+10].tolist()}")
                print(f"Labels:    {sample_labels[first_label_pos:first_label_pos+10].tolist()}")
                print(f"Match: {torch.equal(sample_input_ids[first_label_pos:first_label_pos+10], sample_labels[first_label_pos:first_label_pos+10])}")
            else:
                print("No labels found (all -100)")
                print(f"Raw labels: {sample_labels[:actual_length].tolist()}")
            
            print(f"{'='*80}\n")


        # ============================================================================
        # STEP 2: PROCESS MULTIMODAL INPUTS
        # ============================================================================
        # Get image_sizes from batch if available (required for AnyRes mode)
        image_sizes = batch.get('image_sizes', None)
        with torch.no_grad():
            (_, position_ids, attention_mask, _, inputs_embeds, labels, _) = \
                self.teacher_model.prepare_inputs_labels_for_multimodal(
                    input_ids, None, attention_mask, None, labels,
                    images, ['image'], image_sizes=image_sizes, return_inputs=True
                )
        
        bsz, seq_len = labels.shape
        # Get vocab_size from actual model output to avoid mismatch
        # vocab_size = self.student_model.config.vocab_size
        
        
        # ============================================================================
        # STEP 3: CREATE LABEL MASK - Identify target tokens
        # ============================================================================
        labels_mask = ~(labels == -100)
        
        # Handle FIM tokens if needed
        if self.use_fim:
            infill_token_pos = labels == self.fim_id
        else:
            infill_token_pos = torch.zeros_like(labels, dtype=torch.bool)
        
        # ============================================================================
        # STEP 4: FORWARD PROCESS - Use LLaVa's forward_process()
        # ============================================================================
        # This samples t and creates masked_indices
        masked_indices, p_mask = forward_process(
            bsz, seq_len, device=labels.device,
            eps=self.sampling_eps,
            policy='uniform',
            policy_args=None
        )
        
        # Only mask valid target positions
        final_masked_indices = masked_indices & labels_mask & (~infill_token_pos)
        
        # Create mask embeddings
        noise_embeddings = self.teacher_model.get_model().transformer.wte(
            torch.tensor([self.mask_id], device=labels.device)
        ).view(1, 1, -1)
        
        # Apply masking to embeddings
        inputs_embeds_masked = torch.where(
            final_masked_indices.unsqueeze(-1),
            noise_embeddings,
            inputs_embeds
        )
        
        # Extract t from p_mask for teacher's multi-step rollout
        # p_mask = (1 - eps) * t + eps, so t = (p_mask - eps) / (1 - eps)
        t_start = (p_mask - self.sampling_eps) / (1 - self.sampling_eps)
        
        # ============================================================================
        # STEP 5: TEACHER FORWARD - Run num_distill_steps
        # ============================================================================
        teacher_preds = self._teacher_logprobs_on_mask(
            inputs_embeds_masked,
            final_masked_indices,
            attention_mask,
            t_start
        )
        
        # ============================================================================
        # STEP 6: STUDENT FORWARD - Run 1 step
        # ============================================================================
        student_logits = self.forward_student(inputs_embeds_masked, attention_mask)
        student_preds = F.log_softmax(student_logits, dim=-1)
        
        # ============================================================================
        # STEP 7: COMPUTE LOSS - Only on masked positions
        # ============================================================================
        is_mask = final_masked_indices
        
        # Extract predictions only for masked positions
        target = teacher_preds[is_mask]
        preds = student_preds[is_mask]
        
        # Apply precision
        if self.loss_precision == "64":
            target = target.to(torch.float64)
            preds = preds.to(torch.float64)
        elif self.loss_precision == "32":
            target = target.to(torch.float32)
            preds = preds.to(torch.float32)
        
        # Compute distillation loss
        loss = self._loss_fn(preds, target)
        
        # ============================================================================
        # STEP 8: LOGGING
        # ============================================================================
        self.log('train/loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log('train/mask_ratio', is_mask.float().mean(), prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log('train/timestep', t_start.mean(), prog_bar=False, logger=True, on_step=True, on_epoch=True)
        
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        """Validation: compare teacher vs student generation quality"""
        if batch_idx > 0:
            return
        
        input_ids = batch['input_ids'][:1]
        attention_mask_from_batch = batch['attention_mask'][:1]
        labels_from_batch = batch['labels'][:1]
        image_sizes = batch["image_sizes"][:1]
        
        # Remove answer part from input_ids
        # The batch contains both prompt AND answer (for training)
        # For generation, we only want the prompt part (labels == -100)
        # Find where labels start (first non -100 token)
        labels_mask = labels_from_batch != -100
        if labels_mask.any():
            # Find first position where label is not -100
            first_label_pos = labels_mask[0].nonzero(as_tuple=True)[0][0].item()
            # Trim input_ids to exclude the answer part
            input_ids = input_ids[:, :first_label_pos]
        else:
            # If no labels, trim to actual length using attention mask
            actual_length = attention_mask_from_batch.sum().item()
            input_ids = input_ids[:, :actual_length]
        
        # batch['images'] is a list of tensors, need to wrap it properly
        # predict.py format: images = [tensor1, tensor2, ...] (already a list)
        images = batch['images'][:1]
        
        # Convert images to correct dtype (bfloat16)
        # The batch images are in float32 by default
        images = [img.to(dtype=torch.bfloat16, device=self.device_str) for img in images]
        
        # Ensure model is in eval mode
        self.teacher_model.eval()
        self.student_model.eval()
        
        # Teacher generation (original speed)
        print(f"\nGenerating with teacher...")
        teacher_output_ids, teacher_hist = self.teacher_model.generate(
            input_ids,
            images=images,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0.1,  # Use small temperature like in predict.py
            max_new_tokens=128,  # Increase to see full output
            block_length=128,
            steps=32,
            tokenizer=self.tokenizer,
            prefix_lm=True,
            verbose=True,  # Enable verbose for history
            schedule='shift',
        )
        
        # Student generation (faster)  
        student_num_steps = max(1, self.teacher_num_inference_steps // self.num_distill_steps)
        student_output_ids, student_hist = self.student_model.generate(
            input_ids,
            images=images,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0.1,  # Use small temperature
            max_new_tokens=128,  # Increase to see full output
            block_length=128,
            steps=16,
            tokenizer=self.tokenizer,
            prefix_lm=True,
            verbose=True,  # Enable verbose for history
            schedule='shift',
        )
        
        # Debug output
        print(f"\nDEBUG OUTPUT:")
        print(f"Teacher output shape: {teacher_output_ids.shape}")
        print(f"Teacher output IDs (first 30): {teacher_output_ids[0, :30].tolist()}")
        print(f"Student output shape: {student_output_ids.shape}")
        print(f"Student output IDs (first 30): {student_output_ids[0, :30].tolist()}")
        
        teacher_text = self.tokenizer.batch_decode(teacher_output_ids, skip_special_tokens=True)
        student_text = self.tokenizer.batch_decode(student_output_ids, skip_special_tokens=True)
        
        print(f"\n{'='*50}")
        print(f"Teacher output ({self.teacher_num_inference_steps} steps):\n{teacher_text}")
        print(f"\nStudent output ({student_num_steps} steps):\n{student_text}")
        print(f"{'='*50}\n")
        
        
        # # Display generation history of the two models
        # print(f"{'='*50}\n")
        # print("Teacher history")
        # for i, v in enumerate(teacher_hist):
        #     print(i,self.tokenizer.batch_decode(v, skip_special_tokens=False)[0].lstrip('!').replace("<|mdm_mask|>",'*'))
        # print(f"{'='*50}\n")
        
        # print(f"{'='*50}\n")
        # print("Student history")
        # for i, v in enumerate(student_hist):
        #     print(i,self.tokenizer.batch_decode(v, skip_special_tokens=False)[0].lstrip('!').replace("<|mdm_mask|>",'*'))
        # print(f"{'='*50}\n")
        return None

    def configure_optimizers(self):
        """Configure optimizer for student model only"""
        optimizer = torch.optim.AdamW(
            self.student_model.parameters(),
            lr=self.config.get('learning_rate', 1e-5),
            weight_decay=self.config.get('weight_decay', 0.01)
        )
        
        # Optional: Add learning rate scheduler
        if self.config.get('use_scheduler', False):
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config.get('max_steps', 10000),
                eta_min=self.config.get('min_lr', 1e-7)
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1
                }
            }
        
        return optimizer
