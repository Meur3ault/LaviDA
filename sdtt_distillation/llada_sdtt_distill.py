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
        Distiller class for one-round SDTT distillation.
        The steps should be reduced by half
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Prepare the teacher and student models
        self.checkpoint_path = config['checkpoint_path']
        self.model_name = config['model_name']
        self.device_str = config['device']
        self.device_map = config.get('device_map', 'cuda:0')
        self.torch_dtype = config.get('torch_dtype', 'bfloat16')
        self.use_fim = config.get('use_fim', False)
        
        # Calculate how many tokens the teacher will infer per step
        self.teacher_num_inference_steps = config.get('teacher_num_inference_steps', 32)
        self.max_new_tokens = config.get('max_new_tokens', 64)
        self.teacher_tokens_per_step = self.max_new_tokens // self.teacher_num_inference_steps
        self.student_tokens_per_step = self.teacher_tokens_per_step * 2  # Student does double steps

        # Remasking policy
        # Mask the tokens based on low confidence predictions from teacher
        self.remask_policy = config.get('remask_policy', 'low_confidence')

        # Hard configuration for lavida
        self.eos_id = 126081 # hack
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

        # Freeze the teacher model
        self.student_model = copy.deepcopy(self.teacher_model)
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        
        # Trick to make sure teacher_model is never in the training mode
        self.teacher_model = [self.teacher_model]

        # Move both model to device
        self.teacher_model[0].to(self.device_str)
        self.student_model.to(self.device_str)

        # Loss function
        self.loss_fn = nn.KLDivLoss(log_target=True, reduction='batchmean')

    
    def teacher_1_step_forward(self, model_inputs):
        return self.teacher_model[0](**model_inputs)

    
    def teacher_2_steps_forward(self, inputs_embeds, masked_indices):
        """Perform 2 steps of teacher forward, unmask teacher_tokens_per_step each step"""
        bsz, seq_len, hidden_dim = inputs_embeds.shape
        device = inputs_embeds.device

        # Step 1: First forward pass
        outputs_step_1 = self.teacher_model[0](
            input_ids=None,
            attention_mask=None,
            inputs_embeds=inputs_embeds,
            labels=None,
            policy='uniform',
            policy_args=None,
            return_dict=True
        )
        logits_step_1 = outputs_step_1.logits
        predicted_tokens_step_1 = torch.argmax(logits_step_1, dim=-1)
        
        # Calculate confidence for step 1
        p = F.softmax(logits_step_1, dim=-1)
        confidence = torch.gather(p, dim=-1, index=predicted_tokens_step_1.unsqueeze(-1)).squeeze(-1)
        confidence[~masked_indices] = float('-inf')

        # Select top teacher_tokens_per_step tokens to unmask in step 1
        transfer_index_step_1 = torch.zeros_like(masked_indices, dtype=torch.bool, device=device)
        for j in range(bsz):
            if masked_indices[j].sum() > 0:
                k = min(self.teacher_tokens_per_step, masked_indices[j].sum().item())
                _, select_index = torch.topk(confidence[j], k)
                transfer_index_step_1[j, select_index] = True
        
        # Update embeddings: replace selected positions with predicted embeddings
        predicted_embeds_step_1 = self.teacher_model[0].get_model().transformer.wte(predicted_tokens_step_1)
        inputs_embeds_step_2 = torch.where(
            transfer_index_step_1.unsqueeze(-1),
            predicted_embeds_step_1,
            inputs_embeds
        )

        # Step 2: Second forward pass with partially unmasked embeddings
        outputs_step_2 = self.teacher_model[0](
            input_ids=None,
            attention_mask=None,
            inputs_embeds=inputs_embeds_step_2,
            labels=None,
            policy='uniform',
            policy_args=None,
            return_dict=True
        )
        logits_step_2 = outputs_step_2.logits
        predicted_tokens_step_2 = torch.argmax(logits_step_2, dim=-1)
        
        # Calculate confidence for step 2
        p2 = F.softmax(logits_step_2, dim=-1)
        confidence2 = torch.gather(p2, dim=-1, index=predicted_tokens_step_2.unsqueeze(-1)).squeeze(-1)
        
        # Only consider remaining masked positions
        remaining_mask = masked_indices & (~transfer_index_step_1)
        confidence2[~remaining_mask] = float('-inf')

        # Select top teacher_tokens_per_step tokens to unmask in step 2
        transfer_index_step_2 = torch.zeros_like(masked_indices, dtype=torch.bool, device=device)
        for j in range(bsz):
            if remaining_mask[j].sum() > 0:
                k = min(self.teacher_tokens_per_step, remaining_mask[j].sum().item())
                _, select_index = torch.topk(confidence2[j], k)
                transfer_index_step_2[j, select_index] = True

        # Aggregate logits: combine from both steps
        final_logits = logits_step_1.clone()
        final_logits[transfer_index_step_1] = logits_step_1[transfer_index_step_1]
        final_logits[transfer_index_step_2] = logits_step_2[transfer_index_step_2]
        
        return final_logits
            
    
    def student_1_step_forward(self, model_inputs):
        return self.student_model(**model_inputs)


    def training_step(self, batch, batch_idx):
        # ============================================================================
        # STEP 1: PREPARE INPUTS - Get data from batch
        # ============================================================================
        input_ids = batch['input_ids']  # Shape: [B, L]
        images = batch['images']  # Image tensors
        labels = batch['labels']  # Shape: [B, L], -100 for non-target positions
        attention_mask = batch['attention_mask']  # Shape: [B, L]
        
        bsz = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        
        # Validate labels exist for distillation
        assert labels is not None, "Labels should not be None for distillation"
        assert labels.min() == -100, "Labels should contain -100 for ignored positions"
        
        # ============================================================================
        # STEP 2: PROCESS MULTIMODAL INPUTS - Similar to llava_llada.py line 138
        # ============================================================================
        # This step processes images and merges them with text embeddings
        # Returns: input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels, new_input_ids
        # Note: You need to call prepare_inputs_labels_for_multimodal from the model
        
        raw_input_ids = input_ids.clone()
        
        # Process multimodal inputs (images + text) to get embeddings
        # This is handled by the model's prepare_inputs_labels_for_multimodal method
        with torch.no_grad():
            (_, position_ids, attention_mask, _, inputs_embeds, labels, _) = \
                self.teacher_model[0].prepare_inputs_labels_for_multimodal(
                    input_ids, None, attention_mask, None, labels, 
                    images, ['image'], image_sizes=None, return_inputs=True
                )
        
        # Update sequence length after multimodal processing (images add tokens)
        bsz, seq_len = labels.shape
        
        # ============================================================================
        # STEP 3: CREATE LABEL MASK - Identify target tokens (llava_llada.py line 150)
        # ============================================================================
        # labels_mask: True where we want to predict tokens, False where labels == -100
        labels_mask = ~(labels == -100)  # Shape: [B, L]
        
        # Handle FIM tokens if needed (optional for COCO captioning)
        if self.use_fim:
            infill_token_pos = labels == self.fim_id  # Shape: [B, L]
        else:
            infill_token_pos = torch.zeros_like(labels, dtype=torch.bool)
        
        # ============================================================================
        # STEP 4: FORWARD PROCESS - Sample timestep and create mask pattern
        # ============================================================================
        # This follows llava_llada.py line 163
        # forward_process returns:
        #   - masked_indices: Boolean tensor [B, L] indicating which positions to mask
        #   - p_mask: Masking probability [B, 1]
        
        masked_indices, p_mask = forward_process(
            bsz, 
            seq_len, 
            device=labels.device,
            eps=1e-3,
            policy='uniform',
            policy_args=None
        )
        
        # ============================================================================
        # STEP 5: COMBINE MASKS - Only mask target tokens (llava_llada.py line 165-166)
        # ============================================================================
        # final_masked_indices: Mask only positions that are:
        #   1. Randomly selected by forward_process (masked_indices)
        #   2. Are target tokens (labels_mask)
        #   3. Are NOT FIM tokens (~infill_token_pos)
        
        final_masked_indices = masked_indices & labels_mask & (~infill_token_pos)  # [B, L]
        final_masked_indices_inv = (~masked_indices) & labels_mask & (~infill_token_pos)  # [B, L]
        
        # ============================================================================
        # STEP 6: CREATE NOISE EMBEDDINGS - Prepare [MASK] token embeddings
        # ============================================================================
        # Following llava_llada.py line 159
        # Get embedding for mask_id token and reshape for broadcasting
        
        noise_embeddings = self.teacher_model[0].get_model().transformer.wte(
            torch.tensor([self.mask_id], device=labels.device)
        )  # Shape: [1, hidden_dim]
        noise_embeddings = noise_embeddings.view(1, 1, -1)  # Shape: [1, 1, hidden_dim]
        
        # ============================================================================
        # STEP 7: APPLY MASKING - Replace masked positions with noise embeddings
        # ============================================================================
        # Following llava_llada.py line 172-173
        # Create two versions: masked and inverse-masked for data augmentation
        
        # Version 1: Mask positions where final_masked_indices is True
        inputs_embeds_masked = torch.where(
            final_masked_indices.unsqueeze(-1),  # [B, L, 1] for broadcasting
            noise_embeddings,  # [1, 1, hidden_dim]
            inputs_embeds  # [B, L, hidden_dim]
        )
        
        # Version 2: Mask positions where final_masked_indices_inv is True (complement)
        inputs_embeds_inv = torch.where(
            final_masked_indices_inv.unsqueeze(-1),
            noise_embeddings,
            inputs_embeds
        )
        
        # ============================================================================
        # STEP 8: PREPARE LABELS - Set non-masked positions to -100
        # ============================================================================
        # Following llava_llada.py line 179-182
        # We only compute loss on the masked positions
        
        labels_masked = labels.clone()
        labels_masked[~final_masked_indices] = -100  # Ignore non-masked positions
        labels_masked[labels_masked == self.fim_id] = -100  # Don't predict FIM tokens
        
        labels_inv = labels.clone()
        labels_inv[~final_masked_indices_inv] = -100
        labels_inv[labels_inv == self.fim_id] = -100
        
        # ============================================================================
        # STEP 9: CONCATENATE MASKED & INVERSE - Data augmentation
        # ============================================================================
        # Following llava_llada.py line 184-185
        # Double the batch size by including both masked versions
        
        inputs_embeds_concat = torch.cat([inputs_embeds_masked, inputs_embeds_inv], dim=0)  # [2B, L, H]
        labels_concat = torch.cat([labels_masked, labels_inv], dim=0)  # [2B, L]
        
        # ============================================================================
        # STEP 10: TEACHER FORWARD (2 STEPS) - Get teacher's predictions
        # ============================================================================
        with torch.no_grad():
            teacher_logits = self.teacher_2_steps_forward(
                inputs_embeds_concat,
                torch.cat([final_masked_indices, final_masked_indices_inv], dim=0)
            )
            
        
        # ============================================================================
        # STEP 11: STUDENT FORWARD (1 STEP) - Get student's predictions
        # ============================================================================
        # Student forward pass
        student_logits = self.student_1_step_forward({
            'input_ids': None,
            'attention_mask': attention_mask.repeat(2, 1),
            'inputs_embeds': inputs_embeds_concat,
            'labels': None,
            'policy': 'uniform',
            'policy_args': None,
            'return_dict': True
        }).logits
        
        # ============================================================================
        # STEP 12: COMPUTE DISTILLATION LOSS - KL divergence between teacher & student
        # ============================================================================
        # Create mask for valid positions
        valid_mask = (labels_concat != -100)
        
        # KL divergence loss
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
        

        # Only use the KL loss on valid positions
        kl_loss = F.kl_div(
            student_log_probs,
            teacher_log_probs,
            reduction='none',
            log_target=True
        ).sum(dim=-1)
        
        kl_loss = (kl_loss * valid_mask.float()).sum() / (valid_mask.sum() + 1e-8)
        
        self.log('train_loss', kl_loss, prog_bar=True)
        self.log('mask_ratio', final_masked_indices.float().mean(), prog_bar=True)
        
        return kl_loss

    
    def validation_step(self, batch, batch_idx):
        """Validation: compare teacher vs student generation quality"""
        if batch_idx > 0:
            return
        
        input_ids = batch['input_ids'][:1]  # Take first sample only
        images = batch['images'][:1]
        attention_mask = batch['attention_mask'][:1]
        
        # Process inputs
        with torch.no_grad():
            (_, _, _, _, inputs_embeds, _, _) = \
                self.teacher_model[0].prepare_inputs_labels_for_multimodal(
                    input_ids, None, attention_mask, None, None,
                    images, ['image'], image_sizes=None, return_inputs=True
                )
        
        # Teacher generation (original speed)
        teacher_output = llada_generate(
            self.teacher_model[0].get_model(),
            inputs_embeds=inputs_embeds,
            max_new_tokens=self.max_new_tokens,
            num_inference_steps=self.teacher_num_inference_steps,
            eos_token_id=self.eos_id
        )
        
        # Student generation (2x faster)
        student_output = llada_generate(
            self.student_model.get_model(),
            inputs_embeds=inputs_embeds,
            max_new_tokens=self.max_new_tokens,
            num_inference_steps=self.teacher_num_inference_steps // 2,
            eos_token_id=self.eos_id
        )
        
        # Decode and log
        teacher_text = self.tokenizer.decode(teacher_output[0], skip_special_tokens=True)
        student_text = self.tokenizer.decode(student_output[0], skip_special_tokens=True)
        
        print(f"\n{'='*50}")
        print(f"Teacher output ({self.teacher_num_inference_steps} steps):\n{teacher_text}")
        print(f"\nStudent output ({self.teacher_num_inference_steps // 2} steps):\n{student_text}")
        print(f"{'='*50}\n")
        
        return None



    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.student_model.parameters(),
            lr=self.config.get('learning_rate', 1e-5),
            weight_decay=self.config.get('weight_decay', 0.01)
        )
        return optimizer




            




