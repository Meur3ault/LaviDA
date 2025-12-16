#!/usr/bin/env python3
"""
Quick test script to verify SDTT distillation setup
"""
import torch
from llada_sdtt_distill import OneRoundSDTTDistiller
from coco_captions_dataset import COCOCaptionsDataset, LLaVADataCollator
from torch.utils.data import DataLoader

def test_distiller():
    print("=" * 60)
    print("Testing SDTT Distiller Setup")
    print("=" * 60)
    
    # Minimal config for testing
    config = {
        'checkpoint_path': 'lmms-lab/llava-onevision-qwen2-0.5b-si',
        'model_name': 'llava_qwen',
        'device': 'cuda',
        'device_map': 'cuda:0',
        'torch_dtype': 'bfloat16',
        'use_fim': False,
        'teacher_num_inference_steps': 8,  # Small for testing
        'max_new_tokens': 32,
        'learning_rate': 1e-5,
        'weight_decay': 0.01,
    }
    
    print("\n1. Initializing model...")
    model = OneRoundSDTTDistiller(config)
    print("   ✓ Model initialized")
    
    print("\n2. Loading dataset (small subset)...")
    dataset = COCOCaptionsDataset(
        coco_root='data/coco',
        annotation_file='data/coco/annotations/captions_val2017.json',
        tokenizer=model.tokenizer,
        image_processor=model.image_processor,
        model_config=model.teacher_model[0].config,
        use_all_captions=False,
        image_aspect_ratio='pad'
    )
    
    # Subset for testing
    subset_indices = list(range(min(10, len(dataset))))
    dataset = torch.utils.data.Subset(dataset, subset_indices)
    print(f"   ✓ Dataset loaded: {len(dataset)} samples")
    
    print("\n3. Creating dataloader...")
    collator = LLaVADataCollator(tokenizer=model.tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=collator
    )
    print("   ✓ Dataloader created")
    
    print("\n4. Testing forward pass...")
    model.to('cuda')
    model.eval()
    
    batch = next(iter(loader))
    print(f"   Batch shapes:")
    print(f"   - input_ids: {batch['input_ids'].shape}")
    print(f"   - images: {batch['images'].shape}")
    print(f"   - labels: {batch['labels'].shape}")
    
    try:
        with torch.no_grad():
            # Test training step logic (without actual training)
            loss = model.training_step(batch, 0)
        print(f"   ✓ Forward pass successful! Loss: {loss.item():.4f}")
    except Exception as e:
        print(f"   ✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n5. Testing teacher 2-step forward...")
    try:
        with torch.no_grad():
            input_ids = batch['input_ids'][:1]
            images = batch['images'][:1]
            attention_mask = batch['attention_mask'][:1]
            
            (_, _, _, _, inputs_embeds, _, _) = \
                model.teacher_model[0].prepare_inputs_labels_for_multimodal(
                    input_ids, None, attention_mask, None, None,
                    images, ['image'], image_sizes=None, return_inputs=True
                )
            
            masked_indices = torch.ones(inputs_embeds.shape[:2], dtype=torch.bool, device='cuda')
            teacher_logits = model.teacher_2_steps_forward(inputs_embeds, masked_indices)
            print(f"   ✓ Teacher logits shape: {teacher_logits.shape}")
    except Exception as e:
        print(f"   ✗ Teacher forward failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED!")
    print("=" * 60)
    print("\nYou can now run: python train_sdtt.py")
    
    return True

if __name__ == '__main__':
    success = test_distiller()
    exit(0 if success else 1)
