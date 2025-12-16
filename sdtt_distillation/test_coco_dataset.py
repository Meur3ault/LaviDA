"""
Test script for COCO Captions Dataset

This script demonstrates how to:
1. Load the COCO Captions dataset
2. Create a DataLoader
3. Iterate through batches
4. Visualize samples
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coco_captions_dataset import COCOCaptionsDataset, COCOCaptionsDataCollator, create_coco_data_module
from llava.model.builder import load_pretrained_model


@dataclass
class TestDataArguments:
    """Data arguments for testing"""
    image_folder: str = field(default="")
    image_aspect_ratio: str = field(default="pad")
    image_grid_pinpoints: Optional[str] = field(default=None)
    is_multimodal: bool = field(default=True)
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=False)
    image_processor: Optional[any] = field(default=None)


def test_dataset_basic():
    """Test basic dataset functionality"""
    print("\n" + "="*50)
    print("Test 1: Basic Dataset Loading")
    print("="*50)
    
    # These paths need to be updated to your actual COCO data location
    annotation_file = "/home/quyennd/Data/KAIST_CS632_Project/coco2017/annotations/captions_train2017.json"
    image_folder = "/home/quyennd/Data/KAIST_CS632_Project/coco2017/train2017"
    
    # Check if files exist
    if not os.path.exists(annotation_file):
        print(f"❌ Annotation file not found: {annotation_file}")
        print("Please update the paths in this script to point to your COCO dataset.")
        return False
    
    print(f"✓ Annotation file found: {annotation_file}")
    print(f"✓ Image folder: {image_folder}")
    
    return True


def test_with_llada_model(
    model_path: str,
    annotation_file: str,
    image_folder: str,
    num_samples: int = 10,
    batch_size: int = 2,
):
    """
    Test dataset with actual LLaDA model
    
    Args:
        model_path: Path to LLaDA checkpoint
        annotation_file: Path to COCO annotations
        image_folder: Path to COCO images
        num_samples: Number of samples to test
        batch_size: Batch size for testing
    """
    print("\n" + "="*50)
    print("Test 2: Integration with LLaDA Model")
    print("="*50)
    
    # Load model components
    print("Loading LLaDA model...")
    try:
        # Load on CPU to avoid RTX 5090 compatibility issues with PyTorch 2.6
        tokenizer, model, image_processor, max_length = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name="llava_llada",
            device_map="cuda:0",  # Use CPU for testing
            torch_dtype="bfloat16"
        )
        model.to(dtype=torch.bfloat16, device="cuda:0")
        print("✓ Model loaded successfully (CUDA mode)")
        print("  Note: Using CUDA due to RTX 5090 (sm_120) incompatibility with PyTorch 2.6")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False
    
    # Create data arguments
    data_args = TestDataArguments(
        image_folder=image_folder,
        image_aspect_ratio="pad",
    )
    data_args.image_processor = image_processor
    
    # Create dataset
    print(f"\nCreating COCO dataset (max {num_samples} samples)...")
    try:
        data_module = create_coco_data_module(
            annotation_file=annotation_file,
            image_folder=image_folder,
            tokenizer=tokenizer,
            image_processor=image_processor,
            data_args=data_args,
            max_samples=num_samples,
            use_all_captions=False,
            prompt_template="Describe this image in detail.",
        )
        dataset = data_module['train_dataset']
        collator = data_module['data_collator']
        print(f"✓ Dataset created with {len(dataset)} samples")
    except Exception as e:
        print(f"❌ Failed to create dataset: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Create DataLoader
    print(f"\nCreating DataLoader (batch_size={batch_size})...")
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collator,
        shuffle=False,
        num_workers=0,  # Use 0 for debugging
    )
    print("✓ DataLoader created")
    
    # Test iteration
    print("\nTesting batch iteration...")
    try:
        for i, batch in enumerate(dataloader):
            if i >= 3:  # Test first 3 batches
                break
            
            print(f"\n--- Batch {i+1} ---")
            print(f"Input IDs shape: {batch['input_ids'].shape}")
            print(f"Labels shape: {batch['labels'].shape}")
            print(f"Attention mask shape: {batch['attention_mask'].shape}")
            print(f"Number of images: {len(batch['images'])}")
            print(f"Image sizes: {batch['image_sizes']}")
            
            # Decode first sample in batch (filter negative values)
            input_ids_sample = batch['input_ids'][0]
            input_ids_valid = input_ids_sample[input_ids_sample >= 0]
            input_text = tokenizer.decode(input_ids_valid, skip_special_tokens=False)
            
            label_ids = batch['labels'][0]
            label_ids = label_ids[label_ids != -100]  # Remove ignored tokens
            label_ids = label_ids[label_ids >= 0]  # Remove negative values
            label_text = tokenizer.decode(label_ids, skip_special_tokens=True)
            
            print(f"\nSample 1 in batch:")
            print(f"Input (first 200 chars): {input_text[:200]}...")
            print(f"Label: {label_text}")

            
            # Test forward pass
            print("\nTesting forward pass...")
            try:
                # Move batch to device and cast to bfloat16
                batch_gpu = {}
                for k, v in batch.items():
                    if k == 'images':
                        # Images is a list of tensors - cast each one
                        batch_gpu[k] = [img.to(dtype=torch.bfloat16, device=model.device) for img in v]
                    elif isinstance(v, torch.Tensor):
                        # For input_ids, labels, attention_mask - keep as long/int
                        if v.dtype in [torch.long, torch.int, torch.int32, torch.int64]:
                            batch_gpu[k] = v.to(device=model.device)
                        else:
                            batch_gpu[k] = v.to(dtype=torch.bfloat16, device=model.device)
                    else:
                        batch_gpu[k] = v
                
                # Add masking parameters for LLaDA masked diffusion
                # Sample a masking ratio (0.0 = no masking, 1.0 = full masking)
                import random
                mask_ratio = random.uniform(0.3, 0.7)  # Mask 30-70% of tokens
                batch_gpu['policy'] = 'uniform'  # Uniform masking policy
                batch_gpu['policy_args'] = {'ratio': mask_ratio}  # Masking ratio in policy_args
                
                print(f"  Testing with mask_ratio={mask_ratio:.2f}")
                
                # Forward pass
                with torch.no_grad():
                    outputs = model(**batch_gpu)
                
                print(f"✓ Forward pass successful")
                print(f"  Loss: {outputs['loss'].item():.4f}")
                if 'logits' in outputs:
                    print(f"  Logits shape: {outputs['logits'].shape}")
                
                # Analyze masking statistics
                if 'final_masked_indices' in outputs:
                    masked_indices = outputs['final_masked_indices'][0]  # First sample
                    num_masked = masked_indices.sum().item()
                    total_tokens = masked_indices.numel()
                    actual_mask_ratio = num_masked / total_tokens
                    print(f"  Masked tokens: {num_masked}/{total_tokens} ({actual_mask_ratio:.2%})")
                
                if 'p_mask' in outputs:
                    print(f"  p_mask value: {outputs['p_mask']}")

                # Show masked vs original tokens
                if "new_input_ids" in outputs:
                    print("\n  Analyzing masking results:")
                    orig_ids = batch_gpu['input_ids'][0]
                    new_ids = outputs["new_input_ids"][0]
                    
                    print(f"  Original input_ids length: {len(orig_ids)}")
                    print(f"  New input_ids length: {len(new_ids)}")
                    
                    # Use the shorter length for comparison
                    compare_len = min(len(orig_ids), len(new_ids))
                    
                    # Find positions where tokens were changed
                    changed_positions = (orig_ids[:compare_len] != new_ids[:compare_len]).nonzero(as_tuple=True)[0]
                    if len(changed_positions) > 0:
                        print(f"  Tokens changed at {len(changed_positions)} positions")
                        # Show a few examples
                        for idx, pos in enumerate(changed_positions[:5]):
                            pos = pos.item()
                            orig_token = tokenizer.decode([orig_ids[pos].item()], skip_special_tokens=False)
                            new_token = tokenizer.decode([new_ids[pos].item()], skip_special_tokens=False) if new_ids[pos] >= 0 else "[MASKED]"
                            print(f"    Position {pos}: '{orig_token}' -> '{new_token}'")
                    else:
                        print("  No tokens were masked (all unchanged)")
                    
                    # Decode and show full sequences
                    print("\n  Full Original Input:")
                    orig_valid = orig_ids[orig_ids >= 0]
                    orig_text = tokenizer.decode(orig_valid, skip_special_tokens=False)
                    print(f"  {orig_text}")
                    
                    print("\n  Full Masked Input (new_input_ids):")
                    new_valid = new_ids[new_ids >= 0]
                    new_text = tokenizer.decode(new_valid, skip_special_tokens=False)
                    print(f"  {new_text}")
                    
                    # Show labels
                    print("\n  Labels (target tokens to predict):")
                    label_ids = batch_gpu['labels'][0]
                    label_valid = label_ids[(label_ids != -100) & (label_ids >= 0)]
                    label_text = tokenizer.decode(label_valid, skip_special_tokens=False)
                    print(f"  {label_text}")
                
                # Visualize the outputs after the first step
                print(f"\n  Output keys: {list(outputs.keys())}")
                
            except Exception as e:
                print(f"❌ Forward pass failed: {e}")
                import traceback
                traceback.print_exc()
        
        print("\n✓ All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Batch iteration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_only(
    annotation_file: str,
    image_folder: str,
    num_samples: int = 5,
):
    """
    Test dataset without model (for quick validation)
    """
    print("\n" + "="*50)
    print("Test 3: Dataset Only (No Model)")
    print("="*50)
    
    from transformers import AutoTokenizer
    from transformers import CLIPImageProcessor
    
    # Load tokenizer (using a public model for testing)
    print("Loading tokenizer...")
    try:
        # Try Llama-3 tokenizer (has chat_template for LLAMA_3 style)
        # If not available, fall back to gpt2
        try:
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
            print("✓ Tokenizer loaded (Llama-3)")
        except:
            # Fallback: Load a tokenizer and manually set chat_template
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            # Set Llama-3 style chat template
            tokenizer.chat_template = (
                "{% set loop_messages = messages %}"
                "{% for message in loop_messages %}"
                "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'+ message['content'] | trim + '<|eot_id|>' %}"
                "{% if loop.index0 == 0 %}"
                "{% set content = bos_token + content %}"
                "{% endif %}"
                "{{ content }}"
                "{% endfor %}"
                "{% if add_generation_prompt %}"
                "{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}"
                "{% endif %}"
            )
            # Add special tokens
            special_tokens = {
                "bos_token": "<|startoftext|>",
                "eos_token": "<|eot_id|>",
                "pad_token": "<|finetune_right_pad_id|>",
                "additional_special_tokens": [
                    "<|start_header_id|>",
                    "<|end_header_id|>",
                    "<|eot_id|>",
                    "<|startoftext|>",
                    "<|finetune_right_pad_id|>",
                ]
            }
            tokenizer.add_special_tokens(special_tokens)
            print("✓ Tokenizer loaded (gpt2 with Llama-3 chat template)")
            
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"❌ Failed to load tokenizer: {e}")
        return False
    
    image_processor = CLIPImageProcessor()
    
    # Create minimal data args
    data_args = TestDataArguments(
        image_folder=image_folder,
        image_aspect_ratio="pad",
    )
    data_args.image_processor = image_processor
    
    # Create dataset
    print(f"\nCreating dataset with {num_samples} samples...")
    try:
        dataset = COCOCaptionsDataset(
            annotation_file=annotation_file,
            image_folder=image_folder,
            tokenizer=tokenizer,
            image_processor=image_processor,
            data_args=data_args,
            max_samples=num_samples,
            use_all_captions=False,
        )
        print(f"✓ Dataset created with {len(dataset)} samples")
    except Exception as e:
        print(f"❌ Failed to create dataset: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test getting samples
    print("\nTesting sample retrieval...")
    try:
        for i in range(min(3, len(dataset))):
            print(f"\n--- Sample {i+1} ---")
            sample = dataset[i]
            
            print(f"Keys: {sample.keys()}")
            print(f"Input IDs shape: {sample['input_ids'].shape}")
            print(f"Labels shape: {sample['labels'].shape}")
            print(f"Number of images: {len(sample['image'])}")
            
            # Decode (filter negative values used for special tokens)
            input_ids_valid = sample['input_ids'][sample['input_ids'] >= 0]
            input_text = tokenizer.decode(input_ids_valid, skip_special_tokens=False)
            label_ids = sample['labels'][sample['labels'] != -100]
            label_ids = label_ids[label_ids >= 0]  # Filter negative values
            label_text = tokenizer.decode(label_ids, skip_special_tokens=True)
            
            print(f"Input preview: {input_text[:150]}...")
            print(f"Label: {label_text}")

        
        print("\n✓ Sample retrieval test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Sample retrieval failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test COCO Captions Dataset")
    parser.add_argument(
        "--annotation_file",
        type=str,
        default="/home/quyennd/Data/KAIST_CS632_Project/coco2017/annotations/captions_train2017.json",
        help="Path to COCO annotations JSON file"
    )
    parser.add_argument(
        "--image_folder",
        type=str,
        default="/home/quyennd/Data/KAIST_CS632_Project/coco2017/train2017",
        help="Path to COCO images folder"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/quyennd/Data/KAIST_CS632_Project/sdtt_orig/lavida_llada_checkpoints/lavida-llada-v1.0-instruct",
        help="Path to LLaDA model checkpoint (optional)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of samples to test"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for testing"
    )
    parser.add_argument(
        "--test_mode",
        type=str,
        choices=["basic", "dataset", "full"],
        default="dataset",
        help="Test mode: basic (check files), dataset (test dataset only), full (test with model)"
    )
    
    args = parser.parse_args()
    
    print("COCO Captions Dataset Test")
    print("=" * 50)
    print(f"Annotation file: {args.annotation_file}")
    print(f"Image folder: {args.image_folder}")
    print(f"Test mode: {args.test_mode}")
    
    if args.test_mode == "basic":
        test_dataset_basic()
    
    elif args.test_mode == "dataset":
        success = test_dataset_only(
            annotation_file=args.annotation_file,
            image_folder=args.image_folder,
            num_samples=args.num_samples,
        )
        if success:
            print("\n✅ All dataset tests passed!")
        else:
            print("\n❌ Some tests failed.")
            sys.exit(1)
    
    elif args.test_mode == "full":
        if args.model_path is None:
            print("❌ Error: --model_path is required for full test mode")
            sys.exit(1)
        
        success = test_with_llada_model(
            model_path=args.model_path,
            annotation_file=args.annotation_file,
            image_folder=args.image_folder,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
        )
        if success:
            print("\n✅ All tests passed!")
        else:
            print("\n❌ Some tests failed.")
            sys.exit(1)
