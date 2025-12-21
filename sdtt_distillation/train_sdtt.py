"""
Training script for SDTT Distillation on COCO Captions
"""

import sys
sys.path.append('.')

import os
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from torch.utils.data import DataLoader

from llada_sdtt_distill_v2 import OneRoundSDTTDistiller
from callbacks import SaveLaViDaCheckpoint, MemoryMonitorCallback
from coco_captions_dataset import COCOCaptionsDataset, COCOCaptionsDataCollator
from dataclasses import dataclass, field
from typing import Optional

import tempfile

# CRITICAL FIX: Change temp directory to a location with more space
# Thay '/your/large/disk/tmp' bằng thư mục có nhiều dung lượng
TEMP_DIR = '/root/autodl-tmp/LaViDa/tmp'  # Hoặc bất kỳ đâu có > 100GB free
os.makedirs(TEMP_DIR, exist_ok=True)

# Set environment variables
os.environ['TMPDIR'] = TEMP_DIR
os.environ['TEMP'] = TEMP_DIR
os.environ['TMP'] = TEMP_DIR

# Also set for Python's tempfile module
tempfile.tempdir = TEMP_DIR

print(f"✓ Temp directory set to: {TEMP_DIR}")
print(f"  Free space: {os.statvfs(TEMP_DIR).f_bavail * os.statvfs(TEMP_DIR).f_frsize / 1e9:.1f} GB")

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


def main():
    # ============================================================================
    # Configuration
    # ============================================================================
    config = {
        # Model
        'checkpoint_path': '/root/autodl-tmp/KAIST_CS632_Project/SDTT-Distillation/checkpoints/lavida-llada-v1.0-instruct',  # Teacher model checkpoint path
        #'checkpoint_path': '/root/autodl-tmp/KAIST_CS632_Project/SDTT-Distillation/LaViDa-SDTT/sdtt_distillation/outputs/sdtt_distill/lavida_checkpoints/epoch_000_loss_0.3887', # Check saved checkpoint 
        'student_checkpoint_path': '/root/autodl-tmp/KAIST_CS632_Project/LaviDA-new/lavida_pruned-modified-v2/lavida_llada_pruned_3p04B-norm2-f_r0.3-a_r0.1',  # Optional: Student model checkpoint path. If None, student will be created as copy of teacher
        #'student_checkpoint_path': '/root/autodl-tmp/KAIST_CS632_Project/LaviDA-new/lavida_pruned-modified-v2/lavida_llada_pruned_3p04B-norm2-f_r0.3-a_r0.1',  # Example: path to pruned student model
        'model_name': 'llava_llada',
        'device': 'cuda',
        'device_map': 'cuda:0',
        'torch_dtype': 'bfloat16',
        
        # Distillation
        'use_fim': False,  # COCO captioning doesn't need FIM
        'teacher_num_inference_steps': 32,
        'max_new_tokens': 64,
        'num_distill_steps': 2,  # Number of distillation steps (teacher runs num_distill_steps, student runs 1 step)
        
        # Training
        'learning_rate': 6e-5, # may be too small
        'weight_decay': 0.01,
        'batch_size': 24*2,  # Reduced for AnyRes mode to prevent OOM
        'num_workers': 4,
        'max_epochs': 15,
        'accumulate_grad_batches': 16,  # Increased to maintain effective batch size (16 * 16 = 256)
        'gradient_clip_val': 1.0,
        
        # Data
        'coco_root': '/root/autodl-tmp/KAIST_CS632_Project/Datasets/COCO_Captions',
        'annotation_file': '/root/autodl-tmp/KAIST_CS632_Project/Datasets/COCO_Captions/annotations/captions_train2017.json',
        'val_annotation_file': '/root/autodl-tmp/KAIST_CS632_Project/Datasets/COCO_Captions/annotations/captions_val2017.json',
        'use_all_captions': False,
        'image_aspect_ratio': 'pad',
        
        # Logging
        'output_dir': 'outputs/sdtt_distill',
        'save_top_k': 1,
        'log_every_n_steps': 20,

        # WandB
        'use_wandb': True,
        'wandb_project': 'lavida-sdtt-distillation',
        'wandb_name': 'sdtt_coco_captions',
        'wandb_tags': ['sdtt', 'distillation', 'lavida', 'coco'],

    }
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # ============================================================================
    # Initialize Model
    # ============================================================================
    print("Initializing SDTT Distiller...")
    print(f"Teacher checkpoint: {config['checkpoint_path']}")
    if config.get('student_checkpoint_path'):
        print(f"Student checkpoint: {config['student_checkpoint_path']}")
    else:
        print("Student model: Will be created as copy of teacher")
    
    model = OneRoundSDTTDistiller(config)
    # Get image processing config from teacher model (which handles image processing)
    model_config = model.teacher_model.config
    image_aspect_ratio = getattr(model_config, "image_aspect_ratio", "anyres")
    image_grid_pinpoints = getattr(model_config, "image_grid_pinpoints", None)
    
    print(f"Image aspect ratio: {image_aspect_ratio}")
    if image_grid_pinpoints:
        print(f"Image grid pinpoints: {image_grid_pinpoints[:2]}... (showing first 2)")

    # ============================================================================
    # Prepare Datasets
    # ============================================================================
    print("Loading COCO Captions dataset...")
    
    # Convert image_grid_pinpoints from list to string if it's a list
    if image_grid_pinpoints is not None and isinstance(image_grid_pinpoints, list):
        image_grid_pinpoints_str = str(image_grid_pinpoints)
    else:
        image_grid_pinpoints_str = image_grid_pinpoints
    
    train_data_args = TestDataArguments(
        image_folder="/root/autodl-tmp/KAIST_CS632_Project/Datasets/COCO_Captions/train2017",
        image_aspect_ratio=image_aspect_ratio,
        image_grid_pinpoints=image_grid_pinpoints_str,
    )
    train_data_args.image_processor = model.image_processor

    val_data_args = TestDataArguments(
        image_folder="/root/autodl-tmp/KAIST_CS632_Project/Datasets/COCO_Captions/val2017",
        image_aspect_ratio=image_aspect_ratio,
        image_grid_pinpoints=image_grid_pinpoints_str,
    )
    val_data_args.image_processor = model.image_processor


    train_dataset = COCOCaptionsDataset(
        annotation_file="/root/autodl-tmp/KAIST_CS632_Project/Datasets/COCO_Captions/annotations/captions_train2017.json",
        image_folder="/root/autodl-tmp/KAIST_CS632_Project/Datasets/COCO_Captions/train2017",
        tokenizer=model.tokenizer,
        image_processor=model.image_processor,
        data_args=train_data_args,
        max_samples=None, # Use all the samples
        use_all_captions=False,
        model_config=model_config
    )
    
    val_dataset = COCOCaptionsDataset(
        annotation_file="/root/autodl-tmp/KAIST_CS632_Project/Datasets/COCO_Captions/annotations/captions_val2017.json",
        image_folder="/root/autodl-tmp/KAIST_CS632_Project/Datasets/COCO_Captions/val2017",
        tokenizer=model.tokenizer,
        image_processor=model.image_processor,
        data_args=val_data_args,
        max_samples=5, # Use all the samples
        use_all_captions=False,
        model_config=model_config
    )
    
    # Data collator
    data_collator = COCOCaptionsDataCollator(tokenizer=model.tokenizer)
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        collate_fn=data_collator,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # Single sample for validation
        shuffle=True,
        num_workers=0,
        collate_fn=data_collator
    )
    
    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Val dataset: {len(val_dataset)} samples")
    
    # ============================================================================
    # Setup Callbacks & Logger
    # ============================================================================
    
    memory_callback = MemoryMonitorCallback(
        log_every_n_steps=1,
        checkpoint_dir=os.path.join(config['output_dir'], 'lavida_checkpoints'),
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(config['output_dir'], 'checkpoints'),
        filename='sdtt-{epoch:02d}-{train_loss:.4f}',
        save_top_k=config['save_top_k'],
        monitor='train/loss',
        mode='min',
        save_last=True
    )

    # Custom LaViDa format checkpoint (for development and inference)
    lavida_checkpoint_callback = SaveLaViDaCheckpoint(
        save_dir=os.path.join(config['output_dir'], 'lavida_checkpoints'),
        original_checkpoint_path=config['checkpoint_path'],
        save_every_n_epochs=3,  # Save every epoch
        save_top_k=config['save_top_k'],
        monitor='train/loss',
        mode='min'
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # Setup loggers
    loggers = []
    
    # TensorBoard logger
    tb_logger = TensorBoardLogger(
        save_dir=config['output_dir'],
        name='sdtt_logs'
    )
    loggers.append(tb_logger)
    
    # WandB logger (optional)
    if config.get('use_wandb', False):
        try:
            wandb_logger = WandbLogger(
                project=config.get('wandb_project', 'lavida-sdtt'),
                name=config.get('wandb_name', 'sdtt_distillation'),
                save_dir=config['output_dir'],
                tags=config.get('wandb_tags', ['sdtt', 'distillation']),
                config={
                    'teacher_steps': config['teacher_num_inference_steps'],
                    'student_steps': config['teacher_num_inference_steps'] // config.get('num_distill_steps', 2),
                    'num_distill_steps': config.get('num_distill_steps', 2),
                    'learning_rate': config['learning_rate'],
                    'batch_size': config['batch_size'],
                    'accumulate_grad_batches': config['accumulate_grad_batches'],
                    'effective_batch_size': config['batch_size'] * config['accumulate_grad_batches'],
                    'max_epochs': config['max_epochs'],
                    'model': config['model_name'],
                    'student_checkpoint_path': config.get('student_checkpoint_path', 'copy_of_teacher'),
                }
            )
            loggers.append(wandb_logger)
            print("✓ WandB logger initialized")
        except Exception as e:
            print(f"⚠ Could not initialize WandB: {e}")
            print("  Continuing with TensorBoard only...")
    
    # ============================================================================
    # Initialize Trainer
    # ============================================================================
    trainer = L.Trainer(
        max_epochs=config['max_epochs'],
        accelerator='gpu',
        devices=3,
        strategy='ddp_find_unused_parameters_true',  # Enable detection of unused parameters in DDP
        precision='bf16-mixed',
        accumulate_grad_batches=config['accumulate_grad_batches'],
        gradient_clip_val=config['gradient_clip_val'],
        callbacks=[memory_callback, lavida_checkpoint_callback, lr_monitor],
        enable_checkpointing=False, # For memory saving
        logger=loggers,
        log_every_n_steps=config['log_every_n_steps'],
        val_check_interval=0.5,  # Validate each 100 steps
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    # ============================================================================
    # Train
    # ============================================================================
    print("\nStarting SDTT Distillation Training...")
    num_distill_steps = config.get('num_distill_steps', 2)
    student_steps = config['teacher_num_inference_steps'] // num_distill_steps
    speedup = config['teacher_num_inference_steps'] / student_steps if student_steps > 0 else 1
    print(f"Teacher steps: {config['teacher_num_inference_steps']}")
    print(f"Student steps: {student_steps} ({speedup:.1f}x speedup, num_distill_steps={num_distill_steps})")
    print(f"Batch size: {config['batch_size']} x {config['accumulate_grad_batches']} = {config['batch_size'] * config['accumulate_grad_batches']} effective")
    print(f"Note: Reduced batch_size for AnyRes mode to prevent OOM")
    
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )
    
    print("\nTraining completed!")
    print(f"Checkpoints saved to: {config['output_dir']}/checkpoints")

if __name__ == '__main__':
    main()