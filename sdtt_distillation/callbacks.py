"""
Custom Lightning Callback to save student model in LaViDa checkpoint format.

This callback saves only the student model weights in the same format as the original
LaViDa checkpoint, ensuring compatibility with other components in the project.
"""


import json
import shutil
from pathlib import Path
from typing import Dict, Any
import torch
from lightning.pytorch.callbacks import Callback
from safetensors.torch import save_file, save_model
import shutil
import psutil

class SaveLaViDaCheckpoint(Callback):
    """
    Save student model in LaViDa checkpoint format using safetensors.
    
    This callback:
    1. Extracts student model state dict
    2. Saves it in safetensors format (safer and faster than .bin)
    3. Copies config files and tokenizer files for compatibility
    """
    
    def __init__(
        self,
        save_dir: str,
        original_checkpoint_path: str,
        save_every_n_epochs: int = 1,
        save_top_k: int = 3,
        monitor: str = "train/loss",
        mode: str = "min",
        max_shard_size: str = "5GB"  # Maximum size per shard
    ):
        """
        Args:
            save_dir: Directory to save checkpoints
            original_checkpoint_path: Path to original LaViDa checkpoint (for config files)
            save_every_n_epochs: Save checkpoint every N epochs
            save_top_k: Keep only top K checkpoints
            monitor: Metric to monitor for best checkpoint
            mode: 'min' or 'max' for monitoring metric
            max_shard_size: Maximum size per shard (e.g., "5GB", "2GB")
        """
        super().__init__()
        self.save_dir = Path(save_dir)
        self.original_checkpoint_path = Path(original_checkpoint_path)
        self.save_every_n_epochs = save_every_n_epochs
        self.save_top_k = save_top_k
        self.monitor = monitor
        self.mode = mode
        self.max_shard_size = self._parse_size(max_shard_size)
        
        # Track saved checkpoints
        self.saved_checkpoints = []
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        
        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def _parse_size(self, size_str: str) -> int:
        """Parse size string like '5GB' to bytes"""
        size_str = size_str.upper()
        if 'GB' in size_str:
            return int(size_str.replace('GB', '')) * 1024 * 1024 * 1024
        elif 'MB' in size_str:
            return int(size_str.replace('MB', '')) * 1024 * 1024
        else:
            return int(size_str)
    
    def _should_save(self, current_epoch: int) -> bool:
        """Check if should save checkpoint at current epoch"""
        return (current_epoch + 1) % self.save_every_n_epochs == 0
    
    def _is_better(self, current_metric: float) -> bool:
        """Check if current metric is better than best metric"""
        if self.mode == 'min':
            return current_metric < self.best_metric
        else:
            return current_metric > self.best_metric
    
    def _get_state_dict_shards(self, state_dict: Dict[str, torch.Tensor]) -> list:
        """
        Split state dict into shards based on max_shard_size.
        
        Returns:
            List of (shard_dict, shard_size) tuples
        """
        shards = []
        current_shard = {}
        current_size = 0
        
        for key, tensor in state_dict.items():
            tensor_size = tensor.numel() * tensor.element_size()
            
            # If adding this tensor exceeds max shard size, start new shard
            if current_size + tensor_size > self.max_shard_size and current_shard:
                shards.append((current_shard, current_size))
                current_shard = {}
                current_size = 0
            
            current_shard[key] = tensor
            current_size += tensor_size
        
        # Add last shard
        if current_shard:
            shards.append((current_shard, current_size))
        
        return shards
    
    def _save_checkpoint(self, trainer, pl_module, epoch: int, metric_value: float):
        """Save checkpoint in LaViDa format using safetensors"""
        
        # Create checkpoint directory
        checkpoint_name = f"epoch_{epoch:03d}_loss_{metric_value:.4f}"
        checkpoint_dir = self.save_dir / checkpoint_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Saving LaViDa checkpoint (safetensors): {checkpoint_name}")
        print(f"{'='*60}")
        
        # 1. Get student model state dict
        student_model = pl_module.student_model
        student_model.resize_token_embeddings(126464)
        student_model.tie_weights()
        state_dict = student_model.state_dict()
        
        # Convert all tensors to contiguous memory layout (required by safetensors)
        state_dict = {k: v.contiguous() for k, v in state_dict.items()}
        
        # Calculate total size
        total_size = sum(tensor.numel() * tensor.element_size() 
                        for tensor in state_dict.values())
        total_size_gb = total_size / (1024**3)
        
        print(f"  Total model size: {total_size_gb:.2f} GB")
        
        # 2. Split into shards and save
        shards = self._get_state_dict_shards(state_dict)
        num_shards = len(shards)
        
        print(f"  Splitting into {num_shards} shard(s)")
        
        # Create weight map for index file
        weight_map = {}
        
        for shard_idx, (shard_dict, shard_size) in enumerate(shards):
            shard_filename = f"model-{shard_idx+1:05d}-of-{num_shards:05d}.safetensors"
            shard_path = checkpoint_dir / shard_filename
            
            # Save shard using safetensors
            # Metadata can include dtype, shape info, etc.
            metadata = {
                "format": "pt",
                "shard": f"{shard_idx+1}/{num_shards}",
                "total_size": str(total_size)
            }
            
            save_file(shard_dict, shard_path, metadata=metadata)
            
            shard_size_gb = shard_size / (1024**3)
            print(f"  Saved shard {shard_idx+1}/{num_shards}: {shard_filename} ({shard_size_gb:.2f} GB)")
            
            # Update weight map
            for key in shard_dict.keys():
                weight_map[key] = shard_filename
        
        # 3. Save index file (model.safetensors.index.json)
        index_dict = {
            "metadata": {
                "total_size": total_size,
                "format": "safetensors"
            },
            "weight_map": weight_map
        }
        
        index_path = checkpoint_dir / "model.safetensors.index.json"
        with open(index_path, 'w') as f:
            json.dump(index_dict, f, indent=2)
        print(f"  Saved index: model.safetensors.index.json")
        
        # 4. Copy config files from original checkpoint
        config_files = [
            "config.json",
            "generation_config.json",
            "special_tokens_map.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "modeling_llada.py"
        ]
        
        for config_file in config_files:
            src_path = self.original_checkpoint_path / config_file
            if src_path.exists():
                dst_path = checkpoint_dir / config_file
                shutil.copy2(src_path, dst_path)
                print(f"  Copied: {config_file}")
        
        # 5. Copy vocab files
        for vocab_file in ["merges.txt", "vocab.txt", "vocab.json"]:
            src_path = self.original_checkpoint_path / vocab_file
            if src_path.exists():
                dst_path = checkpoint_dir / vocab_file
                shutil.copy2(src_path, dst_path)
                print(f"  Copied: {vocab_file}")
        
        # 6. Update config.json with distillation metadata
        config_path = checkpoint_dir / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Add distillation metadata
            config['_distillation_metadata'] = {
                'method': 'SDTT',
                'teacher_steps': pl_module.teacher_num_inference_steps,
                'student_steps': pl_module.teacher_num_inference_steps // pl_module.num_distill_steps,
                'original_checkpoint': str(self.original_checkpoint_path),
                'epoch': epoch,
                'loss': float(metric_value),
                'format': 'safetensors'
            }
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"  Updated config with distillation metadata")
        
        # 7. Save training info
        training_info = {
            'epoch': epoch,
            'global_step': trainer.global_step,
            'loss': float(metric_value),
            'learning_rate': trainer.optimizers[0].param_groups[0]['lr'] if trainer.optimizers else 0.0,
            'distillation_config': {
                'teacher_num_inference_steps': pl_module.teacher_num_inference_steps,
                'num_distill_steps': pl_module.num_distill_steps,
                'distill_mode': pl_module.distill_mode,
                'loss_precision': pl_module.loss_precision,
            },
            'checkpoint_info': {
                'format': 'safetensors',
                'num_shards': num_shards,
                'total_size_bytes': total_size,
                'total_size_gb': total_size_gb
            }
        }
        
        training_info_path = checkpoint_dir / "training_info.json"
        with open(training_info_path, 'w') as f:
            json.dump(training_info, f, indent=2)
        print(f"  Saved training info")
        
        print(f"{'='*60}\n")
        
        # Track checkpoint
        self.saved_checkpoints.append({
            'path': checkpoint_dir,
            'epoch': epoch,
            'metric': metric_value
        })
        
        # Remove old checkpoints if exceeding save_top_k
        # if len(self.saved_checkpoints) > self.save_top_k:
        #     # Sort by metric
        #     self.saved_checkpoints.sort(
        #         key=lambda x: x['metric'],
        #         reverse=(self.mode == 'max')
        #     )
            
        #     # Remove worst checkpoint
        #     worst = self.saved_checkpoints.pop()
        #     if worst['path'].exists():
        #         shutil.rmtree(worst['path'])
        #         print(f"Removed old checkpoint: {worst['path'].name}")
        
        # after saving the model, return the shape of the embedding layer
        pl_module.student_model.resize_token_embeddings(len(pl_module.tokenizer))
        pl_module.student_model.tie_weights()
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Called at the end of each training epoch"""
        current_epoch = trainer.current_epoch
        
        # Check if should save
        if not self._should_save(current_epoch):
            return
        
        # Get metric value
        if self.monitor in trainer.callback_metrics:
            metric_value = trainer.callback_metrics[self.monitor].item()
        else:
            # Fallback to train/loss
            metric_value = trainer.callback_metrics.get('train/loss', float('inf')).item()
        
        # Update best metric
        if self._is_better(metric_value):
            self.best_metric = metric_value
        
        # Save checkpoint
        self._save_checkpoint(trainer, pl_module, current_epoch, metric_value)
    
    def on_train_end(self, trainer, pl_module):
        """Called at the end of training - save final checkpoint"""
        print("\nSaving final checkpoint...")
        
        # Get final metric
        if self.monitor in trainer.callback_metrics:
            metric_value = trainer.callback_metrics[self.monitor].item()
        else:
            metric_value = trainer.callback_metrics.get('train/loss', float('inf')).item()
        
        # Save final checkpoint
        self._save_checkpoint(trainer, pl_module, trainer.current_epoch, metric_value)
        
        print(f"\nTraining completed! Checkpoints saved to: {self.save_dir}")
        print(f"Best metric ({self.monitor}): {self.best_metric:.4f}")


class MemoryMonitorCallback(Callback):
    def __init__(self, log_every_n_steps=10, checkpoint_dir=None):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.checkpoint_dir = checkpoint_dir
        
    def _log_memory(self, stage=""):
        # RAM usage
        ram = psutil.virtual_memory()
        ram_used_gb = ram.used / (1024**3)
        ram_total_gb = ram.total / (1024**3)
        ram_percent = ram.percent
        
        # GPU usage
        if torch.cuda.is_available():
            gpu_allocated = torch.cuda.memory_allocated() / (1024**3)
            gpu_reserved = torch.cuda.memory_reserved() / (1024**3)
            gpu_free = (torch.cuda.get_device_properties(0).total_memory - 
                       torch.cuda.memory_allocated()) / (1024**3)
        else:
            gpu_allocated = gpu_reserved = gpu_free = 0
        
        # Disk space
        if self.checkpoint_dir:
            disk = shutil.disk_usage(self.checkpoint_dir)
            disk_free_gb = disk.free / (1024**3)
            disk_percent = (disk.used / disk.total) * 100
        else:
            disk_free_gb = 0
            disk_percent = 0
        
        msg = (
            f"\n{'='*70}\n"
            f"[{stage}] Memory Status:\n"
            f"  RAM:  {ram_used_gb:.2f}/{ram_total_gb:.2f} GB ({ram_percent:.1f}%)\n"
            f"  GPU:  {gpu_allocated:.2f} GB allocated, {gpu_reserved:.2f} GB reserved, "
            f"{gpu_free:.2f} GB free\n"
        )
        
        if self.checkpoint_dir:
            msg += f"  DISK: {disk_free_gb:.2f} GB free ({disk_percent:.1f}% used)\n"
        
        msg += f"{'='*70}"
        
        # Warning if memory is critical
        if ram_percent > 90:
            msg += f"\n⚠️  WARNING: RAM usage is very high ({ram_percent:.1f}%)!"
        if disk_free_gb < 5 and self.checkpoint_dir:
            msg += f"\n⚠️  WARNING: Low disk space ({disk_free_gb:.2f} GB free)!"
        if gpu_free < 1 and torch.cuda.is_available():
            msg += f"\n⚠️  WARNING: Low GPU memory ({gpu_free:.2f} GB free)!"
        
        print(msg)
        
    def on_train_start(self, trainer, pl_module):
        print("\n🚀 Training started - Initial memory status:")
        self._log_memory("TRAIN START")
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step % self.log_every_n_steps == 0:
            self._log_memory(f"Step {trainer.global_step}")
    
    def on_train_epoch_end(self, trainer, pl_module):
        self._log_memory(f"Epoch {trainer.current_epoch} END")
    
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        print(f"\n💾 Saving checkpoint at step {trainer.global_step}")
        self._log_memory("CHECKPOINT")