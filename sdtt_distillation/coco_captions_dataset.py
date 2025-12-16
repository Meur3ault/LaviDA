"""
COCO Captions Dataset for LLaDA Fine-tuning

This module provides a PyTorch Dataset class to load COCO Captions dataset
in the format expected by LLaVA/LLaDA training pipeline.

Usage:
    from coco_captions_dataset import COCOCaptionsDataset
    
    dataset = COCOCaptionsDataset(
        annotation_file='path/to/captions_train2017.json',
        image_folder='path/to/train2017',
        tokenizer=tokenizer,
        image_processor=image_processor,
        data_args=data_args
    )
"""

import os
import json
import copy
import random
from typing import Dict, List, Optional
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

from llava.mm_utils import process_highres_image, process_anyres_image, process_highres_image_crop_split, process_images
from llava.train.train import preprocess_multimodal, preprocess
from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle


class COCOCaptionsDataset(Dataset):
    """
    COCO Captions Dataset for LLaDA training.
    
    Converts COCO format to LLaVA conversation format:
    - Each image-caption pair becomes a conversation
    - Format: [{"from": "human", "value": "<image>\\nDescribe this image."}, 
               {"from": "gpt", "value": "caption text"}]
    
    Args:
        annotation_file: Path to COCO captions JSON file (e.g., captions_train2017.json)
        image_folder: Path to folder containing COCO images
        tokenizer: HuggingFace tokenizer
        image_processor: Image processor for preprocessing
        data_args: DataArguments object containing preprocessing configs
        max_samples: Optional limit on number of samples to load
        use_all_captions: If True, use all 5 captions per image. If False, randomly pick one.
    """
    
    def __init__(
        self,
        annotation_file: str,
        image_folder: str,
        tokenizer,
        image_processor,
        data_args,
        max_samples: Optional[int] = None,
        use_all_captions: bool = False,
        prompt_template: str = "Describe this image in detail.",
        model_config=None
    ):
        super().__init__()
        
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.data_args = data_args
        self.use_all_captions = use_all_captions
        self.prompt_template = prompt_template
        self.conv_template = "llada" 
        self.question = DEFAULT_IMAGE_TOKEN + "\n{}".format(self.prompt_template)
        self.model_config = model_config

        # Load COCO annotations
        print(f"Loading COCO annotations from {annotation_file}")
        with open(annotation_file, 'r') as f:
            coco_data = json.load(f)
        
        # Build image_id to filename mapping
        self.image_id_to_filename = {
            img['id']: img['file_name'] 
            for img in coco_data['images']
        }
        
        # Build image_id to captions mapping
        image_to_captions = {}
        for ann in coco_data['annotations']:
            image_id = ann['image_id']
            caption = ann['caption'].strip()
            if image_id not in image_to_captions:
                image_to_captions[image_id] = []
            image_to_captions[image_id].append(caption)
        
        # Create sample list
        self.samples = []
        for image_id, captions in image_to_captions.items():
            if image_id not in self.image_id_to_filename:
                continue
                
            filename = self.image_id_to_filename[image_id]
            
            if self.use_all_captions:
                # Create one sample per caption
                for caption in captions:
                    self.samples.append({
                        'image_id': image_id,
                        'filename': filename,
                        'caption': caption,
                        'all_captions': captions
                    })
            else:
                # Store all captions but will randomly pick one at __getitem__
                self.samples.append({
                    'image_id': image_id,
                    'filename': filename,
                    'captions': captions
                })
        
        # Limit samples if requested
        if max_samples is not None and max_samples < len(self.samples):
            random.shuffle(self.samples)
            self.samples = self.samples[:max_samples]
        
        print(f"Loaded {len(self.samples)} samples from COCO Captions")
        
    def __len__(self):
        return len(self.samples)
    
    @property
    def lengths(self):
        """Return approximate token lengths for each sample (for batching)"""
        length_list = []
        for sample in self.samples:
            # Rough estimate: 128 tokens for image + caption length
            if self.use_all_captions:
                caption_len = len(sample['caption'].split())
            else:
                # Average caption length
                caption_len = sum(len(c.split()) for c in sample['captions']) // len(sample['captions'])
            length_list.append(128 + caption_len)
        return length_list
    
    @property
    def modality_lengths(self):
        """Return modality-aware lengths (positive for multimodal samples)"""
        return self.lengths  # All samples are multimodal (image + text)
    
    def process_image(self, image_path: str):
        """Process image according to data_args configuration"""
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Failed to open image {image_path}. Exception: {e}")
            raise e
        
        image_size = image.size
        image_aspect_ratio = self.data_args.image_aspect_ratio
        
        if image_aspect_ratio == "highres":
            image = process_highres_image(
                image, 
                self.image_processor, 
                self.data_args.image_grid_pinpoints
            )
        elif image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
            image = process_anyres_image(
                image, 
                self.image_processor, 
                self.data_args.image_grid_pinpoints
            )
        elif image_aspect_ratio == "crop_split":
            image = process_highres_image_crop_split(image, self.data_args)
        elif image_aspect_ratio == "pad":
            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result
            
            image = expand2square(
                image, 
                tuple(int(x * 255) for x in self.image_processor.image_mean)
            )
            image = self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        else:
            # Default processing
            image = self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        
        return image, image_size, "image"

    
    def create_conversation(self, caption: str) -> Dict[str, List[Dict[str, str]]]:
        """
        Create conversation in the format expected by preprocess_llada.
        
        preprocess_llada expects format:
        {"from": "human"/"gpt", "value": ...}
        
        It will handle the tokenizer.apply_chat_template() internally.
        
        Returns:
            Dictionary with 'conversations' key containing list of message dicts
        """
        conversations = [
            {"from": "human", "value": self.question},
            {"from": "gpt", "value": caption}
        ]
        
        return {"conversations": conversations}
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            Dictionary with keys:
            - input_ids: tokenized input
            - labels: tokenized labels (with -100 for masked positions)
            - image: processed image tensor
            - attention_mask: attention mask
        """
        sample = self.samples[idx]
        
        # Get caption (randomly pick one if multiple)
        if self.use_all_captions:
            caption = sample['caption']
        else:
            caption = random.choice(sample['captions'])
        
        # Load and process image
        image_path = os.path.join(self.image_folder, sample['filename'])
        image = self.process_image(image_path)

        # Create conversation in dict format
        conversation = self.create_conversation(caption)
        
        # Use the existing preprocess pipeline from llava/train/train.py
        # This properly handles LLAMA_3 style tokenization and masking
        from llava.train.train import preprocess_multimodal, preprocess
        from llava import conversation as conversation_lib
        
        # IMPORTANT: Set default conversation to llada so preprocess() calls preprocess_llada()
        conversation_lib.default_conversation = conv_templates[self.conv_template]
        
        sources = [conversation]
        
        # Preprocess multimodal (handles image tokens)
        if self.data_args.is_multimodal:
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args
            )
        
        # Preprocess text (tokenization and label masking)
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=self.data_args.is_multimodal
        )
        
        #print("Labels: ", data_dict["labels"])
        
        if isinstance(idx, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                           labels=data_dict["labels"][0])
        
        # Add image data
        data_dict['image'] = [image]
        
        return data_dict


@dataclass
class COCOCaptionsDataCollator:
    """
    Data collator for COCO Captions dataset.
    Handles batching with proper padding.
    """
    tokenizer: any
    
    def pad_sequence(self, input_ids, batch_first, padding_value, extra_pad=-1):
        """Pad sequences with optional extra padding"""
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        
        input_ids = list(input_ids)
        max_k = max(range(len(input_ids)), key=lambda x: input_ids[x].shape[-1])
        
        if extra_pad > 0:
            extra_pad_seq = torch.tensor([padding_value] * extra_pad)
            input_ids[max_k] = torch.cat([input_ids[max_k], extra_pad_seq])
        
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, 
            batch_first=batch_first, 
            padding_value=padding_value
        )
        
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        
        return input_ids
    
    def __call__(self, instances: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate batch of instances"""
        input_ids = [instance["input_ids"] for instance in instances]
        labels = [instance["labels"] for instance in instances]
        
        # Truncate to max length
        input_ids = [
            _input_ids[:self.tokenizer.model_max_length] 
            for _input_ids in input_ids
        ]
        labels = [
            _labels[:self.tokenizer.model_max_length] 
            for _labels in labels
        ]
        
        # Set pad token if not set
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0
        
        # Random extra padding (data augmentation for length variance)
        extra_pad = np.random.randint(-128, 128)
        
        # Pad sequences
        input_ids = self.pad_sequence(
            input_ids, 
            batch_first=True, 
            padding_value=self.tokenizer.pad_token_id,
            extra_pad=extra_pad
        )
        labels = self.pad_sequence(
            labels, 
            batch_first=True, 
            padding_value=IGNORE_INDEX,
            extra_pad=extra_pad
        )
        
        # Create batch dict
        batch = dict(
            input_ids=input_ids,
            labels=labels.long() if labels.dtype == torch.int32 else labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id)
        )
        
        # Handle images
        if "image" in instances[0]:
            images = [instance["image"] for instance in instances]
            
            # Flatten image list
            batch["image_sizes"] = [im[1] for im_list in images for im in im_list]
            batch["modalities"] = [im[2] for im_list in images for im in im_list]
            images = [im[0] for im_list in images for im in im_list]
            
            batch["images"] = images
        
        return batch


def create_coco_data_module(
    annotation_file: str,
    image_folder: str,
    tokenizer,
    image_processor,
    data_args,
    max_samples: Optional[int] = None,
    use_all_captions: bool = False,
    prompt_template: str = "Describe this image in detail.",
) -> Dict:
    """
    Create dataset and data collator for COCO Captions.
    
    Args:
        annotation_file: Path to COCO annotations JSON
        image_folder: Path to COCO images folder
        tokenizer: Tokenizer instance
        image_processor: Image processor instance
        data_args: DataArguments object
        max_samples: Optional limit on samples
        use_all_captions: Whether to use all 5 captions per image
        prompt_template: Prompt template for generating captions
    
    Returns:
        Dictionary with train_dataset, eval_dataset, and data_collator
    """
    train_dataset = COCOCaptionsDataset(
        annotation_file=annotation_file,
        image_folder=image_folder,
        tokenizer=tokenizer,
        image_processor=image_processor,
        data_args=data_args,
        max_samples=max_samples,
        use_all_captions=use_all_captions,
        prompt_template=prompt_template,
    )
    
    data_collator = COCOCaptionsDataCollator(tokenizer=tokenizer)
    
    return dict(
        train_dataset=train_dataset,
        eval_dataset=None,  # Can add validation set if needed
        data_collator=data_collator
    )


# Example usage
if __name__ == "__main__":
    """
    Example of how to use this dataset with LLaDA training.
    """
    print("Example usage:")
    print("""
    from transformers import AutoTokenizer
    from llava.model.builder import load_pretrained_model
    
    # Load tokenizer and model
    tokenizer, model, image_processor, max_length = load_pretrained_model(
        model_path="path/to/llada/model",
        model_name="llava_llada",
        ...
    )
    
    # Create data module
    data_module = create_coco_data_module(
        annotation_file="path/to/annotations/captions_train2017.json",
        image_folder="path/to/train2017",
        tokenizer=tokenizer,
        image_processor=image_processor,
        data_args=your_data_args,
        max_samples=10000,  # Optional: limit samples
        use_all_captions=False,  # Use one random caption per image
    )
    
    # Use with PyTorch DataLoader
    from torch.utils.data import DataLoader
    
    train_loader = DataLoader(
        data_module['train_dataset'],
        batch_size=4,
        collate_fn=data_module['data_collator'],
        shuffle=True,
        num_workers=4,
    )
    
    # Training loop
    for batch in train_loader:
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        ...
    """)
