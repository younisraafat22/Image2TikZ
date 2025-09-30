#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simplified data processing utilities for knowledge distillation training.
"""

import torch
from torch.utils.data import Dataset
from datasets import load_dataset


# Constants
IGNORE_INDEX = -100


def preprocess(texts, tokenizer, patch_token, num_patches, return_tensors="pt"):
    """Simple preprocessing for TikZ generation with image patches."""
    if isinstance(texts, str):
        texts = [texts]
    
    patch_token_id = tokenizer.convert_tokens_to_ids(patch_token)
    
    all_input_ids = []
    all_labels = []
    
    for text in texts:
        # Tokenize text
        text_tokens = tokenizer(
            text,
            truncation=True,
            max_length=2048 - num_patches,
            return_tensors="pt",
            add_special_tokens=True
        )
        
        # Add patch tokens at beginning
        patch_token_ids = torch.tensor([patch_token_id] * num_patches, dtype=torch.long).unsqueeze(0)
        full_input_ids = torch.cat([patch_token_ids, text_tokens['input_ids']], dim=1)
        
        # Create labels (mask patch tokens)
        labels = full_input_ids.clone()
        labels[0, :num_patches] = IGNORE_INDEX
        
        all_input_ids.append(full_input_ids[0])
        all_labels.append(labels[0])
    
    # Handle single vs multiple texts
    if len(all_input_ids) == 1:
        return {
            'input_ids': all_input_ids[0].unsqueeze(0),
            'labels': all_labels[0].unsqueeze(0),
        }
    else:
        input_ids = torch.nn.utils.rnn.pad_sequence(
            all_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            all_labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        return {'input_ids': input_ids, 'labels': labels}


class KnowledgeDistillationDataset(Dataset):
    """Simplified dataset for knowledge distillation."""
    
    def __init__(self, dataset, training_tokenizer, model_config, dtype=torch.bfloat16):
        self.dataset = dataset
        self.dtype = dtype
        
        # Extract tokenizers based on type - handle nested structure
        if hasattr(training_tokenizer, 'text') and hasattr(training_tokenizer, 'image'):
            # This is likely a DetikzifyProcessor
            self.text_tokenizer = training_tokenizer.text
            self.image_processor = training_tokenizer.image
            
            # Check if text_tokenizer is also a processor with nested tokenizer
            if hasattr(self.text_tokenizer, 'tokenizer'):
                self.text_tokenizer = self.text_tokenizer.tokenizer
            elif hasattr(self.text_tokenizer, 'text_tokenizer'):
                self.text_tokenizer = self.text_tokenizer.text_tokenizer
        else:
            self.text_tokenizer = training_tokenizer
            self.image_processor = None
        
        
        # Try to get patch token with error handling
        try:
            self.patch_token = self.text_tokenizer.convert_ids_to_tokens(model_config.patch_token_id)
        except AttributeError:
            if hasattr(self.text_tokenizer, 'tokenizer'):
                self.text_tokenizer = self.text_tokenizer.tokenizer
                self.patch_token = self.text_tokenizer.convert_ids_to_tokens(model_config.patch_token_id)
            else:
                raise
        
        self.num_patches = model_config.num_patches
        
        # Simple fallback for images
        from torchvision import transforms
        self.fallback_transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if idx >= len(self.dataset):
            raise IndexError(f"Index {idx} out of range for dataset with {len(self.dataset)} items")
        
        item = self.dataset[idx]
        
        # Process image
        image = item.get('image')
        if self.image_processor and image:
            try:
                processed_image = self.image_processor(image)
                if isinstance(processed_image, dict):
                    processed_image = processed_image["pixel_values"]
                if processed_image.dim() == 4:
                    processed_image = processed_image[0]
                processed_image = processed_image.to(dtype=self.dtype)
            except:
                processed_image = self.fallback_transform(image).to(dtype=self.dtype)
        else:
            # Create dummy image if needed
            from PIL import Image
            import numpy as np
            dummy = Image.fromarray(np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8))
            processed_image = self.fallback_transform(dummy).to(dtype=self.dtype)
        
        # Process text
        text = item.get('code', '') or item.get('text', '')
        if not text:
            text = "\\begin{tikzpicture}\\draw (0,0) circle (1cm);\\end{tikzpicture}"
        
        # Tokenize
        tokenized = preprocess(
            texts=text,
            tokenizer=self.text_tokenizer,
            patch_token=self.patch_token,
            num_patches=self.num_patches,
        )
        
        return {
            "input_ids": tokenized["input_ids"][0],
            "labels": tokenized["labels"][0],
            "images": processed_image,
            "text": text
        }


class KnowledgeDistillationCollator:
    """Simple data collator."""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def __call__(self, batch):
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [item['input_ids'] for item in batch], 
            batch_first=True, 
            padding_value=self.tokenizer.pad_token_id
        )
        
        labels = torch.nn.utils.rnn.pad_sequence(
            [item['labels'] for item in batch], 
            batch_first=True, 
            padding_value=IGNORE_INDEX
        )
        
        images = torch.stack([item['images'] for item in batch])
        
        return {'input_ids': input_ids, 'labels': labels, 'images': images}


def create_dummy_dataset(num_samples=1000, max_length=512):
    """Create a dummy dataset for testing."""
    from PIL import Image
    import numpy as np
    
    examples = []
    for i in range(num_samples):
        dummy_image = Image.fromarray(np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8))
        tikz_code = f"\\begin{{tikzpicture}}\\draw (0,0) circle ({i % 5 + 1}cm);\\end{{tikzpicture}}"
        examples.append({'image': dummy_image, 'code': tikz_code})
    
    from datasets import Dataset
    return Dataset.from_list(examples)


def create_detikzify_dataset_with_split(dataset_name="nllg/datikz-v2", split="train", max_samples=None, max_length=512):
    """
    Load a specific split of dataset for knowledge distillation.
    
    Args:
        dataset_name: Dataset to load
        split: Split to load (e.g., 'train', 'test')
        max_samples: Maximum samples (None for all samples)
        max_length: Maximum sequence length
    """
    print(f"Loading dataset: {dataset_name} (split: {split})")
    if max_samples is not None:
        print(f"  Max samples: {max_samples:,}")
    else:
        print(f"  Max samples: all")
    print(f"  Max length: {max_length}")
    
    try:
        # Load dataset with specific split
        dataset = load_dataset(dataset_name)
        
        if split not in dataset:
            available_splits = list(dataset.keys())
            print(f"Split '{split}' not found. Available: {available_splits}")
            if "train" in dataset:
                split = "train"
            else:
                split = available_splits[0]
                print(f"Using split: '{split}'")
        
        dataset_split = dataset[split]
        
        # Limit samples if specified
        if max_samples and len(dataset_split) > max_samples:
            print(f"  Limited to {max_samples:,} samples")
            dataset_split = dataset_split.select(range(max_samples))
        print(f"  Size: {len(dataset_split):,} samples")
        return dataset_split
    except Exception as e:
        print(f"Error loading dataset: {e}; using dummy data")
        return create_dummy_dataset(max_samples or 1000, max_length)


