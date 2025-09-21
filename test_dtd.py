"""
Script to test the DTD dataset loader
"""

import os
import sys
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np

# Add parent directory to path to import from data.datasets
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import the DTD dataset
from data.datasets.dtd import Dataset
from data.utils import augm_transforms

def test_dtd_dataset():
    print("Testing DTD dataset loader...")

    # Initialize transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])

    # Create dataset
    try:
        dataset = Dataset(transform=transform)
        print(f"Dataset loaded successfully with {len(dataset)} samples")
        print(f"Dataset classnames: {dataset.classnames[:5]}...")
        
        # Test getting a sample
        sample = dataset[0]
        print(f"Sample image shape: {sample['img'].shape}")
        print(f"Sample label: {sample['label']}")
        print(f"Sample classname: {sample['classname']}")

        # Visualize a few samples
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes = axes.flatten()
        
        for i in range(min(6, len(dataset))):
            sample = dataset[i]
            img = sample['img']
            label = sample['label']
            classname = sample['classname']
            
            # Convert tensor to numpy for visualization
            img_np = img.permute(1, 2, 0).numpy()
            # Denormalize
            img_np = img_np * np.array([0.26862954, 0.26130258, 0.27577711]) + np.array([0.48145466, 0.4578275, 0.40821073])
            img_np = np.clip(img_np, 0, 1)
            
            axes[i].imshow(img_np)
            axes[i].set_title(f"Class: {classname}")
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig('dtd_samples.png')
        print("Saved sample visualizations to dtd_samples.png")

    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dtd_dataset()