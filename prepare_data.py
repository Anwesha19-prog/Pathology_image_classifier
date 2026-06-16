"""
Data Preparation Script for Breast Cancer Classifier

This script helps organize the BreakHis dataset into the required folder structure.

Usage:
    python prepare_data.py --source /path/to/breakhis --magnification 400X

Author: Anwesha Sarkar
Date: June 2026
"""

import os
import shutil
from pathlib import Path
import argparse
from sklearn.model_selection import train_test_split
import random

def organize_breakhis_dataset(source_path, target_path, magnification='400X', valid_split=0.2):
    """
    Organize BreakHis dataset into train/valid folders
    
    Args:
        source_path: Path to downloaded BreakHis dataset
        target_path: Path to organized dataset (data folder)
        magnification: Which magnification to use (40X, 100X, 200X, 400X)
        valid_split: Fraction of data to use for validation
    """
    
    source = Path(source_path)
    target = Path(target_path)
    
    print("=" * 60)
    print("📁 BreakHis Dataset Organizer")
    print("=" * 60)
    print(f"\nSource: {source.absolute()}")
    print(f"Target: {target.absolute()}")
    print(f"Magnification: {magnification}")
    print(f"Validation split: {valid_split * 100}%")
    
    # Create target directories
    for split in ['train', 'valid']:
        for class_name in ['benign', 'malignant']:
            (target / split / class_name).mkdir(parents=True, exist_ok=True)
    
    # BreakHis structure: BreaKHis_v1/histology_slides/breast/{benign|malignant}/SOB/...
    benign_path = source / 'histology_slides' / 'breast' / 'benign' / 'SOB'
    malignant_path = source / 'histology_slides' / 'breast' / 'malignant' / 'SOB'
    
    # Alternative structure (if different)
    if not benign_path.exists():
        benign_path = source / 'benign'
        malignant_path = source / 'malignant'
    
    if not benign_path.exists() and not malignant_path.exists():
        print("\n❌ Error: Could not find benign/malignant folders in source path")
        print("Please check the source path and dataset structure")
        return
    
    # Process each class
    for class_name, class_path in [('benign', benign_path), ('malignant', malignant_path)]:
        if not class_path.exists():
            print(f"\n⚠️ Warning: {class_name} folder not found at {class_path}")
            continue
        
        print(f"\n📊 Processing {class_name} images...")
        
        # Find all images with specified magnification
        # First, get all image files
        all_image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            all_image_files.extend(list(class_path.rglob(ext)))
        
        # Filter by magnification
        image_files = [f for f in all_image_files if magnification in f.name]
        
        if len(image_files) == 0:
            print(f"⚠️ No images found for {magnification} magnification")
            print(f"Using all available images instead...")
            image_files = all_image_files
        
        print(f"Found {len(image_files)} images")
        
        # Split into train and validation
        train_files, valid_files = train_test_split(
            image_files, 
            test_size=valid_split, 
            random_state=42
        )
        
        # Copy files
        print(f"Copying {len(train_files)} training images...")
        for i, img_file in enumerate(train_files):
            target_file = target / 'train' / class_name / f"{class_name}_{i:04d}{img_file.suffix}"
            shutil.copy2(img_file, target_file)
            if (i + 1) % 100 == 0:
                print(f"  Copied {i + 1}/{len(train_files)} files...")
        
        print(f"Copying {len(valid_files)} validation images...")
        for i, img_file in enumerate(valid_files):
            target_file = target / 'valid' / class_name / f"{class_name}_{i:04d}{img_file.suffix}"
            shutil.copy2(img_file, target_file)
            if (i + 1) % 100 == 0:
                print(f"  Copied {i + 1}/{len(valid_files)} files...")
        
        print(f"✅ {class_name}: {len(train_files)} train, {len(valid_files)} valid")
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ Dataset organization complete!")
    print("=" * 60)
    
    # Count final files
    for split in ['train', 'valid']:
        benign_count = len(list((target / split / 'benign').glob('*')))
        malignant_count = len(list((target / split / 'malignant').glob('*')))
        print(f"\n{split.upper()}:")
        print(f"  Benign: {benign_count}")
        print(f"  Malignant: {malignant_count}")
        print(f"  Total: {benign_count + malignant_count}")
    
    print("\n📝 Next steps:")
    print("  1. Run data exploration: jupyter notebook notebooks/01_data_exploration.ipynb")
    print("  2. Train the model: python train.py")
    print("=" * 60)


def create_sample_dataset(target_path, n_samples_per_class=50):
    """
    Create a small sample dataset for quick testing
    This creates dummy images - replace with actual images for real use
    
    Args:
        target_path: Path to create sample dataset
        n_samples_per_class: Number of samples per class
    """
    
    print("=" * 60)
    print("🎨 Creating Sample Dataset (for testing)")
    print("=" * 60)
    print("\n⚠️ Note: This creates placeholder images for testing only")
    print("For actual training, use real histopathology images\n")
    
    target = Path(target_path)
    
    # Create directories
    for split in ['train', 'valid']:
        for class_name in ['benign', 'malignant']:
            (target / split / class_name).mkdir(parents=True, exist_ok=True)
    
    try:
        from PIL import Image
        import numpy as np
        
        # Create sample images
        for split, n_samples in [('train', n_samples_per_class), ('valid', n_samples_per_class // 5)]:
            for class_name in ['benign', 'malignant']:
                print(f"Creating {n_samples} {class_name} images for {split}...")
                
                for i in range(n_samples):
                    # Create random image (replace with actual images)
                    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                    img = Image.fromarray(img_array)
                    
                    img_path = target / split / class_name / f"{class_name}_{i:04d}.png"
                    img.save(img_path)
        
        print("\n✅ Sample dataset created!")
        print("⚠️ Remember: Replace with real images before actual training")
        
    except ImportError:
        print("❌ PIL not installed. Install with: pip install pillow")


def main():
    parser = argparse.ArgumentParser(description='Organize BreakHis dataset')
    parser.add_argument('--source', type=str, help='Path to source dataset')
    parser.add_argument('--target', type=str, default='data', help='Path to target folder')
    parser.add_argument('--magnification', type=str, default='400X', 
                       choices=['40X', '100X', '200X', '400X'],
                       help='Magnification level to use')
    parser.add_argument('--valid-split', type=float, default=0.2,
                       help='Validation split ratio (default: 0.2)')
    parser.add_argument('--create-sample', action='store_true',
                       help='Create sample dataset for testing')
    
    args = parser.parse_args()
    
    if args.create_sample:
        create_sample_dataset(args.target)
    elif args.source:
        organize_breakhis_dataset(
            args.source, 
            args.target, 
            args.magnification, 
            args.valid_split
        )
    else:
        print("❌ Error: Please provide --source path or use --create-sample")
        print("\nUsage examples:")
        print("  python prepare_data.py --source /path/to/breakhis")
        print("  python prepare_data.py --create-sample  # For testing only")
        parser.print_help()


if __name__ == "__main__":
    main()
