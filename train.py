"""
Breast Cancer Classifier - Training Script

This script trains the breast cancer classification model.
Alternative to using the Jupyter notebook.

Usage:
    python train.py

Author: Your Name
Date: June 2026
"""

from fastai.vision.all import *
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def main():
    """Main training function"""
    
    print("=" * 60)
    print("🔬 Breast Cancer Classifier - Training Script")
    print("=" * 60)
    
    # Configuration
    DATA_PATH = Path('data')
    MODEL_PATH = Path('models')
    MODEL_PATH.mkdir(exist_ok=True)
    
    BATCH_SIZE = 32
    IMAGE_SIZE = 224
    EPOCHS = 10
    FREEZE_EPOCHS = 3
    ARCHITECTURE = resnet34  # Change to resnet50 for better accuracy
    
    print(f"\n📁 Data path: {DATA_PATH.absolute()}")
    print(f"💾 Model save path: {MODEL_PATH.absolute()}")
    print(f"\n⚙️ Configuration:")
    print(f"  - Architecture: {ARCHITECTURE.__name__}")
    print(f"  - Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Total epochs: {EPOCHS}")
    print(f"  - Freeze epochs: {FREEZE_EPOCHS}")
    
    # Check if data exists
    if not DATA_PATH.exists():
        print(f"\n❌ ERROR: Data folder not found at {DATA_PATH.absolute()}")
        print("\n📝 Please ensure you have:")
        print("  1. Downloaded the BreakHis dataset")
        print("  2. Organized it into the following structure:")
        print("     data/train/benign/")
        print("     data/train/malignant/")
        print("     data/valid/benign/")
        print("     data/valid/malignant/")
        return
    
    print("\n✅ Data folder found!")
    
    # Set random seed for reproducibility
    set_seed(42, reproducible=True)
    
    # Create DataLoaders
    print("\n📊 Creating DataLoaders...")
    try:
        dls = ImageDataLoaders.from_folder(
            DATA_PATH,
            train='train',
            valid='valid',
            valid_pct=0.2,  # Use if no separate valid folder
            seed=42,
            item_tfms=Resize(IMAGE_SIZE),
            batch_tfms=aug_transforms(
                mult=2.0,
                do_flip=True,
                flip_vert=True,
                max_rotate=20.0,
                max_lighting=0.2,
                max_warp=0.2,
                p_affine=0.75,
                p_lighting=0.75
            ),
            bs=BATCH_SIZE
        )
        
        print(f"✅ DataLoaders created successfully!")
        print(f"  - Training samples: {len(dls.train_ds)}")
        print(f"  - Validation samples: {len(dls.valid_ds)}")
        print(f"  - Classes: {dls.vocab}")
        
    except Exception as e:
        print(f"\n❌ Error creating DataLoaders: {str(e)}")
        return
    
    # Create learner
    print("\n🧠 Creating model...")
    learn = vision_learner(
        dls,
        ARCHITECTURE,
        metrics=[accuracy, error_rate, Precision(), Recall(), F1Score()],
        pretrained=True
    )
    print("✅ Model created successfully!")
    
    # Find optimal learning rate
    print("\n🔍 Finding optimal learning rate...")
    try:
        lr_min, lr_steep = learn.lr_find(suggest_funcs=(minimum, steep))
        print(f"  - Minimum LR: {lr_min:.2e}")
        print(f"  - Steepest LR: {lr_steep:.2e}")
        learning_rate = lr_steep
    except:
        print("  - Using default learning rate: 1e-3")
        learning_rate = 1e-3
    
    # Train the model
    print(f"\n🚀 Starting training for {EPOCHS} epochs...")
    print("=" * 60)
    
    try:
        learn.fine_tune(
            epochs=EPOCHS,
            base_lr=learning_rate,
            freeze_epochs=FREEZE_EPOCHS
        )
        print("\n" + "=" * 60)
        print("✅ Training completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        return
    
    # Evaluate the model
    print("\n📈 Evaluating model performance...")
    try:
        # Get predictions
        preds, targets = learn.get_preds()
        pred_labels = preds.argmax(dim=1)
        
        # Calculate metrics
        from sklearn.metrics import classification_report, roc_auc_score
        
        print("\n📊 Classification Report:")
        print(classification_report(targets, pred_labels, target_names=dls.vocab))
        
        # Calculate AUC-ROC
        auc_score = roc_auc_score(targets, preds[:, 1])
        print(f"\n🎯 AUC-ROC Score: {auc_score:.4f}")
        
    except Exception as e:
        print(f"⚠️ Warning: Could not generate full evaluation: {str(e)}")
    
    # Save the model
    print(f"\n💾 Saving model...")
    try:
        model_file = MODEL_PATH / 'breast_cancer_classifier.pkl'
        learn.export(model_file)
        print(f"✅ Model saved successfully!")
        print(f"   Location: {model_file.absolute()}")
        
    except Exception as e:
        print(f"❌ Error saving model: {str(e)}")
        return
    
    # Final summary
    print("\n" + "=" * 60)
    print("🎉 Training pipeline completed successfully!")
    print("=" * 60)
    print("\n📝 Next steps:")
    print("  1. Run the Gradio app: python app.py")
    print("  2. Test the model with new images")
    print("  3. Review the training notebook for detailed analysis")
    print("  4. Share your project on GitHub")
    print("\n💡 Tips for your portfolio:")
    print("  - Document your results in the README")
    print("  - Create visualizations of model performance")
    print("  - Write about challenges and solutions")
    print("  - Explain the medical context and impact")
    print("=" * 60)

if __name__ == "__main__":
    main()
