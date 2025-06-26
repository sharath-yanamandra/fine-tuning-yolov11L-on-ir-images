#!/usr/bin/env python3
"""
YOLOv11 Training Script for IR Person Detection
"""
'''
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

'''

import os
import yaml
from ultralytics import YOLO
import torch

def create_dataset_config(dataset_path):
    """Create the dataset configuration file"""
    
    # Convert to absolute path to avoid path issues
    dataset_path = os.path.abspath(dataset_path)
    
    # Define paths for your structure
    train_images = os.path.join(dataset_path, 'train', 'images')
    val_images = os.path.join(dataset_path, 'test', 'images')  # Using test as validation
    
    # Verify paths exist
    if not os.path.exists(train_images):
        raise FileNotFoundError(f"Training images path not found: {train_images}")
    if not os.path.exists(val_images):
        raise FileNotFoundError(f"Validation images path not found: {val_images}")
    
    # Also verify labels exist
    train_labels = os.path.join(dataset_path, 'train', 'labels')
    val_labels = os.path.join(dataset_path, 'test', 'labels')
    
    if not os.path.exists(train_labels):
        raise FileNotFoundError(f"Training labels path not found: {train_labels}")
    if not os.path.exists(val_labels):
        raise FileNotFoundError(f"Validation labels path not found: {val_labels}")
    
    print(f"Found training images: {len(os.listdir(train_images))} files")
    print(f"Found training labels: {len(os.listdir(train_labels))} files")
    print(f"Found validation images: {len(os.listdir(val_images))} files")
    print(f"Found validation labels: {len(os.listdir(val_labels))} files")
    
    # Create dataset config with absolute paths
    config = {
        'train': train_images.replace('\\', '/'),  # Use forward slashes for consistency
        'val': val_images.replace('\\', '/'),
        'nc': 1,  # number of classes (person only)
        'names': ['person']  # class names
    }
    
    # Save config file
    config_path = os.path.join(dataset_path, 'data.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Dataset config created: {config_path}")
    print(f"Training path: {config['train']}")
    print(f"Validation path: {config['val']}")
    return config_path

def train_yolo_model(dataset_path, model_size='yolo11l.pt', epochs=100, batch_size=16, img_size=640):
    """Train YOLOv11 model on IR dataset"""
    
    # Create dataset config
    config_path = create_dataset_config(dataset_path)
    
    # Load pre-trained model
    print(f"Loading model: {model_size}")
    model = YOLO(model_size)
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on device: {device}")
    
    # Training parameters
    training_args = {
        'data': config_path,
        'epochs': epochs,
        'batch': batch_size,
        'imgsz': img_size,
        'device': device,
        'patience': 50,  # early stopping patience
        'save_period': 10,  # save model every 10 epochs
        'workers': 8,  # number of dataloader workers
        'project': 'runs/detect',  # project directory
        'name': 'ir_person_detection',  # experiment name
        'exist_ok': True,  # overwrite existing experiment
        'pretrained': True,  # use pretrained weights
        'optimizer': 'AdamW',  # optimizer
        'lr0': 0.01,  # initial learning rate
        'lrf': 0.1,  # final learning rate factor
        'momentum': 0.937,  # momentum
        'weight_decay': 0.0005,  # weight decay
        'warmup_epochs': 3,  # warmup epochs
        'warmup_momentum': 0.8,  # warmup momentum
        'box': 7.5,  # box loss gain
        'cls': 0.5,  # class loss gain
        'dfl': 1.5,  # DFL loss gain
        'augment': True,  # apply augmentations
        'mosaic': 1.0,  # mosaic probability
        'mixup': 0.1,  # mixup probability
        'copy_paste': 0.1,  # copy paste probability
        'val': True,  # validate during training
        'plots': True,  # save training plots
        'verbose': True  # verbose output
    }
    
    print("Starting training...")
    print(f"Training parameters: {training_args}")
    
    # Train the model
    results = model.train(**training_args)
    
    # Validation
    print("Evaluating model...")
    metrics = model.val()
    
    # Export model
    print("Exporting model...")
    model.export(format='onnx')  # export to ONNX format
    
    return model, results, metrics

def test_model(model_path, test_image_path):
    """Test the trained model on a single image"""
    
    # Load trained model
    model = YOLO(model_path)
    
    # Run inference
    results = model(test_image_path)
    
    # Display results
    for r in results:
        # Print detections
        print(f"Detected {len(r.boxes)} persons")
        
        # Save results
        r.save(filename='detection_result.jpg')
        
    return results

def main():
    """Main training function"""
    
    # Configuration - Since you're running from inside the dataset folder
    DATASET_PATH = '.'  # Current directory (sib ir images folder)
    MODEL_SIZE = 'yolo11l.pt'  # YOLOv11 Large model
    EPOCHS = 100
    BATCH_SIZE = 16  # Adjust based on your GPU memory
    IMG_SIZE = 640
    
    try:
        # Train the model
        model, results, metrics = train_yolo_model(
            dataset_path=DATASET_PATH,
            model_size=MODEL_SIZE,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            img_size=IMG_SIZE
        )
        
        print("Training completed successfully!")
        print(f"Best model saved at: runs/detect/ir_person_detection/weights/best.pt")
        print(f"Validation mAP@0.5: {metrics.box.map50}")
        print(f"Validation mAP@0.5:0.95: {metrics.box.map}")
        
        # Optional: Test on a sample image
        # test_results = test_model('runs/detect/ir_person_detection/weights/best.pt', 'path/to/test/image.jpg')
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()

# Additional utility functions

def resume_training(checkpoint_path, additional_epochs=50):
    """Resume training from a checkpoint"""
    model = YOLO(checkpoint_path)
    results = model.train(resume=True, epochs=additional_epochs)
    return results

def evaluate_model(model_path, dataset_config):
    """Evaluate trained model"""
    model = YOLO(model_path)
    metrics = model.val(data=dataset_config)
    return metrics

def predict_batch(model_path, image_folder, output_folder):
    """Run predictions on a batch of images"""
    model = YOLO(model_path)
    results = model.predict(
        source=image_folder,
        save=True,
        project=output_folder,
        name='predictions',
        conf=0.5,  # confidence threshold
        iou=0.45   # IoU threshold for NMS
    )
    return results

# Example usage for inference after training:
"""
# Load your trained model
model = YOLO('runs/detect/ir_person_detection/weights/best.pt')

# Run inference on new IR images
results = model.predict(
    source='path/to/ir/images',
    save=True,
    conf=0.5,
    show=True
)

# For real-time detection
results = model.predict(source=0, show=True)  # webcam
"""

