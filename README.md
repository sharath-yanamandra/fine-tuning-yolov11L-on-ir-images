# IR Person Detection with YOLOv11

A complete pipeline for training and deploying YOLOv11 models on Infrared (IR) thermal images for person detection. This project fine-tunes a pre-trained YOLOv11 model specifically for thermal/IR imagery to detect people in various scenarios.

## 🚀 Overview

This project addresses the challenge of person detection in IR thermal images, which have different characteristics compared to regular RGB images. The model is trained on a custom dataset of IR images with person annotations and can be used for surveillance, security, and monitoring applications.

## 📋 Features

- **Custom YOLOv11 Training**: Fine-tuned on IR thermal images
- **Dual-mode Detection**: Works with both RGB and IR images
- **Batch Processing**: Inference on multiple images simultaneously
- **GPU Acceleration**: Optimized for NVIDIA GPUs with CUDA support
- **Comprehensive Evaluation**: Detailed metrics and visualizations
- **Production Ready**: Easy deployment and inference scripts

## 🔧 Requirements

### Hardware
- **GPU**: NVIDIA GPU with 4GB+ VRAM (recommended: RTX A2000 8GB or better)
- **RAM**: 8GB+ system memory
- **Storage**: 5GB+ free space

### Software
- Python 3.8-3.12
- CUDA 11.8+ (for GPU acceleration)
- NVIDIA drivers 450+

## 📦 Installation

### 1. Clone Repository
```bash
git clone <repository-url>
cd ir-person-detection
```

### 2. Install Dependencies
```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install Ultralytics YOLO
pip install ultralytics

# Install additional dependencies
pip install opencv-python matplotlib pillow pyyaml
```

### 3. Verify GPU Setup
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU'}")
```

## 📂 Dataset Structure

Organize your dataset in the following structure:

```
sib ir images/
├── train/
│   ├── images/          # Training IR images (.jpg, .png)
│   └── labels/          # YOLO format labels (.txt)
├── test/
│   ├── images/          # Test/validation IR images
│   └── labels/          # Corresponding labels
└── extracted_frames/
    └── test frames/     # Additional test images for inference
```

### Label Format (YOLO)
Each `.txt` file should contain:
```
class_id x_center y_center width height confidence
```
Example:
```
0 0.5 0.5 0.3 0.6 0.95
```
Where `class_id = 0` for person, and all coordinates are normalized (0-1).

## 🏋️ Training

### Basic Training
```bash
python yolo_ir_images.py
```

### Training Configuration
Key parameters in the training script:

```python
MODEL_SIZE = 'yolo11l.pt'     # Model size (n/s/m/l/x)
EPOCHS = 100                  # Training epochs
BATCH_SIZE = 32               # Batch size (adjust for your GPU)
IMG_SIZE = 640                # Input image resolution
CONFIDENCE = 0.5              # Detection confidence threshold
```

### Training Outputs
- **Model weights**: `runs/detect/ir_person_detection/weights/best.pt`
- **Training plots**: `runs/detect/ir_person_detection/`
- **Validation metrics**: mAP@0.5, mAP@0.5:0.95

### Expected Performance
- **Training time**: 2-6 hours (with GPU)
- **Dataset size**: ~12K training + 3K validation images
- **mAP@0.5**: 0.85+ (expected)

## 🔍 Inference

### Single Image Inference
```python
from ultralytics import YOLO

# Load trained model
model = YOLO('runs/detect/ir_person_detection/weights/best.pt')

# Run inference
results = model.predict('path/to/image.jpg', conf=0.5, save=True)

# Print results
for result in results:
    print(f"Detected {len(result.boxes)} persons")
```

### Batch Inference
```bash
python batch_inference.py
```

This will:
- Process all images in `extracted_frames/test frames/`
- Save annotated images to `runs/predict/`
- Generate detection files (.txt) with coordinates
- Provide detailed processing summary

### Inference Outputs
```
runs/predict/
├── image1.jpg              # Annotated images
├── image2.jpg
├── ...
└── labels/
    ├── image1.txt          # Detection coordinates
    ├── image2.txt
    └── ...
```

## 📊 Model Performance

### Training Metrics
- **Loss Functions**: Box loss, Class loss, DFL loss
- **Validation**: Automatic validation during training
- **Early Stopping**: Prevents overfitting (patience=50)

### Detection Capabilities
- **Person Detection**: High accuracy in IR thermal images
- **Confidence Scoring**: Reliable confidence estimates
- **Multi-person**: Handles multiple people in single image
- **Robustness**: Works in various thermal conditions

## 🛠️ Scripts Overview

### Core Scripts

1. **`yolo_ir_images.py`** - Main training script
   - Dataset configuration
   - Model training and validation
   - Automatic hyperparameter optimization

2. **`batch_inference.py`** - Batch processing
   - Process entire folders
   - Generate annotated results
   - Performance statistics

3. **`inference.py`** - Advanced inference utilities
   - Single image processing
   - Confidence threshold testing
   - Visual result display

### Key Functions

```python
# Training
train_yolo_model(dataset_path, model_size, epochs, batch_size)

# Inference
predict_single_image(model, image_path, confidence)
batch_inference(model, folder_path, confidence)

# Evaluation
evaluate_model(model_path, dataset_config)
```

## 🚨 Troubleshooting

### Common Issues

**1. CUDA Not Available**
```bash
# Check GPU detection
nvidia-smi

# Reinstall PyTorch with CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**2. Dataset Path Errors**
- Ensure absolute paths are used
- Check folder structure matches requirements
- Verify image and label file counts match

**3. Memory Issues**
- Reduce batch size: `BATCH_SIZE = 16` or `8`
- Use smaller model: `yolo11s.pt` instead of `yolo11l.pt`
- Reduce image size: `IMG_SIZE = 416`

**4. Slow Training (CPU)**
```
Training on device: cpu  # Should be cuda:0
```
- Install CUDA-enabled PyTorch
- Check GPU compatibility
- Update NVIDIA drivers

## 📈 Performance Optimization

### GPU Settings
```python
# Optimal settings for RTX A2000 8GB
BATCH_SIZE = 32
IMG_SIZE = 640
MODEL_SIZE = 'yolo11l.pt'
```

### CPU Fallback
```python
# If GPU unavailable
BATCH_SIZE = 4
IMG_SIZE = 416
MODEL_SIZE = 'yolo11s.pt'
EPOCHS = 50
```

## 🔄 Model Updates

### Resume Training
```python
from ultralytics import YOLO

model = YOLO('runs/detect/ir_person_detection/weights/last.pt')
model.train(resume=True, epochs=50)
```

### Fine-tuning
```python
# Fine-tune on new data
model = YOLO('runs/detect/ir_person_detection/weights/best.pt')
model.train(data='new_dataset.yaml', epochs=25, lr0=0.001)
```

## 📋 Model Deployment

### Export Formats
```python
model = YOLO('best.pt')

# Export options
model.export(format='onnx')     # ONNX
model.export(format='torchscript')  # TorchScript
model.export(format='tflite')   # TensorFlow Lite
```

### Production Inference
```python
# Optimized for production
results = model.predict(
    source=image_path,
    conf=0.5,
    iou=0.45,
    max_det=100,
    augment=False,
    visualize=False
)
```

## 📚 Dataset Information

### Current Dataset
- **Training Images**: 12,023 IR thermal images
- **Validation Images**: 3,463 IR thermal images
- **Classes**: 1 (person)
- **Annotation Format**: YOLO bounding boxes
- **Image Types**: .jpg, .png thermal/IR images

### Data Augmentation
Automatic augmentations applied during training:
- Mosaic (probability: 1.0)
- Mixup (probability: 0.1)
- Copy-paste (probability: 0.1)
- Horizontal flip (probability: 0.5)
- HSV color space variations

## 🔍 Results Analysis

### Evaluation Metrics
- **mAP@0.5**: Mean Average Precision at IoU 0.5
- **mAP@0.5:0.95**: Mean Average Precision at IoU 0.5-0.95
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)

### Confidence Thresholds
- **High Precision**: conf=0.7+ (fewer false positives)
- **Balanced**: conf=0.5 (recommended)
- **High Recall**: conf=0.3 (catch more detections)

## 📝 License

This project is for educational and research purposes. Please ensure compliance with relevant licenses when using pre-trained models and datasets.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review YOLO documentation: https://docs.ultralytics.com
3. Check PyTorch CUDA setup: https://pytorch.org/get-started/locally/

## 🎯 Future Improvements

- [ ] Multi-class detection (person, vehicle, etc.)
- [ ] Real-time video processing
- [ ] Model quantization for edge deployment
- [ ] Integration with security systems
- [ ] Advanced post-processing filters

---

**Project Status**: ✅ Production Ready  
**Last Updated**: June 2025  
**Model Version**: YOLOv11l trained on IR thermal images
