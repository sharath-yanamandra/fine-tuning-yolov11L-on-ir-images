"""
Bank Vault IR Person Detection Inference Script
Use your trained YOLOv11 model to detect people in vault images
"""
'''
import os
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import time

class VaultPersonDetector:
    def __init__(self, model_path, confidence_threshold=0.5, iou_threshold=0.45):
        """
        Initialize the vault person detector
        
        Args:
            model_path: Path to your trained best.pt model
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for non-maximum suppression
        """
        self.model = YOLO(model_path)
        self.conf_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        
        print(f"Loaded model: {model_path}")
        print(f"Confidence threshold: {confidence_threshold}")
        print(f"IoU threshold: {iou_threshold}")
    
    def detect_single_image(self, image_path, save_result=True, show_result=False):
        """
        Detect people in a single vault image
        
        Args:
            image_path: Path to the vault image
            save_result: Whether to save the annotated image
            show_result: Whether to display the result
        
        Returns:
            results: YOLO detection results
            person_count: Number of people detected
        """
        # Run inference
        results = self.model.predict(
            source=image_path,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            save=save_result,
            show=show_result,
            verbose=False
        )
        
        # Count detections
        person_count = len(results[0].boxes) if results[0].boxes is not None else 0
        
        # Print results
        print(f"\nImage: {os.path.basename(image_path)}")
        print(f"People detected: {person_count}")
        
        if person_count > 0:
            confidences = results[0].boxes.conf.cpu().numpy()
            print(f"Confidence scores: {[f'{conf:.3f}' for conf in confidences]}")
            
            # Get bounding box coordinates
            boxes = results[0].boxes.xyxy.cpu().numpy()
            for i, (box, conf) in enumerate(zip(boxes, confidences)):
                x1, y1, x2, y2 = box.astype(int)
                print(f"  Person {i+1}: [{x1}, {y1}, {x2}, {y2}] confidence: {conf:.3f}")
        
        return results, person_count
    
    def detect_batch_images(self, image_folder, output_folder=None):
        """
        Detect people in multiple vault images
        
        Args:
            image_folder: Folder containing vault images
            output_folder: Folder to save annotated results
        
        Returns:
            summary: Dictionary with detection summary
        """
        if output_folder is None:
            output_folder = os.path.join(image_folder, 'detections')
        
        # Create output folder
        os.makedirs(output_folder, exist_ok=True)
        
        # Supported image formats
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # Find all images
        image_files = []
        for ext in image_extensions:
            image_files.extend(Path(image_folder).glob(f'*{ext}'))
            image_files.extend(Path(image_folder).glob(f'*{ext.upper()}'))
        
        if not image_files:
            print(f"No images found in {image_folder}")
            return {}
        
        print(f"Found {len(image_files)} images to process")
        
        # Process all images
        summary = {
            'total_images': len(image_files),
            'images_with_people': 0,
            'total_people_detected': 0,
            'detection_details': []
        }
        
        for i, image_path in enumerate(image_files, 1):
            print(f"\nProcessing {i}/{len(image_files)}: {image_path.name}")
            
            try:
                # Run detection
                results = self.model.predict(
                    source=str(image_path),
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    save=True,
                    project=output_folder,
                    name='results',
                    exist_ok=True,
                    verbose=False
                )
                
                # Count detections
                person_count = len(results[0].boxes) if results[0].boxes is not None else 0
                
                if person_count > 0:
                    summary['images_with_people'] += 1
                    summary['total_people_detected'] += person_count
                    
                    confidences = results[0].boxes.conf.cpu().numpy()
                    avg_confidence = np.mean(confidences)
                    
                    print(f"  ✓ {person_count} people detected (avg confidence: {avg_confidence:.3f})")
                else:
                    print(f"  ○ No people detected")
                
                # Store details
                summary['detection_details'].append({
                    'image': image_path.name,
                    'people_count': person_count,
                    'confidences': confidences.tolist() if person_count > 0 else []
                })
                
            except Exception as e:
                print(f"  ✗ Error processing {image_path.name}: {str(e)}")
        
        # Print summary
        print("\n" + "="*50)
        print("DETECTION SUMMARY")
        print("="*50)
        print(f"Total images processed: {summary['total_images']}")
        print(f"Images with people: {summary['images_with_people']}")
        print(f"Total people detected: {summary['total_people_detected']}")
        print(f"Average people per image: {summary['total_people_detected'] / summary['total_images']:.2f}")
        print(f"Detection rate: {summary['images_with_people'] / summary['total_images'] * 100:.1f}%")
        print(f"Results saved to: {output_folder}")
        
        return summary
    
    def detect_realtime_stream(self, source=0, save_video=False):
        """
        Real-time detection for vault security cameras
        
        Args:
            source: Camera source (0 for webcam, or RTSP URL for IP camera)
            save_video: Whether to save the annotated video
        """
        print(f"Starting real-time detection from source: {source}")
        print("Press 'q' to quit")
        
        # Start streaming prediction
        results = self.model.predict(
            source=source,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            show=True,
            save=save_video,
            stream=True,
            verbose=False
        )
        
        # Process results
        for result in results:
            person_count = len(result.boxes) if result.boxes is not None else 0
            if person_count > 0:
                print(f"ALERT: {person_count} person(s) detected in vault!")
    
    def analyze_image_quality(self, image_path):
        """
        Analyze image quality for better detection insights
        """
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        
        # Calculate image statistics
        stats = {
            'resolution': f"{img.shape[1]}x{img.shape[0]}",
            'mean_brightness': np.mean(img),
            'brightness_std': np.std(img),
            'contrast_ratio': np.max(img) - np.min(img),
            'file_size_mb': os.path.getsize(image_path) / (1024*1024)
        }
        
        return stats

def main():
    """Main function to run vault detection"""
    
    # Configuration
    MODEL_PATH = "C:/Users/DELL/Desktop/SIB demo/sib ir images/runs/detect/ir_person_detection/weights/best.pt"  # Your trained model
    VAULT_IMAGES_FOLDER = "vault_images"  # Folder with your vault images
    CONFIDENCE_THRESHOLD = 0.5  # Adjust based on your needs
    IOU_THRESHOLD = 0.45
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found: {MODEL_PATH}")
        print("Make sure training is complete and the model file exists.")
        return
    
    # Initialize detector
    detector = VaultPersonDetector(
        model_path=MODEL_PATH,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        iou_threshold=IOU_THRESHOLD
    )
    
    # Example 1: Single image detection
    single_image_path = "sample_vault_image.jpg"  # Replace with your image path
    if os.path.exists(single_image_path):
        print("\n" + "="*50)
        print("SINGLE IMAGE DETECTION")
        print("="*50)
        results, count = detector.detect_single_image(
            single_image_path, 
            save_result=True, 
            show_result=True
        )
    
    # Example 2: Batch processing
    if os.path.exists(VAULT_IMAGES_FOLDER):
        print("\n" + "="*50)
        print("BATCH PROCESSING")
        print("="*50)
        summary = detector.detect_batch_images(
            VAULT_IMAGES_FOLDER,
            output_folder="vault_detection_results"
        )
    
    # Example 3: Real-time detection (uncomment to use)
    # print("\n" + "="*50)
    # print("REAL-TIME DETECTION")
    # print("="*50)
    # detector.detect_realtime_stream(source=0, save_video=True)

if __name__ == "__main__":
    main()

# Quick usage examples:

"""
# Basic usage after training:
from ultralytics import YOLO

# Load your trained model
model = YOLO('runs/detect/ir_person_detection/weights/best.pt')

# Detect on a single vault image
results = model.predict('vault_image.jpg', conf=0.5, save=True)

# Detect on multiple images
results = model.predict('vault_images_folder/', conf=0.5, save=True)

# Real-time detection from camera
results = model.predict(source=0, show=True, conf=0.5)

# For IP camera (RTSP stream)
results = model.predict(source='rtsp://camera_ip:port/stream', show=True, conf=0.5)
"""

'''

#!/usr/bin/env python3
"""
Batch Inference Script for IR Person Detection
Run inference on a folder of test images and save results
"""

from ultralytics import YOLO
import os
import time

def batch_inference():
    """Run inference on all images in test folder"""
    
    # Model and folder paths
    MODEL_PATH = r'C:/Users/DELL/Desktop/SIB demo/sib ir images/runs/detect/ir_person_detection/weights/best.pt'
    TEST_FOLDER = r'C:/Users/DELL/Desktop/SIB demo/sib ir images/extracted_frames/test frames'
    
    # Load the trained model
    print("Loading trained IR person detection model...")
    model = YOLO(MODEL_PATH)
    print(f"✅ Model loaded successfully!")
    print(f"Model classes: {model.names}")
    
    # Check if test folder exists
    if not os.path.exists(TEST_FOLDER):
        print(f"❌ Error: Test folder not found at {TEST_FOLDER}")
        return
    
    # Count images in folder
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    image_files = [f for f in os.listdir(TEST_FOLDER) 
                   if any(f.lower().endswith(ext) for ext in image_extensions)]
    
    if not image_files:
        print(f"❌ No image files found in {TEST_FOLDER}")
        return
    
    print(f"📁 Found {len(image_files)} images in test folder")
    print(f"🚀 Starting batch inference...")
    
    # Start timing
    start_time = time.time()
    
    # Run inference on the entire folder
    results = model.predict(
        source=TEST_FOLDER,
        conf=0.5,           # Confidence threshold
        iou=0.45,           # IoU threshold for NMS
        save=True,          # Save annotated images
        save_txt=True,      # Save detection results as txt files
        save_conf=True,     # Save confidence scores in txt files
        project='runs',     # Save to runs folder
        name='predict',     # Save to predict subfolder
        exist_ok=True,      # Overwrite existing results
        verbose=True,       # Show progress
        show_labels=True,   # Show labels on images
        show_conf=True,     # Show confidence scores
        line_width=2,       # Bounding box line width
    )
    
    # Calculate processing time
    end_time = time.time()
    total_time = end_time - start_time
    
    # Print summary
    print("\n" + "="*60)
    print("🎯 BATCH INFERENCE COMPLETED!")
    print("="*60)
    
    total_detections = 0
    images_with_detections = 0
    
    for result in results:
        if result.boxes is not None and len(result.boxes) > 0:
            total_detections += len(result.boxes)
            images_with_detections += 1
    
    print(f"📊 Summary:")
    print(f"  • Total images processed: {len(results)}")
    print(f"  • Images with detections: {images_with_detections}")
    print(f"  • Images without detections: {len(results) - images_with_detections}")
    print(f"  • Total persons detected: {total_detections}")
    print(f"  • Average detections per image: {total_detections/len(results):.2f}")
    print(f"  • Processing time: {total_time:.2f} seconds")
    print(f"  • Speed: {len(results)/total_time:.2f} images/second")
    
    print(f"\n📁 Results saved to:")
    print(f"  • Annotated images: runs/predict/")
    print(f"  • Detection files (.txt): runs/predict/labels/")
    
    # Show some detection details
    print(f"\n🔍 Detection Details (first 5 images):")
    for i, result in enumerate(results[:5]):
        filename = os.path.basename(result.path)
        if result.boxes is not None and len(result.boxes) > 0:
            confidences = [f"{conf.item():.3f}" for conf in result.boxes.conf]
            print(f"  {filename}: {len(result.boxes)} persons (conf: {', '.join(confidences)})")
        else:
            print(f"  {filename}: No detections")
    
    if len(results) > 5:
        print(f"  ... and {len(results) - 5} more images")

def view_specific_results(image_name=None):
    """View results for a specific image or show folder contents"""
    
    results_folder = "runs/predict"
    
    if not os.path.exists(results_folder):
        print(f"❌ Results folder not found: {results_folder}")
        print("Run batch inference first!")
        return
    
    if image_name:
        # Look for specific image result
        image_files = [f for f in os.listdir(results_folder) if f.lower().endswith(('.jpg', '.png'))]
        matching_files = [f for f in image_files if image_name.lower() in f.lower()]
        
        if matching_files:
            print(f"Found results for '{image_name}':")
            for file in matching_files:
                print(f"  • {file}")
        else:
            print(f"No results found for '{image_name}'")
    else:
        # List all result files
        if os.path.exists(results_folder):
            files = os.listdir(results_folder)
            image_files = [f for f in files if f.lower().endswith(('.jpg', '.png'))]
            
            print(f"📁 Found {len(image_files)} result images in {results_folder}:")
            for i, file in enumerate(image_files[:10]):  # Show first 10
                print(f"  {i+1}. {file}")
            
            if len(image_files) > 10:
                print(f"  ... and {len(image_files) - 10} more files")

def main():
    """Main function"""
    
    print("🤖 IR Person Detection - Batch Inference")
    print("="*50)
    
    # Run batch inference
    batch_inference()
    
    # Show results summary
    print("\n" + "="*50)
    view_specific_results()
    
    print(f"\n✅ All done! Check the 'runs/predict/' folder for your results.")

if __name__ == "__main__":
    main()

# Additional utility functions

def inference_with_custom_settings():
    """Run inference with custom settings"""
    
    MODEL_PATH = r'C:/Users/DELL/Desktop/SIB demo/sib ir images/runs/detect/ir_person_detection/weights/best.pt'
    TEST_FOLDER = r'C:/Users/DELL/Desktop/SIB demo/sib ir images/extracted_frames/test frames'
    
    model = YOLO(MODEL_PATH)
    
    # Custom inference settings
    results = model.predict(
        source=TEST_FOLDER,
        conf=0.3,           # Lower confidence threshold
        iou=0.5,            # Higher IoU threshold
        max_det=10,         # Maximum detections per image
        save=True,
        project='runs',
        name='predict_custom',
        exist_ok=True,
        imgsz=640,          # Image size
        augment=False,      # No test-time augmentation
        visualize=False,    # Don't save feature maps
        save_crop=True,     # Save cropped detections
    )
    
    return results

def count_detections_summary():
    """Analyze detection results from txt files"""
    
    labels_folder = "runs/predict/labels"
    
    if not os.path.exists(labels_folder):
        print("No labels folder found. Run inference first!")
        return
    
    txt_files = [f for f in os.listdir(labels_folder) if f.endswith('.txt')]
    
    total_detections = 0
    files_with_detections = 0
    
    print("📊 Detection Analysis:")
    print("-" * 40)
    
    for txt_file in txt_files[:10]:  # Show first 10
        txt_path = os.path.join(labels_folder, txt_file)
        
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            num_detections = len([line for line in lines if line.strip()])
            
        if num_detections > 0:
            files_with_detections += 1
            total_detections += num_detections
            
        print(f"{txt_file}: {num_detections} detections")
    
    if len(txt_files) > 10:
        print(f"... and {len(txt_files) - 10} more files")
    
    print(f"\nSummary:")
    print(f"Files with detections: {files_with_detections}/{len(txt_files)}")
    print(f"Total detections: {total_detections}")

# Uncomment to run custom inference:
# inference_with_custom_settings()

# Uncomment to analyze detection files:
# count_detections_summary()