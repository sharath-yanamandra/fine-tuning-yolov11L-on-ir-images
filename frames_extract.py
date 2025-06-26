#!/usr/bin/env python3
"""
Video Frame Extraction Tool
Extracts frames from video and saves them as images for YOLO inference
"""

import cv2
import os
from pathlib import Path
import math

def extract_frames_all(video_path, output_dir, image_format='jpg'):
    """
    Extract ALL frames from video
    
    Args:
        video_path: Path to input video file
        output_dir: Directory to save extracted frames
        image_format: Format to save images ('jpg', 'png')
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps
    
    print(f"Video Info:")
    print(f"  - Total frames: {total_frames}")
    print(f"  - FPS: {fps:.2f}")
    print(f"  - Duration: {duration:.2f} seconds")
    print(f"  - Output directory: {output_dir}")
    
    frame_count = 0
    extracted_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Save frame
        frame_filename = f"frame_{frame_count:06d}.{image_format}"
        frame_path = os.path.join(output_dir, frame_filename)
        
        cv2.imwrite(frame_path, frame)
        extracted_count += 1
        
        # Progress update
        if frame_count % 100 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
        
        frame_count += 1
    
    cap.release()
    print(f"\n✅ Extraction complete!")
    print(f"Extracted {extracted_count} frames to {output_dir}")

def extract_frames_interval(video_path, output_dir, interval_seconds=1, image_format='jpg'):
    """
    Extract frames at specific time intervals
    
    Args:
        video_path: Path to input video file
        output_dir: Directory to save extracted frames
        interval_seconds: Time interval between extractions (in seconds)
        image_format: Format to save images ('jpg', 'png')
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    # Calculate frame interval
    frame_interval = int(fps * interval_seconds)
    
    print(f"Video Info:")
    print(f"  - Total frames: {total_frames}")
    print(f"  - FPS: {fps:.2f}")
    print(f"  - Duration: {duration:.2f} seconds")
    print(f"  - Extracting every {interval_seconds} second(s)")
    print(f"  - Frame interval: {frame_interval}")
    print(f"  - Output directory: {output_dir}")
    
    frame_count = 0
    extracted_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Extract frame at specified intervals
        if frame_count % frame_interval == 0:
            timestamp = frame_count / fps
            frame_filename = f"frame_{extracted_count:04d}_t{timestamp:.2f}s.{image_format}"
            frame_path = os.path.join(output_dir, frame_filename)
            
            cv2.imwrite(frame_path, frame)
            extracted_count += 1
            
            print(f"Extracted frame {extracted_count}: {frame_filename} (t={timestamp:.2f}s)")
        
        frame_count += 1
    
    cap.release()
    print(f"\n✅ Extraction complete!")
    print(f"Extracted {extracted_count} frames to {output_dir}")

def extract_frames_count(video_path, output_dir, target_count=100, image_format='jpg'):
    """
    Extract a specific number of frames evenly distributed throughout the video
    
    Args:
        video_path: Path to input video file
        output_dir: Directory to save extracted frames
        target_count: Number of frames to extract
        image_format: Format to save images ('jpg', 'png')
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps
    
    # Calculate which frames to extract
    if target_count >= total_frames:
        frame_indices = list(range(total_frames))
    else:
        step = total_frames / target_count
        frame_indices = [int(i * step) for i in range(target_count)]
    
    print(f"Video Info:")
    print(f"  - Total frames: {total_frames}")
    print(f"  - FPS: {fps:.2f}")
    print(f"  - Duration: {duration:.2f} seconds")
    print(f"  - Target extraction count: {target_count}")
    print(f"  - Actual extraction count: {len(frame_indices)}")
    print(f"  - Output directory: {output_dir}")
    
    extracted_count = 0
    current_frame = 0
    
    for target_frame in frame_indices:
        # Seek to the target frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        
        if ret:
            timestamp = target_frame / fps
            frame_filename = f"frame_{extracted_count:04d}_f{target_frame:06d}_t{timestamp:.2f}s.{image_format}"
            frame_path = os.path.join(output_dir, frame_filename)
            
            cv2.imwrite(frame_path, frame)
            extracted_count += 1
            
            print(f"Extracted frame {extracted_count}: {frame_filename}")
    
    cap.release()
    print(f"\n✅ Extraction complete!")
    print(f"Extracted {extracted_count} frames to {output_dir}")

def get_video_info(video_path):
    """Get detailed information about the video"""
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None
    
    # Get properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    
    # Get codec information
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
    
    cap.release()
    
    info = {
        'frame_count': frame_count,
        'fps': fps,
        'width': width,
        'height': height,
        'duration': duration,
        'codec': codec,
        'file_size': os.path.getsize(video_path) / (1024*1024)  # MB
    }
    
    return info

def main():
    """Main function - choose your extraction method"""
    
    # ==================== CONFIGURATION ====================
    
    # UPDATE THESE PATHS
    VIDEO_PATH = r"C:/Users/DELL/Desktop/SIB demo/sib ir images/video1.mp4"  # Change this to your video path
    OUTPUT_DIR = r"C:/Users/DELL/Desktop/SIB demo/sib ir images/extracted_frames"  # Where to save frames
    
    # ==================== VIDEO INFO ====================
    
    print("Getting video information...")
    info = get_video_info(VIDEO_PATH)
    
    if info is None:
        return
    
    print(f"\n📹 Video Information:")
    print(f"  - Resolution: {info['width']}x{info['height']}")
    print(f"  - Total frames: {info['frame_count']}")
    print(f"  - FPS: {info['fps']:.2f}")
    print(f"  - Duration: {info['duration']:.2f} seconds ({info['duration']/60:.1f} minutes)")
    print(f"  - Codec: {info['codec']}")
    print(f"  - File size: {info['file_size']:.1f} MB")
    
    # ==================== CHOOSE EXTRACTION METHOD ====================
    
    print(f"\n🎬 Choose extraction method:")
    print(f"1. Extract ALL frames ({info['frame_count']} frames)")
    print(f"2. Extract every N seconds (recommended)")
    print(f"3. Extract specific number of frames evenly distributed")
    
    choice = input("\nEnter your choice (1/2/3): ").strip()
    
    if choice == "1":
        # Extract all frames
        print("\n⚠️  Warning: This will extract ALL frames. For long videos, this creates many files!")
        confirm = input("Continue? (y/n): ").strip().lower()
        if confirm == 'y':
            extract_frames_all(VIDEO_PATH, OUTPUT_DIR)
    
    elif choice == "2":
        # Extract every N seconds
        interval = float(input("Enter interval in seconds (e.g., 1 for every second, 0.5 for every half second): "))
        estimated_frames = info['duration'] / interval
        print(f"This will extract approximately {estimated_frames:.0f} frames")
        extract_frames_interval(VIDEO_PATH, OUTPUT_DIR, interval)
    
    elif choice == "3":
        # Extract specific count
        count = int(input("Enter number of frames to extract: "))
        extract_frames_count(VIDEO_PATH, OUTPUT_DIR, count)
    
    else:
        print("Invalid choice!")
        return
    
    print(f"\n🎯 Next steps:")
    print(f"1. Check the extracted frames in: {OUTPUT_DIR}")
    print(f"2. Run YOLO inference on these frames using your trained model")
    print(f"3. Use batch inference in the previous script to process all frames")

def batch_extract_multiple_videos():
    """Extract frames from multiple videos"""
    
    video_folder = input("Enter folder path containing videos: ").strip()
    output_base_dir = input("Enter output base directory: ").strip()
    interval = float(input("Enter interval in seconds: "))
    
    if not os.path.exists(video_folder):
        print("Video folder not found!")
        return
    
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
    
    for file in os.listdir(video_folder):
        if any(file.lower().endswith(ext) for ext in video_extensions):
            video_path = os.path.join(video_folder, file)
            video_name = Path(file).stem
            output_dir = os.path.join(output_base_dir, video_name)
            
            print(f"\nProcessing: {file}")
            extract_frames_interval(video_path, output_dir, interval)

if __name__ == "__main__":
    main()
    
    # Uncomment to process multiple videos
    # batch_extract_multiple_videos()