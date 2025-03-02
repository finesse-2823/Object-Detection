import cv2
import numpy as np
import torch
import math
from collections import defaultdict
import argparse
import sys
import time
import os
import subprocess
from pathlib import Path
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv8 object detection with polygon tracking')
    parser.add_argument('--source', type=str, default='0', help='Video source (0 for webcam, or file path)')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='YOLOv8 model path')
    parser.add_argument('--data', type=str, default='coco.yaml', help='Dataset config file')
    parser.add_argument('--color-threshold', type=int, default=50, help='Threshold for color similarity (higher = more lenient)')
    parser.add_argument('--min-polygon-points', type=int, default=4, help='Minimum points to form a polygon (3 or more)')
    parser.add_argument('--show', action='store_true', help='Display the detection video')
    parser.add_argument('--debug', action='store_true', help='Enable debug messages')
    parser.add_argument('--tracking-persistence', type=int, default=10, help='Number of frames to persist tracking when objects disappear')
    
    args = parser.parse_args()
    return args

# Rate-limited debug printing to avoid console flooding
last_debug_time = {}
def debug_print(message, debug_mode=False, rate_limit_secs=2, category="general"):
    """Print debug messages if debug mode is enabled, with rate limiting"""
    if not debug_mode:
        return
        
    current_time = time.time()
    if category not in last_debug_time or (current_time - last_debug_time[category]) >= rate_limit_secs:
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        print(f"[DEBUG {timestamp}] {message}")
        last_debug_time[category] = current_time

def get_available_cameras():
    """List all available camera devices on the system using v4l2 if available, otherwise fallback to OpenCV"""
    available_cameras = {}
    
    try:
        # First try using v4l2-ctl if available (Linux systems)
        result = subprocess.run(['v4l2-ctl', '--list-devices'], capture_output=True, text=True)
        
        if result.returncode == 0:
            # Parse v4l2-ctl output
            current_device = None
            for line in result.stdout.splitlines():
                if ':' in line and 'dev' not in line:
                    # This is a device name line
                    current_device = line.strip().rstrip(':')
                elif '/dev/video' in line:
                    # This is a device path line
                    video_device = line.strip()
                    device_num = int(video_device.replace('/dev/video', ''))
                    available_cameras[device_num] = f"{current_device} ({video_device})"
            
            # Verify which ones can be opened with OpenCV
            verified_cameras = {}
            for device_num, device_info in available_cameras.items():
                cap = cv2.VideoCapture(device_num)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        height, width = frame.shape[:2]
                        verified_cameras[device_num] = f"{device_info} ({width}x{height})"
                    else:
                        verified_cameras[device_num] = f"{device_info} (No signal)"
                    cap.release()
            
            if verified_cameras:
                return verified_cameras
    except Exception as e:
        print(f"Info: v4l2-ctl detection failed, falling back to OpenCV: {e}")
    
    # Fallback to OpenCV's camera detection
    index = 0
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            break
        
        # Get camera information
        ret, frame = cap.read()
        if ret:
            height, width = frame.shape[:2]
            available_cameras[index] = f"Camera #{index} ({width}x{height})"
        else:
            available_cameras[index] = f"Camera #{index} (No signal)"
        
        cap.release()
        index += 1
    
    return available_cameras

def find_color_clusters(detections, frame, color_threshold, debug=False):
    """Group objects by color similarity, used for providing feedback."""
    if detections is None or len(detections) == 0:
        return []
        
    all_objects = []
    for det in detections:
        x1, y1, x2, y2, conf, cls = det.tolist()
        if conf < 0.25:  # Skip low confidence
            continue
        color = get_dominant_color(frame, x1, y1, x2, y2)
        all_objects.append((color, (x1+x2)/2, (y1+y2)/2))
    
    # Group by color
    clusters = []
    for obj in all_objects:
        matched = False
        for i, cluster in enumerate(clusters):
            if color_distance(obj[0], cluster[0][0]) < color_threshold:
                cluster.append(obj)
                matched = True
                break
        if not matched:
            clusters.append([obj])
            
    return clusters

def get_dominant_color(image, x1, y1, x2, y2):
    """Extract dominant color from a region using k-means clustering."""
    # Ensure coordinates are within image bounds
    height, width = image.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(width, int(x2)), min(height, int(y2))
    
    # If the region is too small, return black
    if x2 <= x1 or y2 <= y1:
        return (0, 0, 0)
    
    # Extract the region of interest
    roi = image[y1:y2, x1:x2]
    
    # Check if ROI is empty
    if roi.size == 0:
        return (0, 0, 0)
    
    # Reshape and sample pixels for faster processing
    pixels = roi.reshape(-1, 3)
    pixel_count = len(pixels)
    
    if pixel_count > 1000:  # Sample for large regions
        indices = np.random.choice(pixel_count, 1000, replace=False)
        pixels = pixels[indices]
    
    # Use K-means to find the dominant color
    pixels = np.float32(pixels)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, _, centers = cv2.kmeans(pixels, 1, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Convert the dominant color back to integers
    dominant_color = tuple(map(int, centers[0]))
    return dominant_color

def color_distance(color1, color2):
    """Calculate Euclidean distance between two colors."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(color1, color2)))

def find_best_polygon_objects(detections, frame, min_points=4, color_threshold=50, debug=False):
    """Find the best set of objects that could form a polygon based on color similarity."""
    if detections is None or len(detections) == 0:
        debug_print("No detections available for polygon formation", debug, category="polygon_detection")
        return None, None
    
    # Extract all objects with their colors
    all_objects = []
    for det in detections:
        x1, y1, x2, y2, conf, cls = det.tolist()
        
        # Skip low confidence detections
        if conf < 0.25:
            continue
            
        # Get the dominant color of the object
        dominant_color = get_dominant_color(frame, x1, y1, x2, y2)
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Store as (color, center_x, center_y, cls, x1, y1, x2, y2)
        all_objects.append((dominant_color, center_x, center_y, cls, x1, y1, x2, y2))
    
    if len(all_objects) < min_points:
        debug_print(f"Not enough objects ({len(all_objects)}) to form a polygon (min={min_points})", 
                   debug, category="polygon_detection")
        return None, None
    
    # Find clusters of similar colors
    color_clusters = []
    
    for obj in all_objects:
        obj_color = obj[0]
        matched = False
        
        # Try to add to existing cluster
        for cluster_idx, cluster in enumerate(color_clusters):
            representative_color = cluster[0][0]
            if color_distance(obj_color, representative_color) < color_threshold:
                color_clusters[cluster_idx].append(obj)
                matched = True
                break
        
        # Create new cluster if no match
        if not matched:
            color_clusters.append([obj])
    
    # Find the largest cluster with at least min_points
    valid_clusters = [c for c in color_clusters if len(c) >= min_points]
    
    if not valid_clusters:
        debug_print(f"No color clusters with at least {min_points} objects found", 
                   debug, category="polygon_detection")
        return None, None
    
    # Sort clusters by size (descending)
    valid_clusters.sort(key=len, reverse=True)
    
    # Take the largest cluster
    best_cluster = valid_clusters[0]
    representative_color = best_cluster[0][0]
    
    # Extract points and full object info
    points = [(obj[1], obj[2]) for obj in best_cluster]
    
    debug_print(f"Found best polygon cluster with {len(best_cluster)} objects of color {representative_color}", 
               debug, category="polygon_detection")
    
    # Return the points and the representative color
    return points, representative_color

def arrange_points_clockwise(points):
    """Arrange points in clockwise order around their centroid for proper polygon rendering."""
    # Calculate centroid
    centroid_x = sum(p[0] for p in points) / len(points)
    centroid_y = sum(p[1] for p in points) / len(points)
    
    # Sort points based on angle from centroid
    return sorted(points, key=lambda p: math.atan2(p[1] - centroid_y, p[0] - centroid_x))

def point_in_polygon(point, polygon):
    """Ray casting algorithm for point in polygon test."""
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside

def visualize_frame(frame, detections, active_polygon, objects_in_polygon, class_names, min_points=4, debug=False):
    """Draw detection boxes, polygon and labels on the frame."""
    # Make a copy of the frame to avoid modifying the original
    vis_frame = frame.copy()
    
    # Draw the active polygon if it exists
    if active_polygon is not None:
        vertices, color = active_polygon
        
        if vertices and len(vertices) >= 3:  # Make sure we have at least 3 points for a polygon
            # Convert vertices to numpy array for drawing
            vertices_np = np.array(vertices, np.int32)
            vertices_np = vertices_np.reshape((-1, 1, 2))  # FIX: Properly reshape the array
            
            # Ensure color is a 3-tuple of RGB values for drawing
            if isinstance(color, (list, tuple)) and len(color) == 3:
                rgb_color = color
            else:
                rgb_color = (0, 255, 0)  # Default to green if color is invalid
            
            # Draw the polygon with semi-transparency
            overlay = vis_frame.copy()
            # Draw a semi-transparent polygon
            cv2.fillPoly(overlay, [vertices_np], rgb_color)  # Fill with original color
            cv2.addWeighted(overlay, 0.3, vis_frame, 0.7, 0, vis_frame)  # Blend with original frame
            
            # Draw the polygon outline
            cv2.polylines(vis_frame, [vertices_np], True, rgb_color, 2)
            
            # Add label for polygon
            centroid_x = int(sum(x for x, _ in vertices) / len(vertices))
            centroid_y = int(sum(y for _, y in vertices) / len(vertices))
            cv2.putText(vis_frame, f"Zone", (centroid_x, centroid_y), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Count objects in this polygon
            object_count = len(objects_in_polygon)
            if object_count > 0:
                cv2.putText(vis_frame, f"Objects: {object_count}", (centroid_x, centroid_y + 20), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    else:
        # If no polygon is active, show guidance
        # Count objects by color to provide feedback
        color_clusters = find_color_clusters(detections, frame, 50)
        
        # Display guidance
        if color_clusters:
            largest_cluster_size = max(len(cluster) for cluster in color_clusters)
            cv2.putText(vis_frame, f"Largest color group: {largest_cluster_size}/{min_points} objects needed", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            if largest_cluster_size < min_points:
                cv2.putText(vis_frame, f"Place {min_points} objects of the same color to form a zone", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Draw all detections
    if detections is not None and len(detections) > 0:
        for det in detections:
            x1, y1, x2, y2, conf, cls = det.tolist()
            
            # Skip low confidence detections
            if conf < 0.25:
                continue
                
            # Get class name
            cls_id = int(cls)
            class_name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
            
            # Get dominant color for the box
            if active_polygon is not None:
                _, poly_color = active_polygon
                obj_color = get_dominant_color(frame, x1, y1, x2, y2)
                is_part_of_polygon = color_distance(obj_color, poly_color) < 50  # Use same threshold as polygon detection
                box_color = poly_color if is_part_of_polygon else (0, 255, 0)
            else:
                box_color = (0, 255, 0)  # Default green for boxes
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 2)
            
            # Add label
            label = f"{class_name}: {conf:.2f}"
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(vis_frame, (int(x1), int(y1)-text_size[1]-5), 
                        (int(x1)+text_size[0], int(y1)), box_color, -1)
            cv2.putText(vis_frame, label, (int(x1), int(y1)-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    # Add instructions
    cv2.putText(vis_frame, "Press 'q' to quit, 'r' to reset zone", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return vis_frame

def run_detection(args):
    debug = args.debug
    debug_print("Starting object detection application", debug)
    
    # Ensure model file exists
    model_path = Path(args.model)
    
    if not model_path.exists():
        print(f"Model file {model_path} not found. Downloading...")
    
    # Load YOLOv8 model with GPU acceleration if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    debug_print(f"Loading YOLOv8 model from {args.model} on {device}", debug)
    try:
        model = YOLO(args.model, task='detect')
        print(f"Running on device: {device}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Get class names from model
    class_names = model.names
    debug_print(f"Loaded model with {len(class_names)} classes", debug)
    
    # Get list of available cameras
    available_cameras = get_available_cameras()
    debug_print(f"Available cameras: {available_cameras}", debug)
    print("\n" + "="*50)
    print("Available cameras:")
    for cam_id, cam_info in available_cameras.items():
        print(f"  {cam_id}: {cam_info}")
    print("="*50 + "\n")
    
    # Open video capture
    current_source = args.source
    if current_source.isdigit():
        current_source = int(current_source)  # Convert string to integer for webcam
    
    # Check if the requested camera is available
    if isinstance(current_source, int) and current_source not in available_cameras:
        print("\n" + "="*50)
        print(f"WARNING: Camera {current_source} not found in available cameras!")
        if available_cameras:
            fallback = list(available_cameras.keys())[0]
            print(f"Falling back to camera {fallback}: {available_cameras[fallback]}")
            print("Use one of these available cameras instead:")
            for cam_id, cam_info in available_cameras.items():
                print(f"  {cam_id}: {cam_info}")
            print("="*50 + "\n")
            current_source = fallback
        else:
            print("No cameras available. Exiting.")
            return
    
    debug_print(f"Opening video source: {current_source}", debug)
    
    # Try multiple times to open camera (sometimes it fails on first try)
    max_attempts = 3
    cap = None
    for attempt in range(max_attempts):
        cap = cv2.VideoCapture(current_source)
        if cap.isOpened():
            break
        debug_print(f"Attempt {attempt+1}/{max_attempts} to open video source failed. Retrying...", debug)
        time.sleep(1)  # Wait 1 second between attempts
    
    if not cap or not cap.isOpened():
        print(f"Error: Could not open video source {current_source} after {max_attempts} attempts")
        return
    
    print(f"Successfully opened video source: {current_source}")
    
    # Set camera properties for better performance if possible
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # Request Motion JPEG
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Request 640x480 for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)  # Request 30fps
    
    # Objects currently inside polygon: {object_id: (center_x, center_y, class_id)}
    objects_in_polygon = {}
    
    # Active polygon: (vertices, color) or None if no polygon is active
    active_polygon = None
    
    # Counter for frames where we couldn't detect the polygon
    missing_polygon_frames = 0
    
    # Track processing performance
    frame_count = 0
    start_time = cv2.getTickCount()
    performance_stats = {'fps_history': []}
    last_camera_switch_time = 0  # To prevent too frequent camera switches
    
    print("Starting detection loop. Press 'q' to quit. Press 0-9 to switch cameras. Press 'r' to reset zone.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            debug_print("End of video stream or error reading frame", debug, category="video")
            
            # Try to reconnect to the camera
            debug_print("Attempting to reconnect to the camera...", debug, category="video")
            cap.release()
            cap = cv2.VideoCapture(current_source)
            if not cap.isOpened():
                print("Failed to reconnect to the camera. Exiting.")
                break
            
            continue
        
        frame_count += 1
        
        # Skip frames for better performance if needed
        if frame_count % 2 != 0 and not args.show:  # Process every other frame if not displaying
            continue
        
        # Run YOLOv8 detection
        results = model(frame, conf=args.conf, verbose=False)
        
        # Extract detections
        if results and len(results) > 0 and hasattr(results[0], 'boxes'):
            detections = results[0].boxes.data
            if len(detections) > 0:
                debug_print(f"Found {len(detections)} detections in frame", debug, category="detection", rate_limit_secs=5)
        else:
            detections = torch.zeros((0, 6))  # Empty tensor with correct shape
        
        # Try to form or update the polygon
        polygon_points, polygon_color = find_best_polygon_objects(
            detections, frame, 
            min_points=args.min_polygon_points, 
            color_threshold=args.color_threshold,
            debug=debug
        )
        
        # Update active polygon if we found a valid one
        if polygon_points and len(polygon_points) >= args.min_polygon_points:
            # Sort points in clockwise order for proper polygon rendering
            arranged_points = arrange_points_clockwise(polygon_points)
            active_polygon = (arranged_points, polygon_color)
            missing_polygon_frames = 0
            debug_print(f"Updated active polygon with {len(arranged_points)} points", debug, category="active_polygon")
        else:
            # If we couldn't detect the polygon for too many frames, reset it
            missing_polygon_frames += 1
            if missing_polygon_frames > args.tracking_persistence and active_polygon is not None:
                debug_print(f"Lost polygon for {missing_polygon_frames} frames, resetting", debug, category="active_polygon")
                active_polygon = None
                
            # Provide more helpful feedback when no polygon is formed but only periodically
            if frame_count % 60 == 0:  # Only check periodically to avoid log spam
                # Count how many objects are detected by color groups
                color_clusters = find_color_clusters(detections, frame, args.color_threshold)
                if color_clusters:
                    largest_cluster = max(color_clusters, key=len)
                    debug_print(f"Largest color group has {len(largest_cluster)} objects (need {args.min_polygon_points} for polygon)", 
                               debug, category="help", rate_limit_secs=10)
                    if len(largest_cluster) < args.min_polygon_points:
                        print(f"To form a polygon, place {args.min_polygon_points} objects of the same color in the camera view")
        
        # Track objects inside/outside the polygon
        objects_in_polygon.clear()  # Reset objects in polygon for this frame
        
        if active_polygon is not None:
            polygon_vertices, polygon_color = active_polygon
            
            for det in detections:
                x1, y1, x2, y2, conf, cls = det.tolist()
                
                # Skip low confidence detections
                if conf < args.conf:
                    continue
                
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # Get the dominant color of this object
                object_color = get_dominant_color(frame, x1, y1, x2, y2)
                
                # Skip if this object is part of the polygon itself
                if polygon_color and color_distance(object_color, polygon_color) < args.color_threshold:
                    continue
                
                # Create a unique object ID - more efficient than string formatting
                object_id = f"{int(cls)}_{int(center_x)}_{int(center_y)}"
                
                # Check if the object is inside the polygon
                if point_in_polygon((center_x, center_y), polygon_vertices):
                    objects_in_polygon[object_id] = (center_x, center_y, cls)
                    
                    # Print which objects are in the zone
                    cls_id = int(cls)
                    object_class = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
                    debug_print(f"{object_class} is inside the zone", debug, category="inside_zone", rate_limit_secs=5)
        
        # Calculate and display FPS
        if frame_count % 30 == 0:
            current_time = cv2.getTickCount()
            elapsed_time = (current_time - start_time) / cv2.getTickFrequency()
            if elapsed_time > 0:
                fps = frame_count / elapsed_time
                performance_stats['fps_history'].append(fps)
                # Only show occasional FPS updates to avoid flooding console
                debug_print(f"Processing speed: {fps:.1f} FPS", debug, category="performance", rate_limit_secs=5)
        
        if args.show:
            # Visualize the results
            try:
                vis_frame = visualize_frame(
                    frame.copy(), detections, active_polygon, 
                    objects_in_polygon, class_names, 
                    min_points=args.min_polygon_points, debug=debug
                )
                
                # Display source information
                source_info = f"Source: {current_source}"
                if isinstance(current_source, int) and current_source in available_cameras:
                    source_info = f"Source: {available_cameras[current_source]}"
                cv2.putText(vis_frame, source_info, (10, frame.shape[0] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Display the frame
                cv2.imshow('Object Detection with Polygon Tracking', vis_frame)
            except Exception as e:
                print(f"Error during visualization: {e}")
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            # Break the loop if 'q' is pressed
            if key == ord('q'):
                debug_print("User quit the application", debug)
                break
            
            # Reset the polygon if 'r' is pressed
            if key == ord('r'):
                debug_print("User reset the polygon", debug)
                active_polygon = None
                objects_in_polygon.clear()
                print("Zone has been reset")
                
            # Switch camera if a number key is pressed (0-9)
            current_time = time.time()
            if current_time - last_camera_switch_time > 1.0:  # Prevent too frequent switches
                if ord('0') <= key <= ord('9'):
                    camera_id = key - ord('0')
                    if camera_id in available_cameras:
                        debug_print(f"Switching to camera {camera_id}", debug)
                        # Release current camera
                        cap.release()
                        # Open new camera
                        cap = cv2.VideoCapture(camera_id)
                        current_source = camera_id
                        if cap.isOpened():
                            print(f"Switched to camera {camera_id}: {available_cameras[camera_id]}")
                            # Reset polygon when switching cameras
                            active_polygon = None
                            objects_in_polygon.clear()
                        else:
                            print(f"Failed to open camera {camera_id}")
                            # Try to reopen the previous camera
                            cap = cv2.VideoCapture(current_source)
                    else:
                        debug_print(f"Camera {camera_id} not available", debug)
                    
                    last_camera_switch_time = current_time
    
    # Release resources
    if cap and cap.isOpened():
        cap.release()
    if args.show:
        cv2.destroyAllWindows()
    
    # Print final statistics
    end_time = cv2.getTickCount()
    total_time = (end_time - start_time) / cv2.getTickFrequency()
    if total_time > 0:
        avg_fps = frame_count / total_time
        print(f"Processed {frame_count} frames in {total_time:.2f} seconds ({avg_fps:.2f} FPS)")
        
        # Show performance stats if we collected any
        if performance_stats['fps_history']:
            avg_fps = sum(performance_stats['fps_history']) / len(performance_stats['fps_history'])
            print(f"Average FPS: {avg_fps:.1f}")
    else:
        print(f"Processed {frame_count} frames in {total_time:.2f} seconds")

if __name__ == "__main__":
    args = parse_args()
    run_detection(args)