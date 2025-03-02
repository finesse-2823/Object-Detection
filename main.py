import cv2
import numpy as np
import torch
import math
from collections import defaultdict
import argparse
import sys
import time
from pathlib import Path
from ultralytics import YOLO

def parse_args():
    # Create our parser and parse arguments
    parser = argparse.ArgumentParser(description='YOLOv8 object detection with polygon tracking')
    parser.add_argument('--source', type=str, default='0', help='Video source (0 for webcam, or file path)')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='YOLOv8 model path')
    parser.add_argument('--data', type=str, default='coco.yaml', help='Dataset config file')
    parser.add_argument('--color-threshold', type=int, default=30, help='Threshold for color similarity')
    parser.add_argument('--show', action='store_true', help='Display the detection video')
    parser.add_argument('--debug', action='store_true', help='Enable debug messages')
    
    # Parse our args
    args, unknown = parser.parse_known_args()
    return args

def debug_print(message, debug_mode=False):
    """Print debug messages if debug mode is enabled"""
    if debug_mode:
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        print(f"[DEBUG {timestamp}] {message}")

def get_available_cameras():
    """List all available camera devices on the system"""
    available_cameras = {}
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

def group_objects_by_color(detections, frame, color_threshold=30, debug=False):
    """Group detected objects by their dominant color."""
    color_groups = defaultdict(list)
    
    # Check if detections is empty
    if detections is None or len(detections) == 0:
        debug_print("No detections to group by color", debug)
        return color_groups
        
    # Iterate through each detection
    debug_print(f"Grouping {len(detections)} detections by color", debug)
    for det in detections:
        x1, y1, x2, y2, conf, cls = det.tolist()
        
        # Skip low confidence detections
        if conf < 0.25:
            continue
            
        # Get the dominant color of the object
        dominant_color = get_dominant_color(frame, x1, y1, x2, y2)
        
        # Check if this color is close to any existing group
        found_group = False
        for group_color in list(color_groups.keys()):
            if color_distance(dominant_color, group_color) < color_threshold:
                # Calculate center of the object
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                color_groups[group_color].append((center_x, center_y, cls))
                found_group = True
                break
        
        # If no matching group, create a new one
        if not found_group:
            # Calculate center of the object
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            color_groups[dominant_color].append((center_x, center_y, cls))
    
    debug_print(f"Created {len(color_groups)} color groups", debug)
    return color_groups

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

def visualize_frame(frame, detections, active_polygons, objects_in_polygons, class_names, debug=False):
    """Draw detection boxes, polygons and labels on the frame."""
    # Draw all active polygons
    debug_print(f"Visualizing frame with {len(active_polygons)} polygons and {len(detections)} detections", debug)
    
    for poly_id, (vertices, color) in active_polygons.items():
        # Convert vertices to numpy array for drawing
        vertices_np = np.array(vertices, np.int32)
        vertices_np = vertices_np.reshape((-1, 1, 2))
        
        # Draw the polygon
        cv2.polylines(frame, [vertices_np], True, color, 2)
        
        # Add label for polygon
        centroid_x = int(sum(x for x, _ in vertices) / len(vertices))
        centroid_y = int(sum(y for _, y in vertices) / len(vertices))
        cv2.putText(frame, f"Zone {poly_id}", (centroid_x, centroid_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Count objects in this polygon
        object_count = len(objects_in_polygons[poly_id])
        if object_count > 0:
            cv2.putText(frame, f"Objects: {object_count}", (centroid_x, centroid_y + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
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
            
            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # Add label
            label = f"{class_name}: {conf:.2f}"
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame, (int(x1), int(y1)-text_size[1]-5), 
                        (int(x1)+text_size[0], int(y1)), (0, 255, 0), -1)
            cv2.putText(frame, label, (int(x1), int(y1)-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    # Add camera switch instructions
    cv2.putText(frame, "Press 0-9 to switch cameras", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return frame

def run_detection(args):
    debug = args.debug
    debug_print("Starting object detection application", debug)
    
    # Ensure model and data files exist
    model_path = Path(args.model)
    
    if not model_path.exists():
        print(f"Model file {model_path} not found. Downloading...")
    
    # Load YOLOv8 model with GPU acceleration if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    debug_print(f"Loading YOLOv8 model from {args.model} on {device}", debug)
    # Add this option to prevent YOLO from parsing command line arguments
    model = YOLO(args.model, task='detect')
    print(f"Running on device: {device}")
    
    # Get class names from model
    class_names = model.names
    debug_print(f"Loaded model with {len(class_names)} classes", debug)
    
    # Get list of available cameras
    available_cameras = get_available_cameras()
    debug_print(f"Available cameras: {available_cameras}", debug)
    print("Available cameras:")
    for cam_id, cam_info in available_cameras.items():
        print(f"  {cam_id}: {cam_info}")
    
    # Open video capture
    current_source = args.source
    if current_source.isdigit():
        current_source = int(current_source)  # Convert string to integer for webcam
    
    debug_print(f"Opening video source: {current_source}", debug)
    
    # Try multiple times to open camera (sometimes it fails on first try)
    max_attempts = 3
    for attempt in range(max_attempts):
        cap = cv2.VideoCapture(current_source)
        if cap.isOpened():
            break
        debug_print(f"Attempt {attempt+1}/{max_attempts} to open video source failed. Retrying...", debug)
        cv2.waitKey(1000)  # Wait 1 second between attempts
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {current_source} after {max_attempts} attempts")
        return
    
    print(f"Successfully opened video source: {current_source}")
    
    # Objects currently inside polygons (polygon_id: {object_id: (center_x, center_y, class_id)})
    objects_in_polygons = defaultdict(dict)
    
    # Store polygons (polygon_id: (vertices, color))
    active_polygons = {}
    polygon_counter = 0
    
    # Track processing performance
    frame_count = 0
    start_time = cv2.getTickCount()
    last_camera_switch_time = 0  # To prevent too frequent camera switches
    
    print("Starting detection loop. Press 'q' to quit. Press 0-9 to switch cameras.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            debug_print("End of video stream or error reading frame", debug)
            
            # Try to reconnect to the camera
            debug_print("Attempting to reconnect to the camera...", debug)
            cap.release()
            cap = cv2.VideoCapture(current_source)
            if not cap.isOpened():
                print("Failed to reconnect to the camera. Exiting.")
                break
            
            continue
        
        frame_count += 1
        
        # Run YOLOv8 detection
        debug_print(f"Processing frame {frame_count}", debug)
        results = model(frame, conf=args.conf, verbose=False)
        
        # Extract detections - properly access boxes
        if results and len(results) > 0 and hasattr(results[0], 'boxes'):
            # Extract the tensor directly without assuming batch dimension
            detections = results[0].boxes.data
            if len(detections) > 0:
                debug_print(f"Found {len(detections)} detections in frame", debug)
        else:
            if frame_count % 30 == 0:  # Don't log this too frequently
                debug_print(f"Frame {frame_count}: No valid detections found", debug)
            detections = torch.zeros((0, 6))  # Empty tensor with correct shape
            
        # Group detected objects by color
        color_groups = group_objects_by_color(detections, frame, args.color_threshold, debug)
        
        # Process each color group to find polygons
        for color, points in color_groups.items():
            # If we have 3 or more objects of the same color, form a polygon
            if len(points) >= 3:
                # Extract just the coordinates for the polygon
                vertices = [(x, y) for x, y, _ in points]
                
                # Create a new polygon or update existing one
                polygon_exists = False
                for poly_id, (poly_vertices, poly_color) in list(active_polygons.items()):
                    if color_distance(color, poly_color) < args.color_threshold:
                        active_polygons[poly_id] = (vertices, color)
                        debug_print(f"Updated polygon {poly_id} with {len(vertices)} vertices", debug)
                        polygon_exists = True
                        break
                
                if not polygon_exists:
                    active_polygons[polygon_counter] = (vertices, color)
                    debug_print(f"Created new polygon {polygon_counter} with {len(vertices)} vertices", debug)
                    polygon_counter += 1
        
        # Track objects inside/outside this polygon
        for poly_id, (vertices, poly_color) in active_polygons.items():
            # Clear previous tracking for this polygon
            current_objects = set()
            
            for det in detections:
                x1, y1, x2, y2, conf, cls = det.tolist()
                
                # Skip low confidence detections
                if conf < args.conf:
                    continue
                
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # Get the dominant color of this object
                object_color = get_dominant_color(frame, x1, y1, x2, y2)
                
                # Skip if this is one of the polygon vertices
                if color_distance(object_color, poly_color) < args.color_threshold:
                    continue
                
                # Create a unique object ID
                object_id = f"{int(cls)}_{int(x1)}_{int(y1)}"
                
                # Check if the object is inside the polygon
                is_inside = point_in_polygon((center_x, center_y), vertices)
                
                # If the object is inside, track it
                if is_inside:
                    # Check if this is a new object entering the polygon
                    if object_id not in objects_in_polygons[poly_id]:
                        cls_id = int(cls)
                        object_class = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
                        debug_print(f"New {object_class} entered zone {poly_id}", debug)
                    
                    objects_in_polygons[poly_id][object_id] = (center_x, center_y, cls)
                    current_objects.add(object_id)
            
            # Check for objects that have left the polygon
            for object_id in list(objects_in_polygons[poly_id].keys()):
                if object_id not in current_objects:
                    center_x, center_y, cls = objects_in_polygons[poly_id][object_id]
                    cls_id = int(cls)
                    object_class = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
                    print(f"ALERT: {object_class} has left zone {poly_id}!")
                    del objects_in_polygons[poly_id][object_id]
        
        # Calculate and display FPS
        if frame_count % 30 == 0:  # Update less frequently to reduce console output
            current_time = cv2.getTickCount()
            elapsed_time = (current_time - start_time) / cv2.getTickFrequency()
            if elapsed_time > 0:  # Prevent division by zero
                fps = frame_count / elapsed_time
                debug_print(f"Processing speed: {fps:.1f} FPS", debug)
        
        if args.show:
            # Visualize the results
            vis_frame = visualize_frame(frame.copy(), detections, active_polygons, 
                                      objects_in_polygons, class_names, debug)
            
            # Display camera information
            cam_info = f"Source: {current_source}"
            if isinstance(current_source, int) and current_source in available_cameras:
                cam_info = available_cameras[current_source]
            cv2.putText(vis_frame, cam_info, (10, frame.shape[0] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display the frame
            cv2.imshow('Object Detection with Polygon Tracking', vis_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            # Break the loop if 'q' is pressed
            if key == ord('q'):
                debug_print("User quit the application", debug)
                break
                
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
                        else:
                            print(f"Failed to open camera {camera_id}")
                            # Try to reopen the previous camera
                            cap = cv2.VideoCapture(current_source)
                    else:
                        debug_print(f"Camera {camera_id} not available", debug)
                    
                    last_camera_switch_time = current_time
    
    # Release resources
    cap.release()
    if args.show:
        cv2.destroyAllWindows()
    
    # Print final statistics
    end_time = cv2.getTickCount()
    total_time = (end_time - start_time) / cv2.getTickFrequency()
    if total_time > 0:  # Prevent division by zero
        avg_fps = frame_count / total_time
        print(f"Processed {frame_count} frames in {total_time:.2f} seconds ({avg_fps:.2f} FPS)")
    else:
        print(f"Processed {frame_count} frames in {total_time:.2f} seconds")

if __name__ == "__main__":
    args = parse_args()
    run_detection(args)