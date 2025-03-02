import cv2
import numpy as np
import torch
import math
from collections import defaultdict
import argparse
import sys
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
    
    # Parse our args
    args, unknown = parser.parse_known_args()
    return args

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

def group_objects_by_color(detections, frame, color_threshold=30):
    """Group detected objects by their dominant color."""
    color_groups = defaultdict(list)
    
    # Check if detections is empty
    if detections is None or len(detections) == 0:
        return color_groups
        
    # Iterate through each detection - no indexing with [0]
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

def visualize_frame(frame, detections, active_polygons, objects_in_polygons, class_names):
    """Draw detection boxes, polygons and labels on the frame."""
    # Draw all active polygons
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
    
    return frame

def run_detection(args):
    # Ensure model and data files exist
    model_path = Path(args.model)
    
    if not model_path.exists():
        print(f"Model file {model_path} not found. Downloading...")
    
    # Load YOLOv8 model with GPU acceleration if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Add this option to prevent YOLO from parsing command line arguments
    model = YOLO(args.model, task='detect')
    print(f"Running on device: {device}")
    
    # Get class names from model
    class_names = model.names
    
    # Open video capture
    source = args.source
    if source.isdigit():
        source = int(source)  # Convert string to integer for webcam
    
    # Try multiple times to open camera (sometimes it fails on first try)
    max_attempts = 3
    for attempt in range(max_attempts):
        cap = cv2.VideoCapture(source)
        if cap.isOpened():
            break
        print(f"Attempt {attempt+1}/{max_attempts} to open video source failed. Retrying...")
        cv2.waitKey(1000)  # Wait 1 second between attempts
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {source} after {max_attempts} attempts")
        return
    
    print(f"Successfully opened video source: {source}")
    
    # Objects currently inside polygons (polygon_id: {object_id: (center_x, center_y, class_id)})
    objects_in_polygons = defaultdict(dict)
    
    # Store polygons (polygon_id: (vertices, color))
    active_polygons = {}
    polygon_counter = 0
    
    # Track processing performance
    frame_count = 0
    start_time = cv2.getTickCount()
    
    print("Starting detection loop. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream or error reading frame.")
            break
        
        frame_count += 1
        
        # Run YOLOv8 detection
        results = model(frame, conf=args.conf, verbose=False)
        
        # Extract detections - properly access boxes
        if results and len(results) > 0 and hasattr(results[0], 'boxes'):
            # Extract the tensor directly without assuming batch dimension
            detections = results[0].boxes.data
        else:
            print(f"Frame {frame_count}: No valid detections found.")
            detections = torch.zeros((0, 6))  # Empty tensor with correct shape
            
        # Group detected objects by color
        color_groups = group_objects_by_color(detections, frame, args.color_threshold)
        
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
                        polygon_exists = True
                        break
                
                if not polygon_exists:
                    active_polygons[polygon_counter] = (vertices, color)
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
        if frame_count % 10 == 0:
            current_time = cv2.getTickCount()
            elapsed_time = (current_time - start_time) / cv2.getTickFrequency()
            if elapsed_time > 0:  # Prevent division by zero
                fps = frame_count / elapsed_time
                print(f"Processing speed: {fps:.1f} FPS")
        
        if args.show:
            # Visualize the results
            vis_frame = visualize_frame(frame.copy(), detections, active_polygons, 
                                      objects_in_polygons, class_names)
            
            # Display the frame
            cv2.imshow('Object Detection with Polygon Tracking', vis_frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("User quit the application.")
                break
    
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