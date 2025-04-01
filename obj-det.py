import cv2
from ultralytics import YOLO
import threading
import playsound
import time

class MouseDetector:
    def __init__(self):
        # Initialize YOLO model
        self.model = YOLO('yolov8n.pt')  # Using YOLOv8 nano model
        self.confidence_threshold = 0.20  # Lowered threshold for better detection
        self.mouse_class_id = 64  # Correct mouse class ID in COCO dataset (64 is for mouse)
        self.cup_class_id = 41  # Correct cup class ID in COCO dataset (41 is for cup)
        self.red_box = None  # To store the current red box coordinates
        self.sound_playing = False  # Flag to prevent multiple sound threads
        self.sound_stop_event = threading.Event()  # Event to signal when to stop the sound

    def detect_mice(self, frame):
        # Perform detection
        results = self.model(frame)

        # Initialize list to store mouse detections
        mouse_detections = []
        cup_detections = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Check if detection is a mouse
                if int(box.cls) == self.mouse_class_id and box.conf > self.confidence_threshold:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf)
                    mouse_detections.append({
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'confidence': conf
                    })
                # Check if detection is a cup
                elif int(box.cls) == self.cup_class_id and box.conf > self.confidence_threshold:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf)
                    cup_detections.append({
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'confidence': conf
                    })

        # Limit to four mice detections
        mouse_detections = sorted(mouse_detections, key=lambda x: x['confidence'], reverse=True)[:4]

        return mouse_detections, cup_detections

    def check_overlap(self, rect1, rect2):
        """Check if two rectangles overlap"""
        x1_1, y1_1, x2_1, y2_1 = rect1
        x1_2, y1_2, x2_2, y2_2 = rect2

        # Check if one rectangle is to the left of the other
        if x2_1 < x1_2 or x2_2 < x1_1:
            return False

        # Check if one rectangle is above the other
        if y2_1 < y1_2 or y2_2 < y1_1:
            return False

        return True

    def check_line_intersect(self, rect1, rect2):
        """
        Check if the edges of rect1 intersect with rect2
        rect1 and rect2 are in format (x1, y1, x2, y2)
        """
        x1_1, y1_1, x2_1, y2_1 = rect1
        x1_2, y1_2, x2_2, y2_2 = rect2

        # Define the four lines of rect1
        top_line = (x1_1, y1_1, x2_1, y1_1)
        right_line = (x2_1, y1_1, x2_1, y2_1)
        bottom_line = (x1_1, y2_1, x2_1, y2_1)
        left_line = (x1_1, y1_1, x1_1, y2_1)

        # Check if any of the lines intersect with rect2
        # This is a simplified check: if rect2 overlaps with rect1
        # and rect2 is not completely inside rect1, then edges intersect
        if self.check_overlap(rect1, rect2):
            # Check if rect2 is completely inside rect1
            if (x1_1 <= x1_2 and x2_2 <= x2_1 and
                y1_1 <= y1_2 and y2_2 <= y2_1):
                return False
            return True

        return False

    def play_sound_loop(self):
        """Play alert sound in a loop until stop event is set"""
        while not self.sound_stop_event.is_set():
            try:
                playsound.playsound('alert.mp3')
                time.sleep(0.1)  # Small delay to prevent CPU hogging
            except Exception as e:
                print(f"Error playing sound: {e}")
                break
        self.sound_playing = False

    def check_cup_collision(self, cup_detections, red_box):
        if not red_box:
            # If there's no red box, stop any active sound
            if self.sound_playing:
                self.sound_stop_event.set()
            return False

        # Convert red_box from corners to x1,y1,x2,y2 format
        (x1, y1), (x2, y2) = red_box
        red_box_rect = (x1, y1, x2, y2)

        collision_detected = False

        for cup in cup_detections:
            cup_bbox = cup['bbox']

            # Check if cup's boundary intersects with any of red box's boundary lines
            if self.check_line_intersect(red_box_rect, cup_bbox):
                collision_detected = True
                break

        # Manage sound based on collision state
        if collision_detected and not self.sound_playing:
            # Start playing alert sound in a loop
            self.sound_stop_event.clear()  # Reset the stop flag
            self.sound_playing = True
            sound_thread = threading.Thread(target=self.play_sound_loop, daemon=True)
            sound_thread.start()
        elif not collision_detected and self.sound_playing:
            # Stop the sound if no collision
            self.sound_stop_event.set()

        return collision_detected

    def draw_detections(self, frame, mouse_detections, cup_detections):
        # Draw mouse detections in blue
        for detection in mouse_detections:
            x1, y1, x2, y2 = detection['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue box
            conf_text = f"{detection['confidence']:.2f}"
            cv2.putText(frame, conf_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Draw cup detections in green
        for cup in cup_detections:
            x1, y1, x2, y2 = cup['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
            conf_text = f"{cup['confidence']:.2f}"
            cv2.putText(frame, conf_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        red_box = None
        if len(mouse_detections) == 4:
            # Calculate center point of the 4 mice
            x_sum = sum([(d['bbox'][0] + d['bbox'][2]) // 2 for d in mouse_detections])
            y_sum = sum([(d['bbox'][1] + d['bbox'][3]) // 2 for d in mouse_detections])

            center_x = x_sum // 4
            center_y = y_sum // 4

            # Define the size of the red square
            square_size = 180

            top_left = (center_x - square_size, center_y - square_size)
            bottom_right = (center_x + square_size, center_y + square_size)

            # Draw the red box
            cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), 2)
            red_box = (top_left, bottom_right)
            self.red_box = red_box
        else:
            self.red_box = None

        # Check for cup collision with the red box
        collision = self.check_cup_collision(cup_detections, red_box)

        # Add status text
        status_text = "STATUS: "
        if collision:
            status_text += "ALERT! Cup on boundary"
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif len(mouse_detections) < 4:
            status_text += f"Need {4 - len(mouse_detections)} more mice"
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
        else:
            status_text += "OK - 4 mice detected"
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return frame

def main():
    # Initialize video capture (0 for default camera)
    cap = cv2.VideoCapture(0)
    detector = MouseDetector()

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open video capture device")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to receive frame")
            break

        # Detect mice and cups in the frame
        mouse_detections, cup_detections = detector.detect_mice(frame)

        # Draw detections on frame
        frame = detector.draw_detections(frame, mouse_detections, cup_detections)

        # Display the frame
        cv2.imshow('Mouse & Cup Detection', frame)

        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # Clean up before exiting
            detector.sound_stop_event.set()  # Stop any playing sound
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()