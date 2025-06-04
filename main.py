import cv2
import argparse
from detector import MobileNetSSDDetector
from fan_controller import get_fan_speed

def main():
    """Main function to run the smart fan automation simulation."""
    parser = argparse.ArgumentParser(description="Smart Fan Automation Using Image Processing")
    parser.add_argument("--input", default="0", help="Path to video file or webcam index (default: 0)")
    parser.add_argument("--prototxt", default="models/MobileNetSSD_deploy.prototxt", 
                        help="Path to prototxt file")
    parser.add_argument("--model", default="models/MobileNetSSD_deploy.caffemodel", 
                        help="Path to caffemodel file")
    args = parser.parse_args()
    
    # Initialize video capture
    try:
        if args.input.isdigit():
            cap = cv2.VideoCapture(int(args.input))
        else:
            cap = cv2.VideoCapture(args.input)
        if not cap.isOpened():
            raise Exception("Failed to open video source")
    except Exception as e:
        print(f"Error: {e}")
        return
    
    # Initialize detector
    try:
        detector = MobileNetSSDDetector(args.prototxt, args.model)
    except Exception as e:
        print(f"Error: {e}")
        cap.release()
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame or video ended")
            break
        
        # Detect persons
        boxes = detector.detect(frame)
        person_count = len(boxes)
        fan_speed = get_fan_speed(person_count)
        
        # Draw bounding boxes
        for box in boxes:
            x, y, w, h = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Display fan speed
        cv2.putText(frame, f"Fan Speed: {fan_speed}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Log decision
        print(f"Detected {person_count} persons, Fan Speed: {fan_speed}")
        
        # Display frame
        cv2.imshow("Smart Fan Simulation", frame)
        
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
