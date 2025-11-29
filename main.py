import cv2
import mediapipe as mp

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection_model = mp_face_detection.FaceDetection(
    model_selection=1,  # 0 for short-range (2 meters), 1 for full-range (5 meters)
    min_detection_confidence=0.5
)

camera = cv2.VideoCapture(0)

def face_detection(frame):
    """
    Detect faces using MediaPipe Face Detection
    Returns list of detection results
    """
    # Convert BGR to RGB (MediaPipe uses RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection_model.process(rgb_frame)
    return results

def close_camera():
    camera.release()
    face_detection_model.close()
    cv2.destroyAllWindows()
    exit()

def drawer_boxes(frame):
    """
    Draw bounding boxes around detected faces
    """
    results = face_detection(frame)
    
    if results.detections:
        for detection in results.detections:
            # Get bounding box coordinates
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x = int(bboxC.xmin * iw)
            y = int(bboxC.ymin * ih)
            w = int(bboxC.width * iw)
            h = int(bboxC.height * ih)
            
            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 4)
            
            # Optional: Draw detection score
            score = detection.score[0]
            cv2.putText(frame, f'{score:.2f}', (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            
            # Optional: Draw key points (eyes, nose, etc.)
            # mp_drawing.draw_detection(frame, detection)

def main():
    print("MediaPipe Face Detection Started")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = camera.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        drawer_boxes(frame)
        cv2.imshow("MediaPipe Face Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            close_camera()

if __name__ == "__main__":
    main()