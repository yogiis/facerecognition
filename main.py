import cv2

face_ref = cv2.CascadeClassifier("face_reff.xml")
camera = cv2.VideoCapture(0)

def face_detection(frame):
    optimized_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_ref.detectMultiScale(optimized_frame, scaleFactor=1.1, minNeighbors=3, minSize=(500, 500))
    return faces

def close_camera():
    camera.release()
    cv2.destroyAllWindows()
    exit()

def drawer_boxes(frame):
    for x, y, w, h in face_detection(frame):
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 4)

def main():
    while True:
        _, frame = camera.read()
        drawer_boxes(frame)
        cv2.imshow("Face Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            close_camera()

if __name__ == "__main__":
    main()