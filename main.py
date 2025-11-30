import cv2
import mediapipe as mp
import os
import argparse
import glob
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh_model = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    refine_landmarks=True,
    max_num_faces=5,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

camera = cv2.VideoCapture(0)
known_mesh_embeddings = []
known_names = []
mesh_tolerance = 0.12
mesh_metric = "cosine"
mesh_threshold = 0.9
prev_name = ""
stable_count = 0

def mesh_embedding_from_landmarks(landmarks):
    pts = np.array([[lm.x, lm.y] for lm in landmarks.landmark], dtype=np.float32)
    c = np.mean(pts, axis=0)
    centered = pts - c
    min_xy = np.min(pts, axis=0)
    max_xy = np.max(pts, axis=0)
    diag = np.linalg.norm(max_xy - min_xy)
    if diag <= 1e-6:
        return None
    normed = centered / diag
    return normed.flatten()

def close_camera():
    camera.release()
    face_mesh_model.close()
    cv2.destroyAllWindows()
    exit()

def drawer_boxes(frame):
    global prev_name, stable_count
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = face_mesh_model.process(rgb)
    ih, iw, _ = frame.shape
    if res.multi_face_landmarks:
        for lms in res.multi_face_landmarks:
            xs = [lm.x * iw for lm in lms.landmark]
            ys = [lm.y * ih for lm in lms.landmark]
            x = int(max(0, min(xs)))
            y = int(max(0, min(ys)))
            x2 = int(min(iw, max(xs)))
            y2 = int(min(ih, max(ys)))
            w = max(0, x2 - x)
            h = max(0, y2 - y)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 4)

            emb = mesh_embedding_from_landmarks(lms)
            name = "Unknown"
            score_text = ""
            if emb is not None and known_mesh_embeddings:
                if mesh_metric == "cosine":
                    e = emb
                    en = np.linalg.norm(e)
                    if en > 0:
                        e = e / en
                    sims = []
                    for k in known_mesh_embeddings:
                        kn = np.linalg.norm(k)
                        kk = k / kn if kn > 0 else k
                        sims.append(float(np.dot(e, kk)))
                    bi = int(np.argmax(sims))
                    best = sims[bi]
                    score_text = f"{best:.2f}"
                    if best >= mesh_threshold:
                        name = known_names[bi]
                else:
                    ds = [np.linalg.norm(emb - k) for k in known_mesh_embeddings]
                    bi = int(np.argmin(ds))
                    best = ds[bi]
                    score_text = f"{best:.3f}"
                    if best <= mesh_tolerance:
                        name = known_names[bi]

            if name == prev_name:
                stable_count += 1
            else:
                prev_name = name
                stable_count = 1

            label = name if stable_count >= 2 else (name if score_text == "" else f"{name} {score_text}")
            cv2.putText(frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

def process_image_path(path, save=False, output_dir=None, display=True):
    img = cv2.imread(path)
    if img is None:
        print(f"Gagal membaca gambar: {path}")
        return
    drawer_boxes(img)
    if save:
        out_dir = output_dir or os.path.join(os.getcwd(), "outputs")
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.basename(path)
        out_path = os.path.join(out_dir, base)
        cv2.imwrite(out_path, img)
        print(f"Disimpan: {out_path}")
    if display:
        cv2.imshow("MediaPipe Face Detection", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def process_input(input_path, save=False, output_dir=None, display=True):
    if os.path.isdir(input_path):
        patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
        files = []
        for p in patterns:
            files.extend(glob.glob(os.path.join(input_path, p)))
        if not files:
            print("Tidak ada file gambar ditemukan di direktori")
            return
        for f in sorted(files):
            process_image_path(f, save=save, output_dir=output_dir, display=display)
    else:
        process_image_path(input_path, save=save, output_dir=output_dir, display=display)

def load_known_faces(faces_dir):
    encs = []
    names = []
    if not os.path.isdir(faces_dir):
        return encs, names
    patterns = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    with mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True, max_num_faces=5) as fm:
        subs = [d for d in os.listdir(faces_dir) if os.path.isdir(os.path.join(faces_dir, d))]
        if subs:
            for person in subs:
                pdir = os.path.join(faces_dir, person)
                files = []
                for p in patterns:
                    files.extend(glob.glob(os.path.join(pdir, p)))
                for f in files:
                    try:
                        img = cv2.imread(f)
                        if img is None:
                            continue
                        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        r = fm.process(rgb)
                        if r.multi_face_landmarks:
                            areas = []
                            ih, iw, _ = img.shape
                            for lms in r.multi_face_landmarks:
                                xs = [lm.x * iw for lm in lms.landmark]
                                ys = [lm.y * ih for lm in lms.landmark]
                                a = (max(xs) - min(xs)) * (max(ys) - min(ys))
                                areas.append((a, lms))
                            areas.sort(key=lambda x: x[0], reverse=True)
                            emb = mesh_embedding_from_landmarks(areas[0][1])
                            if emb is not None:
                                encs.append(emb)
                                names.append(person)
                    except Exception:
                        pass
        else:
            files = []
            for p in patterns:
                files.extend(glob.glob(os.path.join(faces_dir, p)))
            for f in files:
                try:
                    img = cv2.imread(f)
                    if img is None:
                        continue
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    r = fm.process(rgb)
                    if r.multi_face_landmarks:
                        ih, iw, _ = img.shape
                        areas = []
                        for lms in r.multi_face_landmarks:
                            xs = [lm.x * iw for lm in lms.landmark]
                            ys = [lm.y * ih for lm in lms.landmark]
                            a = (max(xs) - min(xs)) * (max(ys) - min(ys))
                            areas.append((a, lms))
                        areas.sort(key=lambda x: x[0], reverse=True)
                        emb = mesh_embedding_from_landmarks(areas[0][1])
                        if emb is not None:
                            base = os.path.splitext(os.path.basename(f))[0]
                            encs.append(emb)
                            names.append(base)
                except Exception:
                    pass
    return encs, names

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--faces", type=str, default="faces")
    parser.add_argument("--tolerance", type=float, default=0.9)
    parser.add_argument("--metric", type=str, default="cosine")
    args = parser.parse_args()

    global known_mesh_embeddings, known_names, mesh_tolerance, mesh_metric, mesh_threshold
    mesh_metric = args.metric.lower()
    if mesh_metric == "cosine":
        mesh_threshold = args.tolerance
    else:
        mesh_tolerance = args.tolerance
    known_mesh_embeddings, known_names = load_known_faces(args.faces)
    if not known_mesh_embeddings:
        print("Tidak ada wajah dikenal yang dimuat dari folder faces")
    else:
        print(f"Memuat {len(known_names)} identitas: {', '.join(known_names)}")

    if args.input:
        display = not args.no_display
        process_input(args.input, save=args.save, output_dir=args.output, display=display)
        face_mesh_model.close()
        cv2.destroyAllWindows()
        return

    print("MediaPipe FaceMesh Recognition Started")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = camera.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        drawer_boxes(frame)
        cv2.imshow("MediaPipe Face Detection", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            close_camera()

if __name__ == "__main__":
    main()
