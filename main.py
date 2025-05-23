import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Gaya gambar titik & garis
drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)

# Pakai blok 'with' supaya resource otomatis dibersihkan
with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Konversi ke RGB untuk MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Gambar titik & mesh
                mp_drawing.draw_landmarks(
                    frame,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_TESSELATION,
                    drawing_spec,
                    drawing_spec)

                # Ambil titik-titik bibir
                top_lip = face_landmarks.landmark[13].y
                bottom_lip = face_landmarks.landmark[14].y
                left_mouth = face_landmarks.landmark[61].x
                right_mouth = face_landmarks.landmark[291].x

                # Hitung bukaan mulut dan lebar senyuman
                mouth_open = abs(top_lip - bottom_lip)
                mouth_width = abs(right_mouth - left_mouth)

                # Logika deteksi senyum
                if mouth_open > 0.015 and mouth_width > 0.04:
                    cv2.putText(frame, "Smile", (270, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                else:
                    cv2.putText(frame, "Not Smile", (250, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                # DEBUG info buat bantu tuning
                cv2.putText(frame, f"Open: {mouth_open:.3f}", (10, 460),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"Width: {mouth_width:.3f}", (10, 480),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Smile Detector", frame)

        # Tombol keluar: tekan 'q' atau close window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.getWindowProperty("Smile Detector", cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
