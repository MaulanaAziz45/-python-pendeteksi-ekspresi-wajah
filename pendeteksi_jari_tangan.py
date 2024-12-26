import cv2
import mediapipe as mp

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Buka kamera
cap = cv2.VideoCapture(0)

# Fungsi untuk mendeteksi jari yang diangkat
def detect_fingers(landmarks):
    # Indeks untuk ujung jari
    tips_ids = [4, 8, 12, 16, 20]
    fingers_status = {
        "Jempol": 0,
        "Telunjuk": 0,
        "Tengah": 0,
        "Manis": 0,
        "Kelingking": 0
    }
    
    # Jempol (menghadap ke samping)
    if landmarks[tips_ids[0]].x < landmarks[tips_ids[0] - 1].x:
        fingers_status["Jempol"] = 1

    # Jari lainnya (menghadap ke atas)
    if landmarks[tips_ids[1]].y < landmarks[tips_ids[1] - 2].y:
        fingers_status["Telunjuk"] = 1
    if landmarks[tips_ids[2]].y < landmarks[tips_ids[2] - 2].y:
        fingers_status["Tengah"] = 1
    if landmarks[tips_ids[3]].y < landmarks[tips_ids[3] - 2].y:
        fingers_status["Manis"] = 1
    if landmarks[tips_ids[4]].y < landmarks[tips_ids[4] - 2].y:
        fingers_status["Kelingking"] = 1

    return fingers_status

# Loop utama untuk membaca kamera
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Flip frame untuk mirroring
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Gambar titik-titik dan sambungan tangan
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Dapatkan status jari yang diangkat
            landmarks = hand_landmarks.landmark
            fingers_up = detect_fingers(landmarks)

            # Tampilkan status tiap jari di layar
            y_offset = 30
            for finger, status in fingers_up.items():
                if status == 1:
                    cv2.putText(frame, f'{finger} Terangkat', (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    y_offset += 30

    # Tampilkan hasil frame
    cv2.imshow("Hand Detection", frame)

    # Tombol "q" untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
