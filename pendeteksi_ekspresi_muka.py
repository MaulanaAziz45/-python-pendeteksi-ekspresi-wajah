import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Fungsi untuk preprocess gambar
def preprocess_image(face):
    resized_face = cv2.resize(face, (48, 48))  # Resize ke 48x48
    normalized_face = resized_face / 255.0  # Normalisasi
    return normalized_face.reshape(1, 48, 48, 1)  # Sesuaikan dimensi untuk model

# Fungsi untuk membuat model CNN
def create_model():
    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(7, activation='softmax')  # 7 kelas untuk ekspresi
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Fungsi untuk mendeteksi ekspresi dari wajah
def detect_expression(model, face_image):
    prediction = model.predict(face_image)
    expression = np.argmax(prediction)  # Ambil indeks kelas dengan nilai tertinggi
    return expression

# Fungsi untuk mengonversi indeks ke label ekspresi
def get_expression_label(index):
    labels = ["marah", "jijik", "takut", "bahagia", "sedih", "terkejut", "netral"]
    return labels[index]

# Mulai program utama untuk mendeteksi ekspresi wajah melalui kamera
def main():
    # Buat dan load model (di sini bisa diload model yang sudah dilatih)
    model = create_model()
    # model.load_weights('model_weights.h5')  # Load bobot model jika sudah pernah dilatih dan disimpan

    cap = cv2.VideoCapture(0)  # Buka kamera
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face_image = preprocess_image(face)
            expression_index = detect_expression(model, face_image)
            label = get_expression_label(expression_index)

            # Tampilkan label ekspresi dan kotak wajah
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow('Ekspresi Wajah', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Jalankan program utama
if __name__ == "__main__":
    main()
