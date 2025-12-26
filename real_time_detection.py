import cv2
import numpy as np
from tensorflow.keras.models import load_model


model = load_model("facialemotionmodel.h5")


haar_file = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haar_file)


labels = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "neutral",
    5: "sad",
    6: "surprise"
}

# Feature extraction
def extract_features(image):
    image = np.array(image)
    image = image.reshape(1, 48, 48, 1)
    image = image / 255.0
    return image

# Start webcam
webcam = cv2.VideoCapture(0)

while True:
    ret, frame = webcam.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = extract_features(face)

        prediction = model.predict(face, verbose=0)
        emotion = labels[np.argmax(prediction)]

        # Draw rectangle and emotion label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(
            frame,
            emotion,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2
        )

    cv2.imshow("Real-Time Face Emotion Detection", frame)

    # Press ESC to exit
    if cv2.waitKey(1) == 27:
        break

webcam.release()
cv2.destroyAllWindows()
