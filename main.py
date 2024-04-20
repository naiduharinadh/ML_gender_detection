import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained face detection model from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load your gender classification model
gender_model = load_model('C:/Users/naidu/CodeAlphaTask-1/task-1/Male_Female_Weights.h5')

# Open a connection to the camera (0 represents the default camera)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # Iterate through detected faces
    for (x, y, w, h) in faces:
        # Extract the face region
        face_roi = frame[y:y + h, x:x + w]

        # Preprocess the face image for your gender classification model (resize, normalize, etc.)
        face_roi = cv2.resize(face_roi, (64, 64))  # Adjust the size based on your model input shape
        face_roi = face_roi / 255.0  # Normalize pixel values

        # Expand the dimensions to match the model input shape
        face_roi = np.expand_dims(face_roi, axis=0)

        # Use the gender classification model to predict gender
        gender_prediction = gender_model.predict(face_roi)

        # Display a rectangle around the face and label the gender prediction
        gender_label = "Male" if gender_prediction > 0.5 else "Female"
        color = (0, 255, 0) if gender_prediction > 0.5 else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, gender_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Display the frame
    cv2.imshow('Gender Classification', frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
