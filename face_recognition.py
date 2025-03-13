import cv2
import base64
import os
import numpy as np
import face_recognition

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def validate_voter_face(photo_data, userid):
    try:
        # Decode the base64 image
        captured_photo = base64.b64decode(photo_data.split(',')[1])

        # Convert the decoded image to a numpy array and read it with OpenCV
        nparr = np.frombuffer(captured_photo, np.uint8)
        captured_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Convert the image to grayscale for face detection
        gray_captured_image = cv2.cvtColor(captured_image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale image
        faces = face_cascade.detectMultiScale(gray_captured_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) == 0:
            return False, "No face detected in the captured image."

        # Crop the first detected face from the captured image
        x, y, w, h = faces[0]
        captured_face = captured_image[y:y + h, x:x + w]

        # Load the stored image securely
        stored_photo_path = f"static/photo/user/{userid}.png"
        if not os.path.exists(stored_photo_path):
            return False, "Stored photo not found."

        stored_image = cv2.imread(stored_photo_path)
        gray_stored_image = cv2.cvtColor(stored_image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the stored image
        stored_faces = face_cascade.detectMultiScale(gray_stored_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(stored_faces) == 0:
            return False, "No face detected in the stored image."

        # Crop the first detected face from the stored image
        sx, sy, sw, sh = stored_faces[0]
        stored_face = stored_image[sy:sy + sh, sx:sx + sw]

        # Encode both faces and compare them
        captured_encodings = face_recognition.face_encodings(captured_face)
        stored_encodings = face_recognition.face_encodings(stored_face)

        if len(captured_encodings) == 0 or len(stored_encodings) == 0:
            return False, "Face encoding failed for one of the images."

        captured_face_encoding = captured_encodings[0]
        stored_face_encoding = stored_encodings[0]

        # Compare faces and calculate the confidence score
        face_distance = face_recognition.face_distance([stored_face_encoding], captured_face_encoding)[0]
        confidence_score = 1.0 - face_distance

        # Check if the confidence score is above 60%
        if confidence_score * 100 >= 60:
            return True, "Faces match! Validation successful."
        else:
            return False, "Validation failed. Confidence score too low."

    except Exception as e:
        return False, f"An error occurred during validation: {e}"