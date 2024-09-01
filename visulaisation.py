import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model('best_hand_sign_model.h5')

# Print the model summary to check the input shape
model.summary()

def preprocess_frame(frame):
    # Convert to grayscale if needed
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Resize to match the model input size (28x28)
    resized = cv2.resize(gray, (28, 28))
    # Normalize pixel values
    normalized = resized / 255.0
    # Reshape to include batch dimension and single channel
    return normalized.reshape(1, 28, 28, 1)

def classify_hand_sign(frame):
    preprocessed = preprocess_frame(frame)
    prediction = model.predict(preprocessed)
    
    # Determine the predicted class
    max_index = np.argmax(prediction[0])
    label = chr(max_index + 65) if prediction[0][max_index] > 0.7 else "Not Relevant"
    return label

# Real-time hand sign detection using OpenCV
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame and classify
    hand_sign_label = classify_hand_sign(frame)
    
    # Display the prediction on the frame
    cv2.putText(frame, hand_sign_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Show the frame with prediction
    cv2.imshow('Hand Sign Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
