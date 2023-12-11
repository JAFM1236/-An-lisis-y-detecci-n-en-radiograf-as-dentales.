import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

def sliding_window(image, step_size, window_size):
    for y in range(0, image.shape[0] - window_size[1] + 1, step_size):
        for x in range(0, image.shape[1] - window_size[0] + 1, step_size):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

def predict_impacted_molar(model, image_patch):
    # Preprocess the image patch (resize, normalize, etc.)
    patch = cv2.resize(image_patch, (90, 90))
    patch = np.expand_dims(patch, axis=-1)
    patch = np.expand_dims(patch, axis=0)
    patch = patch / 255.0

    # Make predictions
    prediction = model.predict(patch)

    return prediction[0][0]

def non_max_suppression(boxes, overlap_threshold):
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    pick = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = boxes[:, 4]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    indices = np.argsort(scores)

    while len(indices) > 0:
        last = len(indices) - 1
        i = indices[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[indices[:last]])
        yy1 = np.maximum(y1[i], y1[indices[:last]])
        xx2 = np.minimum(x2[i], x2[indices[:last]])
        yy2 = np.minimum(y2[i], y2[indices[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[indices[:last]]

        indices = np.delete(indices, np.concatenate(([last], np.where(overlap > overlap_threshold)[0])))

    return boxes[pick]

def detect_impacted_molars(model, full_image, window_size=(100, 100), step_size=500, threshold=0.5, nms_threshold=0.3):
    detections = []

    for (x, y, window) in sliding_window(full_image, step_size, window_size):
        confidence = predict_impacted_molar(model, window)

        if confidence > threshold:
            detections.append([x, y, x + window_size[0], y + window_size[1], confidence])

    # Apply non-maximum suppression
    detections = non_max_suppression(detections, nms_threshold)

    return detections

# Load the trained model
model = load_model("teeth_classification_model_fixed.h5")

# Open a connection to the camera (0 is usually the default camera)
cap = cv2.VideoCapture(0)

# Create a CLAHE object
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE to enhance contrast
    enhanced_frame = clahe.apply(gray_frame)

    # Binarize the enhanced frame using thresholding
    _, binarized_frame = cv2.threshold(enhanced_frame, 128, 255, cv2.THRESH_BINARY)

    # Detect impacted molars in the binarized frame
    detections = detect_impacted_molars(model, binarized_frame)

    # Draw rectangles around the detected impacted molars
    for detection in detections:
        x, y, x2, y2, confidence = detection
        x, y, x2, y2 = int(x), int(y), int(x2), int(y2)  # Convert coordinates to integers
        cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)  # Green rectangle

    # Display the number of impacted molars in the frame
    cv2.putText(frame, f"Impacted Molars: {len(detections)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame with rectangles
    cv2.imshow('Detected Impacted Molars', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
