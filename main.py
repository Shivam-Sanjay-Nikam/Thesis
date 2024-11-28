import time
import serial
import csv
import numpy as np
from sklearn.neighbors import KNeighborsClassifier  # KNN model import
from caliberate import SensorCalibrator
from collections import Counter
from write_program import process_gesture
from collections import deque

# Constants
TILT_THRESHOLD_ROLL = 20  # Tilt threshold for roll (Left/Right)
TILT_THRESHOLD_PITCH = 20  # Tilt threshold for pitch (Up/Down)
PROCESSING_INTERVAL = 0.2  # Time interval for processing gestures
GESTURE_BUFFER_TIME = 1.0  # Time window to consider for "major" gesture



def print_final_gesture(gesture):
    """Print the gesture if it occurs consecutively based on specified counts and only when it changes, except for 'Neutral'."""
    # Static variables to track the last detected and last printed gestures
    if not hasattr(print_final_gesture, "last_gesture"):
        print_final_gesture.last_gesture = None  # Initialize the last detected gesture
        print_final_gesture.last_printed_gesture = None  # Initialize the last printed gesture
        print_final_gesture.count = 0  # Initialize count for consecutive occurrences


    # Check if the current gesture matches the last detected gesture
    if gesture == print_final_gesture.last_gesture:
        print_final_gesture.count += 1
    else:
        print_final_gesture.last_gesture = gesture  # Update last detected gesture
        print_final_gesture.count = 1  # Reset count for a new gesture

    # Print "Circle" if it appears 3 times and is different from last printed gesture
    if gesture == "Circle" and print_final_gesture.count >= 3 and gesture != print_final_gesture.last_printed_gesture:
        print("Detected 'Circle' ")
        print_final_gesture.last_printed_gesture = gesture  # Update last printed gesture
        print_final_gesture.count = 0  # Reset count after printing
        # process_gesture(gesture)

    # For other gestures, print if it appears 8 times and is different from last printed gesture
    elif gesture != "Circle" and print_final_gesture.count >= 7 and gesture != print_final_gesture.last_printed_gesture:
        print(f"Detected '{gesture}'")
        print_final_gesture.last_printed_gesture = gesture  # Update last printed gesture
        print_final_gesture.count = 0  # Reset count after printing
        # process_gesture(gesture)



# Initialize and load training data for gestures
def load_training_data(filename='gesture_training_data.csv'):
    """Load gesture training data from a CSV file."""
    try:
        data, labels = [], []
        with open(filename, mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header row
            for row in reader:
                if len(row) >= 5:
                    tilt_data = list(map(int, row[:4]))
                    label = row[4]
                    data.append(tilt_data)
                    labels.append(label)
        print(f"Training data loaded from {filename}")
        return np.array(data), np.array(labels)
    except FileNotFoundError:
        print(f"Error: {filename} not found.")
        return np.array([]), np.array([])


# Train the K-Nearest Neighbors (KNN) model
def train_knn_classifier(data, labels):
    """Train a K-Nearest Neighbors (KNN) classifier."""
    try:
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(data, labels)
        print("KNN model trained successfully.")
        return knn
    except Exception as e:
        print(f"Error training KNN model: {e}")
        return None


# Detect tilt direction based on thresholds
def detect_tilt(corrected_roll, corrected_pitch):
    """Determine the tilt direction based on corrected roll and pitch values."""
    if corrected_roll > TILT_THRESHOLD_ROLL:
        return "Tilt Right"
    elif corrected_roll < -TILT_THRESHOLD_ROLL:
        return "Tilt Left"
    elif corrected_pitch > TILT_THRESHOLD_PITCH:
        return "Tilt Up"
    elif corrected_pitch < -TILT_THRESHOLD_PITCH:
        return "Tilt Down"
    return None


# Update the gesture map based on recent tilt data
def update_gesture_map(window):
    """Update the gesture map for a rolling window of tilt data."""
    gesture_map = {"Tilt Up": 0, "Tilt Down": 0, "Tilt Left": 0, "Tilt Right": 0}
    for roll, pitch in window:
        gesture = detect_tilt(roll, pitch)
        if gesture:
            gesture_map[gesture] += 1
    # print(gesture_map)
    return gesture_map


# Predict the gesture using the gesture map and KNN model
def predict_gesture(gesture_map, knn_model):
    """Predict the gesture based on the gesture map using the KNN model."""
    gesture_vector = [
        gesture_map["Tilt Up"],
        gesture_map["Tilt Down"],
        gesture_map["Tilt Left"],
        gesture_map["Tilt Right"]
    ]
    try:
        prediction = knn_model.predict([gesture_vector])[0]
        return prediction
    except Exception as e:
        print(f"Prediction error: {e}")
        return None


# Filter to get the major gesture in the last 1 second based on frequency
def get_major_gesture(gesture_predictions):
    """Return the major gesture (most frequent) from the recent predictions."""
    if not gesture_predictions:
        return None
    # Count frequency of gestures in the buffer
    counter = Counter(gesture_predictions)
    # Return the most common gesture
    major_gesture, _ = counter.most_common(1)[0]
    return major_gesture


# Receive sensor data, calibrate, and log gesture predictions
def receive_and_store_data(calibrator, knn_model):
    """Receive data from sensors, calibrate it, and log predicted gestures."""
    window = [[0, 0] for _ in range(10)]
    gesture_predictions = []  # Store predictions in the last GESTURE_BUFFER_TIME seconds
    last_processing_time = time.time()

    with open('mpu6050_results.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Corrected Roll", "Corrected Pitch", "Gesture Map", "Predicted Gesture"])

        try:
            initial_data = []
            while len(initial_data) < 100:
                if bluetooth.in_waiting > 0:
                    line = bluetooth.readline().decode('utf-8').strip()
                    data = line.split(',')
                    if len(data) == 4:
                        received_time, roll, pitch, yaw = map(float, data)
                        initial_data.append([received_time, roll, pitch, yaw])
            print("Initial data collection complete.")
            calibrator.calibrate(initial_data)
            print("Sensor data calibrated.")

            while True:
                if bluetooth.in_waiting > 0:
                    line = bluetooth.readline().decode('utf-8').strip()
                    data = line.split(',')
                    if len(data) == 4:
                        received_time, roll, pitch, yaw = map(float, data)
                        corrected_roll = calibrator.correct_value("anglex", roll)
                        corrected_pitch = calibrator.correct_value("angley", pitch)

                        window.append([corrected_roll, corrected_pitch])
                        if len(window) > 30:
                            window.pop(0)

                        current_time = time.time()
                        if current_time - last_processing_time >= PROCESSING_INTERVAL:
                            gesture_map = update_gesture_map(window)
                            predicted_gesture = predict_gesture(gesture_map, knn_model)

                            # Collect gesture predictions
                            gesture_predictions.append(predicted_gesture)

                            # Limit buffer size to gestures within GESTURE_BUFFER_TIME seconds
                            if len(gesture_predictions) > int(GESTURE_BUFFER_TIME / PROCESSING_INTERVAL):
                                gesture_predictions.pop(0)

                            # Get major gesture from the buffer
                            major_gesture = get_major_gesture(gesture_predictions)

                            writer.writerow(
                                [round(corrected_roll), round(corrected_pitch), gesture_map, major_gesture])
                            # print(f"Major Gesture: {major_gesture}")

                            print_final_gesture(major_gesture)

                            last_processing_time = current_time
                time.sleep(0.01)
        except KeyboardInterrupt:
            print("Data collection stopped by user.")
        except Exception as e:
            print(f"Error in data reception: {e}")
        finally:
            bluetooth.close()
            print("Bluetooth connection closed.")


# Main execution script
try:
    bluetooth = serial.Serial('/dev/tty.ESP32_MPU6050', 115200, timeout=1)
    print("Connected to ESP32 via Bluetooth")
except serial.SerialException as e:
    print(f"Error connecting to ESP32: {e}")
    exit()

calibrator = SensorCalibrator()
data, labels = load_training_data('gesture_training_data.csv')
if data.size > 0 and labels.size > 0:
    knn_model = train_knn_classifier(data, labels)
    if knn_model:
        receive_and_store_data(calibrator, knn_model)
else:
    print("Insufficient training data to proceed.")
