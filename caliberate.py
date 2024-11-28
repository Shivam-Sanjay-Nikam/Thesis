import pandas as pd
import numpy as np


class SensorCalibrator:
    def __init__(self):
        """
        Initializes the SensorCalibrator object with default mean values and an empty data buffer.
        """
        self.means = {
            "anglex": 0.0,
            "angley": 0.0,
            "anglez": 0.0
        }
        self.calibrated = False
        self.data_buffer = []  # Buffer to store data points before calibration

    def calibrate(self, data):
        """
        Calibrate the sensors based on the provided data.

        Parameters:
            data (list of lists): A list of sensor readings with columns for Time, anglex, angley, and anglez.
        """
        try:
            # Convert raw data to DataFrame for easier processing
            df = pd.DataFrame(data, columns=["Time", "anglex", "angley", "anglez"])

            # Calculate the mean values for each sensor axis
            self.means["anglex"] = df["anglex"].mean()
            self.means["angley"] = df["angley"].mean()
            self.means["anglez"] = df["anglez"].mean()

            self.calibrated = True
            print(f"Calibration complete. Mean values: {self.means}")
        except Exception as e:
            print(f"Error during calibration: {e}")

    def add_data_point(self, time_value, roll, pitch, yaw):
        """
        Add a data point to the buffer for calibration.

        Parameters:
            time_value (float): The timestamp of the data point.
            roll (float): The roll angle of the sensor.
            pitch (float): The pitch angle of the sensor.
            yaw (float): The yaw angle of the sensor.
        """
        try:
            if not self.calibrated:
                # Store data points if calibration has not been performed yet
                self.data_buffer.append([time_value, roll, pitch, yaw])

                # Limit the buffer size to 100 points
                if len(self.data_buffer) > 100:
                    self.data_buffer.pop(0)

                print(f"Data point added. Buffer size: {len(self.data_buffer)}")
            else:
                print("Calibration already complete. Data points will not be added.")
        except Exception as e:
            print(f"Error while adding data point: {e}")

    def correct_value(self, sensor_name, sensor_value):
        """
        Correct the sensor value using the calibration means.

        Parameters:
            sensor_name (str): The name of the sensor axis ('anglex', 'angley', 'anglez').
            sensor_value (float): The raw value from the sensor.

        Returns:
            float: The corrected sensor value.
        """
        try:
            if not self.calibrated:
                print("Error: Calibration not completed yet. Cannot correct values.")
                return sensor_value

            if sensor_name not in self.means:
                print(
                    f"Error: Invalid sensor name '{sensor_name}'. Available sensors are: 'anglex', 'angley', 'anglez'.")
                return sensor_value

            # Apply correction to the sensor value
            correction = self.means[sensor_name]
            corrected_value = sensor_value - correction

            return corrected_value
        except Exception as e:
            print(f"Error during value correction: {e}")
            return sensor_value
