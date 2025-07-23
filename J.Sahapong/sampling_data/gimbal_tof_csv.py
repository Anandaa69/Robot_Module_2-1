# -*-coding:utf-8-*-
# Copyright (c) 2020 DJI.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License in the file LICENSE.txt or at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import robomaster
from robomaster import robot
import csv
from datetime import datetime
import time

# Global variables for CSV writer and file
csv_writer = None
csv_file = None

# Global variables to store the latest sensor readings
latest_tof = None
latest_pitch_angle = None
latest_yaw_angle = None
latest_pitch_ground_angle = None
latest_yaw_ground_angle = None


def write_combined_data_to_csv():
    """Writes the current combined sensor data to the CSV file."""
    global csv_writer
    if csv_writer:
        timestamp = datetime.now().isoformat()
        # Write all latest values to a single row
        csv_writer.writerow(
            [
                timestamp,
                latest_tof,
                latest_pitch_angle,
                latest_yaw_angle,
                latest_pitch_ground_angle,
                latest_yaw_ground_angle,
            ]
        )


def tof_data_handler(sub_info):
    """Callback function for ToF distance subscription."""
    global latest_tof
    distance = sub_info
    tof_value = distance[0]
    latest_tof = tof_value
    print(f"TOF: {tof_value}")
    write_combined_data_to_csv()  # Write data when ToF updates


def gimbal_data_handler(angle_info):
    """Callback function for gimbal angle subscription."""
    global latest_pitch_angle, latest_yaw_angle, latest_pitch_ground_angle, latest_yaw_ground_angle
    pitch_angle, yaw_angle, pitch_ground_angle, yaw_ground_angle = angle_info
    latest_pitch_angle = pitch_angle
    latest_yaw_angle = yaw_angle
    latest_pitch_ground_angle = pitch_ground_angle
    latest_yaw_ground_angle = yaw_ground_angle
    print(f"Gimbal Angle: pitch={pitch_angle}, yaw={yaw_angle}")
    write_combined_data_to_csv()  # Write data when gimbal angle updates


if __name__ == "__main__":
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")

    ep_gimbal = ep_robot.gimbal
    ep_sensor = ep_robot.sensor

    # Open CSV file for writing
    try:
        csv_file = open("combined_sensor_data.csv", "w", newline="")
        csv_writer = csv.writer(csv_file)
        # Write CSV header for all data points
        csv_writer.writerow(
            [
                "timestamp_iso",
                "tof",
                "gimbal_pitch_angle",
                "gimbal_yaw_angle",
                "gimbal_pitch_ground_angle",
                "gimbal_yaw_ground_angle",
            ]
        )
        print("CSV file 'combined_sensor_data.csv' opened for writing.")
    except IOError as e:
        print(f"Error opening CSV file: {e}")
        ep_robot.close()
        exit()

    print("Moving gimbal to initial position (pitch=0, yaw=90)...")
    ep_gimbal.moveto(pitch=0, yaw=90).wait_for_completed()
    time.sleep(5)  # Give some time for the gimbal to settle at 90 degrees

    print("Starting sensor data collection (ToF and Gimbal angles)...")
    # Subscribe to both ToF distance and gimbal angle data
    ep_sensor.sub_distance(freq=5, callback=tof_data_handler)
    ep_gimbal.sub_angle(freq=5, callback=gimbal_data_handler)
    print("Subscriptions active. Data will be logged while moving.")

    print(
        "Moving gimbal to pitch=0, yaw=-90 (data will be collected during movement)..."
    )
    ep_gimbal.moveto(pitch=0, yaw=-90).wait_for_completed()
    time.sleep(5)  # Allow some time for data to be logged at -90 degrees

    print("Stopping sensor data collection.")
    # Unsubscribe from both sensors
    ep_sensor.unsub_distance()
    ep_gimbal.unsub_angle()

    print("Resetting gimbal to pitch=0, yaw=0...")
    ep_gimbal.moveto(
        pitch=0, yaw=0, pitch_speed=100, yaw_speed=100
    ).wait_for_completed()
    time.sleep(1)  # Small delay after reset

    ep_robot.close()

    # Close the CSV file
    if csv_file:
        csv_file.close()
        print(
            "Sensor data collection complete. Data saved to 'combined_sensor_data.csv'."
        )
