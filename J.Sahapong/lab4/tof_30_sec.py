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

# Global variable to store the latest ToF sensor reading
latest_tof = None


def write_tof_data_to_csv():
    """Writes the current ToF sensor data to the CSV file."""
    global csv_writer
    if csv_writer and latest_tof is not None:
        timestamp = datetime.now().isoformat()
        # Write the latest ToF value to a single row
        csv_writer.writerow(
            [
                timestamp,
                latest_tof,
            ]
        )
        # print(f"Data written to CSV: {timestamp}, ToF: {latest_tof}")


def tof_data_handler(sub_info):
    """Callback function for ToF distance subscription."""
    global latest_tof
    distance = sub_info
    tof_value = distance[0]
    latest_tof = tof_value
    print(f"TOF: {tof_value}")
    write_tof_data_to_csv()  # Write data when ToF updates


if __name__ == "__main__":
    ep_robot = robot.Robot()
    # Initialize the robot connection (e.g., "ap" for Access Point mode)
    ep_robot.initialize(conn_type="ap")

    ep_sensor = ep_robot.sensor

    # Open CSV file for writing
    try:
        csv_file = open("J.Sahapong/lab4/data/tof_paper_1.csv", "w", newline="")
        csv_writer = csv.writer(csv_file)
        # Write CSV header for ToF data
        csv_writer.writerow(
            [
                "timestamp_iso",
                "tof",
            ]
        )
        print("CSV file 'tof_sensor_data.csv' opened for writing.")
    except IOError as e:
        print(f"Error opening CSV file: {e}")
        ep_robot.close()
        exit()

    print("Starting ToF sensor data collection...")
    # Subscribe to ToF distance data at a frequency of 5 Hz
    ep_sensor.sub_distance(freq=20, callback=tof_data_handler)
    print("ToF subscription active. Data will be logged.")

    # Keep the script running to collect data for a duration
    # You can adjust this duration or add other robot movements here.
    print("Collecting ToF data for 30 seconds...")
    time.sleep(30) # Collect data for 20 seconds

    print("Stopping ToF sensor data collection.")
    # Unsubscribe from the ToF sensor
    ep_sensor.unsub_distance()

    # Close the robot connection
    ep_robot.close()

    # Close the CSV file
    if csv_file:
        csv_file.close()
        print(
            "ToF sensor data collection complete. Data saved to 'tof_sensor_data.csv'."
        )
