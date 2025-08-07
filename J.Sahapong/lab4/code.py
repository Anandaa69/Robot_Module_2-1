
import robomaster
from robomaster import robot
import time
import csv
from datetime import datetime

csv_filename = "tof_data30_3.csv"

def sub_data_handler(sub_info):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    distance = sub_info
    # เขียนข้อมูลลงไฟล์ csv ทันที (append)
    with open(csv_filename, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([now, distance[0], distance[1], distance[2], distance[3]])
    print("tof1:{0}  tof2:{1}  tof3:{2}  tof4:{3}".format(distance[0], distance[1], distance[2], distance[3]))

if __name__ == '__main__':
    # สร้างไฟล์และเขียน header ก่อนเริ่ม
    with open(csv_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["timestamp", "tof1", "tof2", "tof3", "tof4"])

    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")

    ep_sensor = ep_robot.sensor
    ep_sensor.sub_distance(freq=20, callback=sub_data_handler)
    # while 1 :
    #     x =0 
    time.sleep(30)
    ep_sensor.unsub_distance()
    ep_robot.close()