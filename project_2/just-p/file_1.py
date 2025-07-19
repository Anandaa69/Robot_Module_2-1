from robomaster import robot
import time
import csv

# CONFIG
TARGET_X = 2.0
KP = 2.5
TOLERANCE = 0.01
CONTROL_INTERVAL = 0.1
MAX_WHEEL_SPEED = 80

current_x = 0.0
start_time = time.time()  # เริ่มจับเวลา ณ จุดเริ่มต้น

# เปิดไฟล์ CSV สำหรับเขียนข้อมูล และเขียน header
csv_file = open("project_2/kp_2_5.csv", mode="w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["elapsed_time", "current_x"])

def sub_position_handler(position_info):
    global current_x
    x, y, z = position_info
    current_x = x
    # เวลาที่ผ่านไปจากเวลาเริ่มต้น
    elapsed_time = time.time() - start_time
    csv_writer.writerow([elapsed_time, current_x])
    csv_file.flush()
    print("x = {:.2f}, y = {:.2f}, z = {:.2f} | ⏱ time: {:.2f}s".format(x, y, z, elapsed_time))

if __name__ == '__main__':
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")

    ep_chassis = ep_robot.chassis
    ep_chassis.sub_position(freq=10, callback=sub_position_handler)

    while True:
        error = TARGET_X - current_x
        print("Error:", error)
        if abs(error) <= TOLERANCE:
            print("✅ ถึงเป้าหมายแล้ว!")
            break

        speed = KP * error
        ep_chassis.drive_speed(x=speed, y=0, z=0, timeout=5)
        time.sleep(CONTROL_INTERVAL)

    ep_chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0)
    csv_file.close()