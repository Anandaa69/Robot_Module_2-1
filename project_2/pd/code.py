from robomaster import robot
import time
import csv
import math

# CONFIG
TARGET_X = 2.0
KP = 10
KD = 2# เพิ่มค่าพารามิเตอร์ Derivative
TOLERANCE = 0.01
CONTROL_INTERVAL = 0.1
MAX_WHEEL_SPEED = 150

current_x = 0.0
previous_error = 0.0    # สำหรับคำนวณ Derivative
start_time = time.time()

# เปิดไฟล์ CSV สำหรับเขียนข้อมูล และเขียน header
csv_file = open(f"project_2/pd_p_{KP}_d_{int(KD)}_{int(math.floor(KD * 10))}.csv", mode="w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["elapsed_time", "current_x", "error", "derivative", "speed"])

def sub_position_handler(position_info):
    global current_x
    x, y, z = position_info
    current_x = x
    elapsed_time = time.time() - start_time
    print("x = {:.2f}, y = {:.2f}, z = {:.2f} | time: {:.2f}s".format(x, y, z, elapsed_time))

if __name__ == '__main__':
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")

    ep_chassis = ep_robot.chassis
    ep_chassis.sub_position(freq=10, callback=sub_position_handler)

    while True:
        error = TARGET_X - current_x
        derivative = (error - previous_error) / CONTROL_INTERVAL
        speed = KP * error + KD * derivative

        # จำกัดความเร็วไม่ให้เกินค่าที่กำหนด
        speed = max(min(speed, MAX_WHEEL_SPEED), -MAX_WHEEL_SPEED)

        elapsed_time = time.time() - start_time
        csv_writer.writerow([elapsed_time, current_x, error, derivative, speed])
        csv_file.flush()

        print("Error: {:.3f}, Derivative: {:.3f}, Speed: {:.2f}".format(error, derivative, speed))
        print("speed:", speed)

        if abs(error) <= TOLERANCE:
            print("✅ ถึงเป้าหมายแล้ว!")
            break

        ep_chassis.drive_speed(x=speed, y=0, z=0, timeout=5)
        time.sleep(CONTROL_INTERVAL)

        previous_error = error

    ep_chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0)
    csv_file.close()