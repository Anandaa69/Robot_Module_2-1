import math
import time
import csv  # ใช้สำหรับบันทึกไฟล์ CSV
import os   # ใช้สำหรับจัดการโฟลเดอร์
from robomaster import robot
import matplotlib.pyplot as plt  # ใช้สำหรับวาดกราฟเส้นทาง

# -------------------- ค่าพารามิเตอร์เริ่มต้น --------------------

position = [0, 0, 0]  # พิกัดเริ่มต้น (X, Y, Z)

# เป้าหมายระยะทางที่หุ่นยนต์ต้องเดิน (หน่วย: เมตร)
TARGET_DISTANCE = 0.6
TOLERANCE = 0.02  # ค่าความคลาดเคลื่อนที่ยอมรับได้

# ค่าคงที่ของ PID Controller
kp = 1.0
ki = 0
kd = 0
max_speed = 10         # ความเร็วสูงสุดที่อนุญาต
max_integral = 5.0     # จำกัดค่าอินทิกรัลไม่ให้สะสมเกินไป

# -------------------- ตัวแปรควบคุม --------------------

start_time = None            # เวลาเริ่มต้นการเดิน
start_x = None               # ตำแหน่งเริ่มต้น X
start_y = None               # ตำแหน่งเริ่มต้น Y
previous_error = None        # ค่าความคลาดเคลื่อนรอบก่อนหน้า
previous_time = None         # เวลารอบก่อนหน้า
integral_error = 0.0         # ผลรวมของ error (ส่วน I)
pid_control_active = False   # Flag สำหรับเปิด/ปิดการควบคุม PID

# เก็บข้อมูลเส้นทางและ log
xy_paths = [[] for _ in range(4)]           # เก็บตำแหน่งแยกตามรอบ (4 รอบ)
error_log_per_round = [[] for _ in range(4)]  # เก็บ log แยกตามรอบ
current_round = 0                            # ตัวแปรบอกว่าอยู่รอบที่เท่าไหร่ (0-3)

# สร้างโฟลเดอร์สำหรับเก็บไฟล์ CSV (ถ้ายังไม่มี)
log_folder = "robot_logs_P"
if not os.path.exists(log_folder):
    os.makedirs(log_folder)

# -------------------- ฟังก์ชัน Callback ตำแหน่ง --------------------

def sub_position_handler(position_info):
    """
    ฟังก์ชันนี้จะถูกเรียกทุกครั้งที่หุ่นยนต์อัปเดตตำแหน่ง
    จะคำนวณ PID และสั่งความเร็วให้หุ่นยนต์ พร้อมเก็บข้อมูลลง log
    """
    global position, start_time, start_x, start_y
    global previous_error, previous_time, integral_error, pid_control_active
    global xy_paths, error_log_per_round, current_round

    position = position_info
    x = position[0]
    y = position[1]

    # บันทึกตำแหน่งเดิน (x, y) ของรอบปัจจุบัน
    if pid_control_active:
        xy_paths[current_round].append((x, y))

    # ถ้ากำลังควบคุม PID อยู่
    if start_time and start_x is not None and start_y is not None and pid_control_active:
        current_time = time.time()
        elapsed = current_time - start_time  # เวลาที่ผ่านไป

        # คำนวณระยะทางจากตำแหน่งเริ่มต้น
        current_distance = math.sqrt((x - start_x) * 2 + (y - start_y) * 2)
        current_error = TARGET_DISTANCE - current_distance  # ความคลาดเคลื่อน

        # PID: ส่วน P
        p_value = kp * current_error

        # PID: ส่วน I (Integral)
        dt = current_time - previous_time if previous_time else 0
        if dt > 0:
            integral_error += current_error * dt
            # จำกัด integral ไม่ให้เกิน max_integral
            integral_error = max(min(integral_error, max_integral), -max_integral)
        i_value = ki * integral_error

        # PID: ส่วน D (Derivative)
        d_value = 0
        if previous_error is not None and dt > 0:
            derror = current_error - previous_error
            d_value = kd * (derror / dt)

        # รวม PID ทั้งหมดเพื่อได้ความเร็ว
        pid_speed = p_value + i_value + d_value
        # จำกัดความเร็วไม่ให้เกิน max_speed
        pid_speed = max(min(pid_speed, max_speed), -max_speed)

        # บันทึก log: เวลา, ตำแหน่ง, ความคลาดเคลื่อน, ความเร็ว (แยกตามรอบ)
        error_log_per_round[current_round].append([elapsed, x, y, current_error, pid_speed])

        # แสดงผลการควบคุม
        print(f"Time: {elapsed:.2f}s | Dist: {current_distance:.3f} | Err: {current_error:.3f} | "
            f"P: {p_value:.2f}, I: {i_value:.2f}, D: {d_value:.2f} → Speed: {pid_speed:.2f}")

        # สั่งหุ่นยนต์ให้เดินด้วยความเร็วที่ได้จาก PID
        if abs(current_error) > TOLERANCE:
            ep_chassis.drive_speed(x=pid_speed, y=0, z=0)
        else:
            # หยุดถ้าถึงระยะเป้าหมาย
            ep_chassis.drive_speed(x=0, y=0, z=0)
            pid_control_active = False
            print("🎯 ถึงเป้าหมายแล้ว")

        previous_error = current_error
        previous_time = current_time

# -------------------- บันทึก Log ลง CSV แยกตามรอบ --------------------

def save_path_to_csv_per_round(round_num):
    """
    บันทึกข้อมูล log ของแต่ละรอบลงไฟล์ CSV แยกไฟล์ตามรอบ
    """
    filename = os.path.join(log_folder, f"robot_path_log_round_P{round_num + 1}.csv")
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Time (s)", "X (m)", "Y (m)", "Error (m)", "Speed"])
        writer.writerows(error_log_per_round[round_num])
    print(f"💾 บันทึก log รอบที่ {round_num + 1} ลงไฟล์ {filename} แล้ว")

# -------------------- ฟังก์ชันหลัก --------------------

if _name_ == "_main_":
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")  # เริ่มเชื่อมต่อ RoboMaster
    ep_chassis = ep_robot.chassis

    print("🔌 กำลังเชื่อมต่อกับ RoboMaster...")
    ep_chassis.sub_position(freq=10, callback=sub_position_handler)  # subscribe ตำแหน่ง
    time.sleep(2)

    # เดินเป็นสี่เหลี่ยม 4 รอบ
    for round_num in range(4):
        current_round = round_num  # บอก callback ว่ากำลังเป็นรอบที่เท่าไหร่
        print(f"\n🔁 เริ่มรอบที่ {current_round + 1}/4")

        for i in range(4):
            print(f"\n===== ด้านที่ {i + 1}/4 ของรอบที่ {current_round + 1} =====")

            # รอจนได้ตำแหน่งเริ่มต้นที่ไม่ใช่ [0,0,0]
            while position == [0, 0, 0]:
                time.sleep(0.01)

            # ตั้งค่าตำแหน่งและเวลาเริ่มต้นสำหรับ PID
            start_x, start_y, _ = position
            start_time = time.time()
            previous_error = TARGET_DISTANCE
            previous_time = time.time()
            integral_error = 0.0
            pid_control_active = True

            timeout = 30  # หยุด PID ถ้าเกิน 30 วินาที
            start_pid_time = time.time()

            # รอจน PID ควบคุมเสร็จหรือ timeout
            while pid_control_active and (time.time() - start_pid_time < timeout):
                time.sleep(0.05)

            # หยุดล้อก่อนหมุน
            ep_chassis.drive_speed(x=0, y=0, z=0)
            time.sleep(1)

            if not pid_control_active:
                print("✅ เดินตรงเสร็จสิ้น")
            else:
                print("⚠️ หมดเวลา PID Controller")

            # หมุน 90 องศา (ถ้ายังไม่ใช่ด้านสุดท้ายของรอบ)
            if i < 4:  # หมุน 3 ครั้งใน 4 ด้าน (ไม่หมุนหลังด้านสุดท้าย)
                print("↪️ กำลังหมุน 90 องศา...")
                ep_chassis.move(x=0, y=0, z=90, z_speed=45).wait_for_completed()
                time.sleep(1)

        print(f"🏁 จบการเดินรอบที่ {current_round + 1}")

        # บันทึกไฟล์ CSV ของรอบนี้
        save_path_to_csv_per_round(current_round)

    # -------------------- วาดกราฟเส้นทาง --------------------
    colors = ['blue', 'green', 'red', 'orange']  # สีสำหรับแต่ละรอบ

    plt.figure(figsize=(6, 6))
    for i, path in enumerate(xy_paths):
        if path:  # ถ้ามีข้อมูล
            xs, ys = zip(*path)
            plt.plot(xs, ys, marker='o', label=f'round {i + 1}', color=colors[i % len(colors)])

    plt.title("Robot Path (4 Square Rounds - Color by Round)")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.show()

    # -------------------- ปิดการเชื่อมต่อ --------------------
    print("📴 กำลังปิดการเชื่อมต่อ...")
    ep_chassis.unsub_position()
    ep_robot.close()
    print("✅ เสร็จสิ้น!")