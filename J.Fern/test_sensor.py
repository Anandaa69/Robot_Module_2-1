# # -*- coding: utf-8 -*-
# """
# เดินสี่เหลี่ยมจัตุรัส + เก็บเซนเซอร์ RoboMaster EP ลง CSV
# – อาศัย ❶ การ subscribe (`sub_*`) รับข้อมูลจาก SDK  
# – ❷ เธรด​ `aggregator` รวมค่าล่าสุดทุก 1 วินาทีเป็น 1 แถว  
# """

# import csv
# import threading
# import time
# from robomaster import robot

# # ---------- ตั้งค่า ----------
# CSV_NAME       = "sensor_log.csv"
# SAMPLE_PERIOD  = 1.0       # วินาทีละ 1 แถว

# X_VAL, Z_VAL   = 0.6, 90   # ความยาวด้าน / มุมหมุน
# ROUNDS         = 1
# WALK_SPEED     = 0.7       # m/s
# TURN_SPEED     = 45        # °/s
# # -----------------------------

# latest = {}                # ค่าล่าสุดของทุกเซนเซอร์
# lock   = threading.Lock()  # ป้องกัน race‑condition
# log    = []                # เก็บแถว CSV


# # ---------- callback แต่ละเซนเซอร์ ----------
# def cb_pos(info):          # [x, y, z]
#     with lock:
#         latest.update(dict(pos_x=info[0], pos_y=info[1], pos_z=info[2]))

# def cb_att(info):          # [yaw, pitch, roll]
#     with lock:
#         latest.update(dict(yaw=info[0], pitch=info[1], roll=info[2]))

# def cb_imu(info):          # [vx, vy, vz, ax, ay, az] ตาม SDK
#     with lock:
#         latest.update(dict(gyro_x=info[0], gyro_y=info[1], gyro_z=info[2],
#                            acc_x=info[3],  acc_y=info[4],  acc_z=info[5]))

# def cb_esc(info):          # [speed, angle, ts, state]
#     speed, angle, ts, state = info
#     with lock:
#         latest.update(dict(esc_speed=speed, esc_angle=angle, esc_state=state))

# def cb_status(info):       # ค่า velocity ฯลฯ (ไม่คงที่ตามเวอร์ชัน)
#     # เก็บไว้ทั้งชุด โดยเติม prefix ว่า status_
#     with lock:
#         for idx, v in enumerate(info):
#             latest[f"status_{idx}"] = v
# # ---------------------------------------------


# def aggregator(stop_evt, start_time):
#     """รวบค่าเซนเซอร์ทุก SAMPLE_PERIOD วินาที"""
#     while not stop_evt.is_set():
#         time.sleep(SAMPLE_PERIOD)
#         with lock:
#             row = {"timestamp": round(time.time() - start_time, 3)}
#             row.update(latest)         # คัดลอกค่าล่าสุด
#         log.append(row)


# def save_csv(path, rows):
#     if not rows:
#         print("⚠ ไม่มีข้อมูลเซนเซอร์ให้บันทึก")
#         return
#     headers = sorted({k for r in rows for k in r})
#     with open(path, "w", newline="") as f:
#         csv.DictWriter(f, headers).writeheader()
#         csv.DictWriter(f, headers).writerows(rows)
#     print(f"✅ บันทึก {path} เรียบร้อย ({len(rows)} แถว)")


# def main():
#     ep_robot = robot.Robot()
#     ep_robot.initialize(conn_type="ap")
#     chassis = ep_robot.chassis

#     # ❶ subscribe เซนเซอร์
#     chassis.sub_position (freq=10, callback=cb_pos)     # ﻿10 Hz
#     chassis.sub_attitude (freq=10, callback=cb_att)
#     chassis.sub_imu      (freq=20, callback=cb_imu)
#     chassis.sub_esc      (freq=20, callback=cb_esc)
#     chassis.sub_status   (freq=50, callback=cb_status)  # ﻿50 Hz

#     # ❷ เธรดรวมข้อมูลทุก 1 วินาที
#     stop_evt = threading.Event()
#     start_t  = time.time()
#     th = threading.Thread(target=aggregator, args=(stop_evt, start_t), daemon=True)
#     th.start()

#     # ❸ เคลื่อนที่เป็นสี่เหลี่ยม
#     for r in range(ROUNDS):
#         print(f"🔰 รอบ {r+1}/{ROUNDS}")
#         for side in range(4):
#             chassis.move(x=X_VAL, y=0, z=0, xy_speed=WALK_SPEED).wait_for_completed()
#             time.sleep(0.3)  # ให้ IMU นิ่ง
#             chassis.move(x=0, y=0, z=Z_VAL, z_speed=TURN_SPEED).wait_for_completed()
#             time.sleep(0.3)
#             print(f"  ✅ ด้าน {side+1}/4 เสร็จ")

#     # ❹ ปิดทุกอย่าง
#     stop_evt.set(); th.join()
#     chassis.unsub_position(); chassis.unsub_attitude(); chassis.unsub_imu()
#     chassis.unsub_esc(); chassis.unsub_status()
#     ep_robot.close()

#     save_csv(CSV_NAME, log)


# if __name__ == "__main__":
#     main()
