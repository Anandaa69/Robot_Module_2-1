
# import csv
# import threading
# import time
# import pandas as pds
# from robomaster import robot

# CSV_NAME   = "sensor_log.csv"
# XLSX_NAME  = "sensor_log.xlsx"
# SAMPLE_PERIOD = 1.0

# X_VAL, Z_VAL = 0.6, 90
# ROUNDS       = 1
# WALK_SPEED   = 0.7
# TURN_SPEED   = 45

# latest, lock, log = {}, threading.Lock(), []

# def cb_pos(info):
#     with lock:
#         latest.update(dict(pos_x=info[0], pos_y=info[1], pos_z=info[2]))

# def cb_att(info):
#     with lock:
#         latest.update(dict(yaw=info[0], pitch=info[1], roll=info[2]))

# def cb_imu(info):
#     with lock:
#         latest.update(dict(gyro_x=info[0], gyro_y=info[1], gyro_z=info[2],
#                            acc_x=info[3],  acc_y=info[4],  acc_z=info[5]))

# def cb_esc(info):
#     speed, angle, ts, state = info
#     with lock:
#         latest.update(dict(esc_speed=speed, esc_angle=angle, esc_state=state))

# def cb_status(info):
#     with lock:
#         for idx, v in enumerate(info):
#             latest[f"status_{idx}"] = v


# def aggregator(stop_evt, t0):
#     while not stop_evt.is_set():
#         time.sleep(SAMPLE_PERIOD)
#         with lock:
#             row = {"timestamp": round(time.time() - t0, 3)}
#             row.update(latest)
#         log.append(row)

# def save_csv(path, rows):
#     if not rows:
#         print("ไม่มีข้อมูลเซนเซอร์ให้บันทึก")
#         return False
#     headers = sorted({k for r in rows for k in r})
#     with open(path, "w", newline="") as f:
#         w = csv.DictWriter(f, headers)
#         w.writeheader(); w.writerows(rows)
#     print(f"✅ บันทึก {path} ({len(rows)} แถว)")
#     return True


# def csv_to_excel(csv_path, xlsx_path):
#     df = pd.read_csv(csv_path)
#     df["timestamp"] = pd.to_timedelta(df["timestamp"], unit="s")
#     df.to_excel(xlsx_path, index=False)
#     print(f"แปลงเป็น {xlsx_path} เรียบร้อย")


# def main():
#     ep_robot = robot.Robot()
#     ep_robot.initialize(conn_type="ap")
#     chassis = ep_robot.chassis

#     # subscribe
#     chassis.sub_position (freq=10, callback=cb_pos)
#     chassis.sub_attitude (freq=10, callback=cb_att)
#     chassis.sub_imu      (freq=20, callback=cb_imu)
#     chassis.sub_esc      (freq=20, callback=cb_esc)
#     chassis.sub_status   (freq=50, callback=cb_status)

#     stop_evt = threading.Event()
#     t0 = time.time()
#     threading.Thread(target=aggregator, args=(stop_evt, t0), daemon=True).start()

#     # move
#     for r in range(ROUNDS):
#         print(f"🔰 รอบ {r+1}/{ROUNDS}")
#         for side in range(4):
#             chassis.move(x=X_VAL, y=0, z=0, xy_speed=WALK_SPEED).wait_for_completed()
#             time.sleep(0.3)
#             chassis.move(x=0, y=0, z=Z_VAL, z_speed=TURN_SPEED).wait_for_completed()
#             time.sleep(0.3)
#             print(f"  ✅ ด้าน {side+1}/4 เสร็จ")

#     # shutdown
#     stop_evt.set()
#     chassis.unsub_position(); chassis.unsub_attitude(); chassis.unsub_imu()
#     chassis.unsub_esc(); chassis.unsub_status()
#     ep_robot.close()

#     if save_csv(CSV_NAME, log):
#         csv_to_excel(CSV_NAME, XLSX_NAME)


# if __name__ == "__main__":
#     main()
