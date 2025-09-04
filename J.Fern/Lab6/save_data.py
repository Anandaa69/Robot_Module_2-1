# run_robot_and_save_data_final.py
import cv2
import numpy as np
import time
import math
import pandas as pd
from robomaster import robot
from robomaster import camera

# --- ส่วนสำหรับรับข้อมูลมุม Gimbal ---
list_of_data = [0, 0, 0, 0]
def sub_data_handler(angle_info):
    global list_of_data
    list_of_data = angle_info

# --- ส่วนของฟังก์ชันการตรวจจับ (ไม่เปลี่ยนแปลง) ---
def create_pink_mask(img_rgb):
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower_pink = np.array([120, 10, 100])
    upper_pink = np.array([170, 100, 200])
    mask = cv2.inRange(hsv, lower_pink, upper_pink)
    mask = cv2.medianBlur(mask, 5)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def match_template_masked(img_masked, tmpl_masked, threshold=0.7):
    if tmpl_masked.shape[0] > img_masked.shape[0] or tmpl_masked.shape[1] > img_masked.shape[1]: return []
    result = cv2.matchTemplate(img_masked, tmpl_masked, cv2.TM_CCOEFF_NORMED)
    locations = np.where(result >= threshold)
    boxes = []
    h, w = tmpl_masked.shape
    for pt in zip(*locations[::-1]):
        confidence = result[pt[1], pt[0]]
        boxes.append(((pt[0], pt[1]), (pt[0] + w, pt[1] + h), confidence))
    return boxes

def calculate_iou(box1, box2):
    x1_1, y1_1 = box1[0]; x2_1, y2_1 = box1[1]
    x1_2, y1_2 = box2[0]; x2_2, y2_2 = box2[1]
    x1_i = max(x1_1, x1_2); y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2); y2_i = min(y2_1, y2_2)
    intersection_area = max(0, x2_i - x1_i) * max(0, y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1); area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - intersection_area
    return intersection_area / union_area if union_area > 0 else 0.0

def advanced_prioritized_nms(boxes_with_scores, iou_threshold=0.3):
    if not boxes_with_scores: return []
    detections = sorted(boxes_with_scores, key=lambda x: x[3])
    final_boxes = []
    while detections:
        current_box = detections.pop(0)
        final_boxes.append(current_box)
        remaining_boxes = []
        for box_to_check in detections:
            iou = calculate_iou(current_box, box_to_check)
            center_x = box_to_check[0][0] + (box_to_check[1][0] - box_to_check[0][0]) / 2
            center_y = box_to_check[0][1] + (box_to_check[1][1] - box_to_check[0][1]) / 2
            is_center_inside = (current_box[0][0] < center_x < current_box[1][0] and current_box[0][1] < center_y < current_box[1][1])
            if iou > iou_threshold or is_center_inside: continue
            remaining_boxes.append(box_to_check)
        detections = remaining_boxes
    return final_boxes

# --- ส่วนการทำงานหลัก ---
def main():
    # --- ค่าคงที่และ Template ---
    PROCESSING_WIDTH = 640.0
    TEMPLATE_FILES = ["image/template/use/template_night_pic1_x_557_y_278_w_120_h_293.jpg", "image/template/use/template_night_pic2_x_609_y_290_w_57_h_138.jpg", "image/template/use/template_night_pic3_x_622_y_293_w_40_h_93.jpg"]
    MATCH_THRESHOLD, IOU_THRESHOLD, Y_AXIS_ADJUSTMENT = 0.55, 0.1, 25
    p, i, d = -0.607, 0, -0.00135

    # --- ส่วนการเชื่อมต่อและตั้งค่าหุ่นยนต์ ---
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")
    ep_camera, ep_gimbal = ep_robot.camera, ep_robot.gimbal
    ep_camera.start_video_stream(display=False, resolution=camera.STREAM_720P)
    ep_gimbal.sub_angle(freq=50, callback=sub_data_handler)
    ep_gimbal.recenter(pitch_speed=200, yaw_speed=200).wait_for_completed()

    frame_for_scale = ep_camera.read_cv2_image(timeout=5)
    if frame_for_scale is None: print("Error: Could not get frame."); ep_robot.close(); return
    h_orig, w_orig, _ = frame_for_scale.shape
    center_x, center_y = w_orig / 2, h_orig / 2
    processing_scale = PROCESSING_WIDTH / w_orig
    h_proc = int(h_orig * processing_scale)
    templates_masked = []
    for f in TEMPLATE_FILES:
        tmpl = cv2.imread(f)
        if tmpl is None: continue
        w_tmpl, h_tmpl = int(tmpl.shape[1] * processing_scale), int(tmpl.shape[0] * processing_scale)
        tmpl_resized = cv2.resize(tmpl, (w_tmpl, h_tmpl), interpolation=cv2.INTER_AREA)
        tmpl_pink_mask = create_pink_mask(cv2.cvtColor(tmpl_resized, cv2.COLOR_BGR2RGB))
        tmpl_gray = cv2.cvtColor(tmpl_resized, cv2.COLOR_BGR2GRAY)
        templates_masked.append(cv2.bitwise_and(tmpl_gray, tmpl_gray, mask=tmpl_pink_mask))

    # --- ตัวแปรสถานะ PID และการเก็บข้อมูล ---
    count = 0
    time.sleep(1)
    accumulate_err_x, accumulate_err_y = 0, 0
    data_for_saving = []
    prev_time = time.time()
    start_time = time.time() 
    
    # [จุดที่แก้ไข 1] เพิ่มตัวแปรสำหรับแสดงผล
    last_display_time = time.time()
    
    print("Starting detection... Press 'q' in the display window to stop.")

    try:
        while True:
            frame = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
            if frame is None: continue
            
            current_time = time.time()

            # --- ส่วนการตรวจจับ ---
            frame_proc = cv2.resize(frame, (int(PROCESSING_WIDTH), h_proc), interpolation=cv2.INTER_AREA)
            main_pink_mask = create_pink_mask(cv2.cvtColor(frame_proc, cv2.COLOR_BGR2RGB))
            main_gray = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2GRAY)
            main_masked = cv2.bitwise_and(main_gray, main_gray, mask=main_pink_mask)

            all_detections = []
            # [จุดที่แก้ไข 2] นำ `colors` กลับมาเพื่อใช้กับ Bounding Box
            colors = [(0, 255, 0), (255, 165, 0), (0, 0, 255)]
            
            for idx, tmpl_masked in enumerate(templates_masked):
                boxes = match_template_masked(main_masked, tmpl_masked, threshold=MATCH_THRESHOLD)
                for top_left, bottom_right, confidence in boxes:
                    # เพิ่ม color และ template_id (idx) เข้าไป
                    all_detections.append((top_left, bottom_right, confidence, idx, colors[idx]))

            final_detections = advanced_prioritized_nms(all_detections, IOU_THRESHOLD)

            # ประกาศตัวแปรแสดงผลไว้ก่อน เพื่อให้แสดงค่า 0 ได้กรณีไม่เจอเป้า
            err_x, err_y = 0, 0
            speed_x, speed_y = 0, 0

            if final_detections:
                # --- ส่วน Logic การคำนวณ PID ---
                best_detection = final_detections[0]
                top_left_proc, bottom_right_proc, confidence, template_id, color = best_detection # <-- รับ color และ template_id
                
                target_center_x_proc = (top_left_proc[0] + bottom_right_proc[0]) / 2
                target_center_y_proc = (top_left_proc[1] + bottom_right_proc[1]) / 2
                x = target_center_x_proc / processing_scale
                y = (target_center_y_proc / processing_scale) + Y_AXIS_ADJUSTMENT

                err_x, err_y = center_x - x, center_y - y
                accumulate_err_x += err_x
                accumulate_err_y += err_y

                if count >= 1:
                    dt = prev_time - current_time
                    if dt == 0: dt = -1e-6

                    speed_x = (p * err_x) + d * ((prev_err_x - err_x) / dt) + i * accumulate_err_x
                    speed_y = (p * err_y) + d * ((prev_err_y - err_y) / dt) + i * accumulate_err_y
                    ep_gimbal.drive_speed(pitch_speed=-speed_y, yaw_speed=speed_x)
                
                # --- ส่วนการเก็บข้อมูล (เหมือนเดิม) ---
                time_elapsed = current_time - start_time
                data_for_saving.append([
                    time_elapsed, math.sqrt(err_x**2 + err_y**2), err_x, err_y,
                    accumulate_err_x, accumulate_err_y, math.sqrt(accumulate_err_x**2 + accumulate_err_y**2),
                    speed_x, speed_y, list_of_data[0], list_of_data[1]
                ])

                count += 1
                prev_time = current_time
                prev_err_x, prev_err_y = err_x, err_y
                time.sleep(0.001)

                # [จุดที่แก้ไข 3] นำโค้ดการวาด Bounding Box แบบละเอียดกลับมา
                tl_orig = (int(top_left_proc[0] / processing_scale), int(top_left_proc[1] / processing_scale))
                br_orig = (int(bottom_right_proc[0] / processing_scale), int(bottom_right_proc[1] / processing_scale))
                tl_adjusted = (tl_orig[0], tl_orig[1] + Y_AXIS_ADJUSTMENT)
                br_adjusted = (br_orig[0], br_orig[1] + Y_AXIS_ADJUSTMENT)
                cv2.rectangle(frame, tl_adjusted, br_adjusted, color, 3) # ใช้ `color` ที่ได้มา
                label = f"T{template_id+1} Conf:{confidence:.2f}" # ใช้ `template_id` และ `confidence`
                cv2.putText(frame, label, (tl_adjusted[0], tl_adjusted[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 255), -1)

            else:
                ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=0)

            # [จุดที่แก้ไข 4] นำโค้ดการแสดงผลบน Display กลับมาทั้งหมด
            delta_display_time = current_time - last_display_time
            display_fps = 1 / delta_display_time if delta_display_time > 0 else 999.0
            last_display_time = current_time
            
            cv2.putText(frame, f"FPS: {display_fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"e_x: {err_x:.2f}, e_y: {err_y:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Sp_x: {speed_x:.2f}, Sp_y: {speed_y:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.circle(frame, (int(center_x), int(center_y)), 5, (255, 0, 255), -1)

            cv2.imshow("Robomaster Pink Cup Detection with PID", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            
    finally:
        print("Stopping robot...")
        ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
        cv2.destroyAllWindows()
        ep_camera.stop_video_stream()
        ep_robot.close()

    # --- ส่วนการบันทึกข้อมูล (เหมือนเดิม) ---
    if not data_for_saving: print("No data collected."); return
    print(f"Saving {len(data_for_saving)} data points to CSV...")
    columns = [
        'time', 'err_total', 'err_x', 'err_y',
        'accumulate_err_x', 'accumulate_err_y', 'accumulate_err_total',
        'controller_output_x', 'controller_output_y',
        'gimbal_pitch_angle', 'gimbal_yaw_angle'
    ]
    df = pd.DataFrame(data_for_saving, columns=columns)
    df.to_csv('pid_data_from_robot_3.csv', index=False)
    print("Data saved successfully to 'pid_data_from_robot.csv'")

if __name__ == '__main__':
    main()