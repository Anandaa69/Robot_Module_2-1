import cv2
import numpy as np
import time
from robomaster import robot
from robomaster import camera
from robomaster import blaster
# from collections import deque # เราจะใช้ Kalman Filter แทน

# --- [เพิ่มใหม่] คลาสสำหรับ Kalman Filter แบบ 1 มิติ ---
class SimpleKalmanFilter:
    def __init__(self, process_noise, measurement_noise, initial_value=0.0):
        self.Q = process_noise 
        self.R = measurement_noise 
        self.P = 1.0
        self.x = initial_value

    def update(self, measurement):
        K = self.P / (self.P + self.R)
        self.x = self.x + K * (measurement - self.x)
        self.P = (1 - K) * self.P

    def predict(self):
        self.P = self.P + self.Q
        
    def get_state(self):
        return self.x
        
    # --- [เพิ่มใหม่] ฟังก์ชันสำหรับรีเซ็ตค่าฟิลเตอร์ ---
    def reset(self, value):
        self.x = value
        self.P = 1.0


# --- ส่วนของฟังก์ชัน (ไม่มีการเปลี่ยนแปลง) ---
def create_pink_mask(img_rgb):
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # 🎯 ปรับ HSV Range ให้แคบลง (กันเงาสะท้อน/แสงฟุ้ง)
    lower_pink = np.array([140, 80, 120])   # Hue แคบลง + เพิ่ม Saturation + Brightness ขั้นต่ำ
    upper_pink = np.array([170, 255, 255])  # Hue ไม่กว้างเกินไป

    mask = cv2.inRange(hsv, lower_pink, upper_pink)

    # 🔧 ลบ noise เล็กๆ
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # 🔧 เติมช่องว่างในวัตถุ
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 📌 เน้นให้ขอบคมขึ้น
    mask = cv2.medianBlur(mask, 3)
    edges = cv2.Canny(mask, 50, 150)       # ตรวจขอบ
    mask = cv2.bitwise_or(mask, edges)     # รวม edges กลับเข้า mask

    return mask


def match_template_masked(img_masked, tmpl_masked, threshold=0.7):
    if tmpl_masked.shape[0] > img_masked.shape[0] or tmpl_masked.shape[1] > img_masked.shape[1]:
        return []
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
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - intersection_area
    return intersection_area / union_area if union_area > 0 else 0.0

def advanced_prioritized_nms(boxes_with_scores, iou_threshold=0.3):
    if not boxes_with_scores:
        return []
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
            is_center_inside = (current_box[0][0] < center_x < current_box[1][0] and
                                current_box[0][1] < center_y < current_box[1][1])
            if iou > iou_threshold or is_center_inside:
                continue
            remaining_boxes.append(box_to_check)
        detections = remaining_boxes
    return final_boxes


# ---------------------------------------------------
# ส่วนที่ 2: การทำงานหลัก (Main Program)
# ---------------------------------------------------
def main():
    """ฟังก์ชันหลักสำหรับเชื่อมต่อหุ่นยนต์และเริ่มการตรวจจับพร้อม PID"""
    
    PROCESSING_WIDTH = 640.0
    
    TEMPLATE_FILES = [
        "image/template/use/template_night_pic1_x_557_y_278_w_120_h_293.jpg",
        "image/template/use/template_new_pic2_x_552_y_246_w_90_h_212.jpg",
        "image/template/use/template_new_pic3_x_577_y_258_w_45_h_117.jpg"
    ]
    MATCH_THRESHOLD = 0.55
    IOU_THRESHOLD = 0.1
    Y_AXIS_ADJUSTMENT = 25
    
    P_GAIN = -0.607
    I_GAIN = 0
    D_GAIN = -0.00135 

    # --- [ปรับแก้] ค่า K ทั้ง 3 ชุดที่คุณ Calibrate มา ---
    # ชุดที่ 1 (สำหรับ Template 1 - ระยะใกล้)
    K_X_T1 = 600.01
    K_Y_T1 = 371.126
    # ชุดที่ 2 (สำหรับ Template 2 - ระยะกลาง)
    K_X_T2 = 591.768 
    K_Y_T2 = 395.899
    # ชุดที่ 3 (สำหรับ Template 3 - ระยะไกล)
    K_X_T3 = 614.324
    K_Y_T3 = 409.928

    REAL_WIDTH_CM = 23.9
    REAL_HEIGHT_CM = 13.9

    kf_distance = SimpleKalmanFilter(process_noise=0.01, measurement_noise=15.0)
    kf_x = SimpleKalmanFilter(process_noise=0.1, measurement_noise=4.0)
    kf_y = SimpleKalmanFilter(process_noise=0.1, measurement_noise=4.0)
    kf_w = SimpleKalmanFilter(process_noise=0.01, measurement_noise=10.0)
    kf_h = SimpleKalmanFilter(process_noise=0.01, measurement_noise=10.0)

    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")
    ep_camera = ep_robot.camera
    ep_gimbal = ep_robot.gimbal
    
    ep_camera.start_video_stream(display=False, resolution=camera.STREAM_720P)
    ep_gimbal.recenter(pitch_speed=200, yaw_speed=200).wait_for_completed()
    time.sleep(1)

    frame_for_scale = ep_camera.read_cv2_image(timeout=5)
    if frame_for_scale is None:
        print("Error: Could not get frame from camera.")
        ep_robot.close()
        return
        
    h_orig, w_orig, _ = frame_for_scale.shape
    processing_scale = PROCESSING_WIDTH / w_orig
    h_proc = int(h_orig * processing_scale)
    center_x_orig, center_y_orig = w_orig / 2, h_orig / 2 

    templates_masked = []
    templates_original = []
    try:
        for f in TEMPLATE_FILES:
            tmpl = cv2.imread(f)
            if tmpl is None: raise FileNotFoundError(f"Template file not found: {f}")
            templates_original.append(tmpl)
    except FileNotFoundError as e:
        print(f"Error: {e}"); return
    for tmpl in templates_original:
        w_tmpl = int(tmpl.shape[1] * processing_scale)
        h_tmpl = int(tmpl.shape[0] * processing_scale)
        tmpl_resized = cv2.resize(tmpl, (w_tmpl, h_tmpl), interpolation=cv2.INTER_AREA)
        tmpl_rgb = cv2.cvtColor(tmpl_resized, cv2.COLOR_BGR2RGB)
        tmpl_pink_mask = create_pink_mask(tmpl_rgb)
        tmpl_gray = cv2.cvtColor(tmpl_rgb, cv2.COLOR_RGB2GRAY)
        tmpl_masked = cv2.bitwise_and(tmpl_gray, tmpl_gray, mask=tmpl_pink_mask)
        templates_masked.append(tmpl_masked)

    prev_err_x, prev_err_y = 0, 0
    accumulate_err_x, accumulate_err_y = 0, 0
    prev_time = time.time()
    err_x, err_y = 0, 0
    speed_x, speed_y = 0, 0
    distance_z_x, distance_z_y = 0, 0
    final_distance = 0.0 
    kalman_distance = 0.0
    
    is_currently_tracking = False

    try:
        while True:
            frame = ep_camera.read_cv2_image()
            if frame is None: continue
            current_time = time.time()
            target_found = False

            frame_proc = cv2.resize(frame, (int(PROCESSING_WIDTH), h_proc), interpolation=cv2.INTER_AREA)
            frame_rgb = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2RGB)
            main_pink_mask = create_pink_mask(frame_rgb)
            main_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
            main_masked = cv2.bitwise_and(main_gray, main_gray, mask=main_pink_mask)
            all_detections = []
            colors = [(0, 255, 0), (255, 165, 0), (0, 0, 255)]
            for i, tmpl_masked in enumerate(templates_masked):
                boxes = match_template_masked(main_masked, tmpl_masked, threshold=MATCH_THRESHOLD)
                for top_left, bottom_right, confidence in boxes:
                    all_detections.append((top_left, bottom_right, confidence, i, colors[i]))
            final_detections = advanced_prioritized_nms(all_detections, IOU_THRESHOLD)

            if final_detections:
                best_detection = final_detections[0] 
                top_left_proc, bottom_right_proc, confidence, template_id, color = best_detection
                
                # (ส่วนแสดงผล Template ยังคงเดิม)
                tl_orig_unadjusted = (int(top_left_proc[0] / processing_scale), int(top_left_proc[1] / processing_scale))
                br_orig_unadjusted = (int(bottom_right_proc[0] / processing_scale), int(bottom_right_proc[1] / processing_scale))
                tl_adjusted = (tl_orig_unadjusted[0], tl_orig_unadjusted[1] + Y_AXIS_ADJUSTMENT)
                br_adjusted = (br_orig_unadjusted[0], br_orig_unadjusted[1] + Y_AXIS_ADJUSTMENT)
                cv2.rectangle(frame, tl_adjusted, br_adjusted, color, 3)
                label = f"T{template_id+1} Conf:{confidence:.2f}"
                cv2.putText(frame, label, (tl_adjusted[0], tl_adjusted[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                roi_x1, roi_y1 = max(0, tl_adjusted[0]), max(0, tl_adjusted[1])
                roi_x2, roi_y2 = min(frame.shape[1], br_adjusted[0]), min(frame.shape[0], br_adjusted[1])
                
                if roi_x2 > roi_x1 and roi_y2 > roi_y1:
                    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
                    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                    roi_pink_mask = create_pink_mask(roi_rgb)
                    contours, _ = cv2.findContours(roi_pink_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    if contours:
                        main_contour = max(contours, key=cv2.contourArea)
                        x_real, y_real, w_real, h_real = cv2.boundingRect(main_contour)
                        
                        # (ส่วนคำนวณตำแหน่ง... เหมือนเดิม)
                        real_top_left = (roi_x1 + x_real, roi_y1 + y_real)
                        real_bottom_right = (roi_x1 + x_real + w_real, roi_y1 + y_real + h_real)
                        cv2.rectangle(frame, real_top_left, real_bottom_right, (0, 255, 255), 2)
                        target_center_x = (real_top_left[0] + real_bottom_right[0]) / 2
                        target_center_y = (real_top_left[1] + real_bottom_right[1]) / 2
                        err_x = center_x_orig - target_center_x
                        err_y = center_y_orig - target_center_y
                        target_found = True
                        cv2.circle(frame, (int(target_center_x), int(target_center_y)), 5, (0, 0, 255), -1)
                        
                        # --- [ปรับแก้] เลือกใช้ค่า K ตาม template_id ที่ตรวจจับได้ ---
                        current_kx = 0
                        current_ky = 0
                        if template_id == 0:  # ถ้าเจอ Template 1
                            current_kx = K_X_T1
                            current_ky = K_Y_T1
                        elif template_id == 1: # ถ้าเจอ Template 2
                            current_kx = K_X_T2
                            current_ky = K_Y_T2
                        else:  # ถ้าเจอ Template 3 (template_id == 2)
                            current_kx = K_X_T3
                            current_ky = K_Y_T3

                        # --- [ปรับแก้] คำนวณระยะทางโดยใช้ค่า K ที่เลือกมา ---
                        if w_real > 0: distance_z_x = (current_ky * REAL_HEIGHT_CM) / w_real
                        if h_real > 0: distance_z_y = (current_kx * REAL_WIDTH_CM) / h_real
                        
                        # (ส่วนที่เหลือ... เหมือนเดิม)
                        if distance_z_x > 0 and distance_z_y > 0: final_distance = (distance_z_x + distance_z_y) / 2
                        elif distance_z_x > 0: final_distance = distance_z_x
                        else: final_distance = distance_z_y
                        
                        if final_distance > 0:
                            kf_distance.predict()
                            kf_distance.update(final_distance)
                            kalman_distance = kf_distance.get_state()
            
            if not target_found:
                 kf_distance.predict()
                 kalman_distance = kf_distance.get_state()

            # --- [ปรับแก้] PID Control Logic ---
            if target_found:
                delta_time = current_time - prev_time
                if delta_time == 0: delta_time = 0.001
                
                # ถ้าเพิ่งเจอเป้าหมาย (จากเดิมที่ไม่เจอ)
                if not is_currently_tracking:
                    # ให้คำนวณแค่ P-term เพื่อการเคลื่อนที่เริ่มต้นที่นุ่มนวล
                    D_term_x, D_term_y = 0, 0
                    accumulate_err_x, accumulate_err_y = 0, 0
                    is_currently_tracking = True # อัปเดตสถานะว่ากำลังติดตามอยู่
                # ถ้าเจอเป้าหมายต่อเนื่อง
                else:
                    D_term_x = D_GAIN * ((err_x - prev_err_x) / delta_time)
                    D_term_y = D_GAIN * ((err_y - prev_err_y) / delta_time)
                    accumulate_err_x += err_x * delta_time
                    accumulate_err_y += err_y * delta_time

                speed_x = (P_GAIN * err_x) + D_term_x + (I_GAIN * accumulate_err_x)
                speed_y = (P_GAIN * err_y) + D_term_y + (I_GAIN * accumulate_err_y)
                
                ep_gimbal.drive_speed(pitch_speed=-speed_y, yaw_speed=speed_x) 
                
                prev_err_x, prev_err_y = err_x, err_y
                prev_time = current_time
            else:
                # รีเซ็ตค่าทั้งหมดเมื่อไม่เจอเป้าหมาย
                ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
                is_currently_tracking = False # อัปเดตสถานะว่าไม่ได้ติดตามแล้ว
                accumulate_err_x, accumulate_err_y, prev_err_x, prev_err_y, err_x, err_y = 0, 0, 0, 0, 0, 0
                distance_z_x, distance_z_y = 0, 0
                final_distance = 0.0

            # Display information
            cv2.putText(frame, f"e_x: {err_x:.2f}, e_y: {err_y:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Sp_x: {speed_x:.2f}, Sp_y: {speed_y:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, f"Raw Dist: {final_distance:.2f} cm", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Kalman Dist: {kalman_distance:.2f} cm", (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.circle(frame, (int(center_x_orig), int(center_y_orig)), 5, (255, 0, 255), -1)
            cv2.imshow("Robomaster Pink Cup Detection with PID", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'): break

    finally:
        print("Stopping detection and releasing resources.")
        ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
        cv2.destroyAllWindows()
        ep_camera.stop_video_stream()
        ep_robot.close()

if __name__ == '__main__':
    main()