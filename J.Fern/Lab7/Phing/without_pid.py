import cv2
import numpy as np
import time
from robomaster import robot
from robomaster import camera
from robomaster import blaster

# --- คลาสสำหรับ Kalman Filter แบบ 1 มิติ ---
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

# --- ส่วนของฟังก์ชัน (ไม่มีการเปลี่ยนแปลง) ---
def create_pink_mask(img_rgb):
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # ใช้ค่าเดิมตามโค้ดล่าสุดของคุณ
    lower_pink = np.array([120, 10, 100])
    upper_pink = np.array([170, 100, 200])
    mask = cv2.inRange(hsv, lower_pink, upper_pink)
    mask = cv2.medianBlur(mask, 5)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
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
    """ฟังก์ชันหลักสำหรับเชื่อมต่อหุ่นยนต์และเริ่มการตรวจจับ (ไม่มี PID)"""
    
    PROCESSING_WIDTH = 640.0
    
    TEMPLATE_FILES = [
        "image/template/use/long_template_new1_pic3_x_327_y_344_w_157_h_345.jpg",
        "image/template/use/template_new2_pic2_x_580_y_291_w_115_h_235.jpg",
        "image/template/use/template_new1_pic3_x_607_y_286_w_70_h_142.jpg"
    ]
    MATCH_THRESHOLD = 0.55
    IOU_THRESHOLD = 0.1
    Y_AXIS_ADJUSTMENT = 25
    
    # --- ค่าคงที่สำหรับคำนวณระยะทาง ---
    K_X = 611.922
    K_Y = 405.797
    REAL_WIDTH_CM = 21.2
    REAL_HEIGHT_CM = 13.9

    # --- ตั้งค่า Kalman Filter ---
    kf_distance = SimpleKalmanFilter(process_noise=0.01, measurement_noise=15.0)

    # --- เชื่อมต่อหุ่นยนต์ ---
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")
    ep_camera = ep_robot.camera
    ep_gimbal = ep_robot.gimbal
    
    ep_camera.start_video_stream(display=False, resolution=camera.STREAM_720P)
    # สั่งให้ Gimbal มองไปข้างหน้าตอนเริ่มต้น
    ep_gimbal.recenter(pitch_speed=200, yaw_speed=200).wait_for_completed()
    time.sleep(1)

    # --- ตั้งค่าขนาดภาพและจุดศูนย์กลาง ---
    frame_for_scale = ep_camera.read_cv2_image(timeout=5)
    if frame_for_scale is None:
        print("Error: Could not get frame from camera.")
        ep_robot.close()
        return
        
    h_orig, w_orig, _ = frame_for_scale.shape
    processing_scale = PROCESSING_WIDTH / w_orig
    h_proc = int(h_orig * processing_scale)
    center_x_orig, center_y_orig = w_orig / 2, h_orig / 2 

    # --- โหลด Templates ---
    templates_masked = []
    templates_original = []
    try:
        for f in TEMPLATE_FILES:
            tmpl = cv2.imread(f)
            if tmpl is None: raise FileNotFoundError(f"Template file not found: {f}")
            templates_original.append(tmpl)
    except FileNotFoundError as e:
        print(f"Error: {e}"); ep_robot.close(); return

    for tmpl in templates_original:
        w_tmpl = int(tmpl.shape[1] * processing_scale)
        h_tmpl = int(tmpl.shape[0] * processing_scale)
        tmpl_resized = cv2.resize(tmpl, (w_tmpl, h_tmpl), interpolation=cv2.INTER_AREA)
        tmpl_rgb = cv2.cvtColor(tmpl_resized, cv2.COLOR_BGR2RGB)
        tmpl_pink_mask = create_pink_mask(tmpl_rgb)
        tmpl_gray = cv2.cvtColor(tmpl_rgb, cv2.COLOR_RGB2GRAY)
        tmpl_masked = cv2.bitwise_and(tmpl_gray, tmpl_gray, mask=tmpl_pink_mask)
        templates_masked.append(tmpl_masked)

    # --- khởi tạoตัวแปรสำหรับแสดงผล ---
    err_x, err_y = 0, 0
    final_distance = 0.0 
    kalman_distance = 0.0

    try:
        # --- ลูปการทำงานหลัก ---
        while True:
            frame = ep_camera.read_cv2_image()
            if frame is None: continue
            
            target_found = False

            # --- ประมวลผลภาพ ---
            frame_proc = cv2.resize(frame, (int(PROCESSING_WIDTH), h_proc), interpolation=cv2.INTER_AREA)
            frame_rgb = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2RGB)
            main_pink_mask = create_pink_mask(frame_rgb)
            main_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
            main_masked = cv2.bitwise_and(main_gray, main_gray, mask=main_pink_mask)
            
            # --- ตรวจจับวัตถุ ---
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
                        
                        real_top_left = (roi_x1 + x_real, roi_y1 + y_real)
                        real_bottom_right = (roi_x1 + x_real + w_real, roi_y1 + y_real + h_real)
                        cv2.rectangle(frame, real_top_left, real_bottom_right, (0, 255, 255), 2)
                        
                        target_center_x = (real_top_left[0] + real_bottom_right[0]) / 2
                        target_center_y = (real_top_left[1] + real_bottom_right[1]) / 2
                        
                        # คำนวณค่า Error เพื่อแสดงผล แต่จะไม่นำไปใช้ควบคุม
                        err_x = center_x_orig - target_center_x
                        err_y = center_y_orig - target_center_y
                        target_found = True
                        cv2.circle(frame, (int(target_center_x), int(target_center_y)), 5, (0, 0, 255), -1)
                        
                        # คำนวณระยะทาง
                        distance_z_x, distance_z_y = 0, 0
                        if w_real > 0: distance_z_x = (K_Y * REAL_HEIGHT_CM) / w_real
                        if h_real > 0: distance_z_y = (K_X * REAL_WIDTH_CM) / h_real
                        
                        if distance_z_x > 0 and distance_z_y > 0: final_distance = (distance_z_x + distance_z_y) / 2
                        elif distance_z_x > 0: final_distance = distance_z_x
                        else: final_distance = distance_z_y
                        
                        # อัปเดต Kalman Filter
                        if final_distance > 0:
                            kf_distance.predict()
                            kf_distance.update(final_distance)
                            kalman_distance = kf_distance.get_state()
            
            # [ปรับปรุง] ถ้าไม่เจอเป้าหมาย ให้รีเซ็ตค่าแสดงผล และยังคง Predict ค่า Kalman
            if not target_found:
                 kf_distance.predict()
                 kalman_distance = kf_distance.get_state()
                 err_x, err_y = 0, 0
                 final_distance = 0.0

            # --- [ส่วนควบคุม PID ถูกลบออกทั้งหมด] ---
            # ไม่มีคำสั่ง ep_gimbal.drive_speed() ในลูปอีกต่อไป

            # --- แสดงผลข้อมูล ---
            cv2.putText(frame, f"e_x: {err_x:.2f}, e_y: {err_y:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Raw Dist: {final_distance:.2f} cm", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Kalman Dist: {kalman_distance:.2f} cm", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.circle(frame, (int(center_x_orig), int(center_y_orig)), 5, (255, 0, 255), -1)
            cv2.imshow("Robomaster Pink Cup Detection (No PID)", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        print("Stopping detection and releasing resources.")
        # หยุดการทำงานของ Gimbal (เผื่อไว้)
        ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
        cv2.destroyAllWindows()
        ep_camera.stop_video_stream()
        ep_robot.close()

if __name__ == '__main__':
    main()