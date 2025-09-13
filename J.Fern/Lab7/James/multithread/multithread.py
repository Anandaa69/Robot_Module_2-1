import cv2
import numpy as np
import time
import threading # <-- เพิ่มเข้ามา
from robomaster import robot
from robomaster import camera

# --- คลาสและฟังก์ชันต่างๆ (ไม่มีการเปลี่ยนแปลง) ---
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
    def reset(self, value):
        self.x = value
        self.P = 1.0

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


# --- [เพิ่มใหม่] ส่วนของ Multithreading ---

# ตัวแปรที่แชร์กันระหว่าง Threads
latest_frame = None
processed_output = {
    "annotated_frame": None,
    "speed_x": 0.0,
    "speed_y": 0.0,
}
# Lock สำหรับป้องกัน Race Condition
frame_lock = threading.Lock()
output_lock = threading.Lock()
# Event สำหรับสั่งให้ Threads หยุดทำงาน
stop_event = threading.Event()

# ---------------------------------------------------
# Thread 1: Frame Grabber
# ---------------------------------------------------
def capture_thread_func(ep_camera):
    """
    Thread นี้ทำหน้าที่รับภาพจากกล้องอย่างเดียว แล้วเก็บไว้ใน latest_frame
    เพื่อให้ Thread อื่นๆ เอาไปใช้ได้ทันที ลดการรอ I/O
    """
    global latest_frame
    print("Capture thread started.")
    while not stop_event.is_set():
        frame = ep_camera.read_cv2_image(timeout=0.5)
        if frame is not None:
            with frame_lock:
                latest_frame = frame.copy()
        else:
            time.sleep(0.01) # รอเล็กน้อยถ้าไม่มีเฟรม
    print("Capture thread stopped.")

# ---------------------------------------------------
# Thread 2: Image Processor & PID Calculator
# ---------------------------------------------------
def processing_thread_func(templates_masked, params):
    """
    Thread นี้จะทำงานหนักที่สุด คือการประมวลผลภาพทั้งหมด
    และคำนวณค่า PID เพื่อหาความเร็วที่เหมาะสม
    """
    global latest_frame, processed_output
    print("Processing thread started.")

    # --- ดึงค่า Parameters ออกมา ---
    h_proc, center_x_orig, center_y_orig, processing_scale = params["dims"]
    P_GAIN, I_GAIN, D_GAIN = params["pid"]
    K_VALUES = params["k_values"]
    REAL_WIDTH_CM, REAL_HEIGHT_CM = params["real_dims"]
    MATCH_THRESHOLD, IOU_THRESHOLD, Y_AXIS_ADJUSTMENT = params["detection"]

    # --- State variables สำหรับ Thread นี้โดยเฉพาะ ---
    prev_err_x, prev_err_y = 0, 0
    accumulate_err_x, accumulate_err_y = 0, 0
    prev_time = time.time()
    is_currently_tracking = False

    kf_distance = SimpleKalmanFilter(process_noise=0.01, measurement_noise=15.0)
    # (สามารถเพิ่ม Kalman Filter ตัวอื่นๆ ได้ที่นี่ถ้าต้องการ)

    while not stop_event.is_set():
        current_frame = None
        with frame_lock:
            if latest_frame is not None:
                current_frame = latest_frame.copy()
        
        if current_frame is None:
            time.sleep(0.01) # รอถ้ายังไม่มีเฟรมแรก
            continue
        
        # --- เริ่มส่วนการประมวลผล (นำมาจากโค้ดเดิม) ---
        target_found = False
        err_x, err_y = 0, 0
        speed_x, speed_y = 0, 0
        final_distance = 0.0
        
        frame_proc = cv2.resize(current_frame, (int(params["PROCESSING_WIDTH"]), h_proc), interpolation=cv2.INTER_AREA)
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
            
            tl_orig_unadjusted = (int(top_left_proc[0] / processing_scale), int(top_left_proc[1] / processing_scale))
            br_orig_unadjusted = (int(bottom_right_proc[0] / processing_scale), int(bottom_right_proc[1] / processing_scale))
            tl_adjusted = (tl_orig_unadjusted[0], tl_orig_unadjusted[1] + Y_AXIS_ADJUSTMENT)
            br_adjusted = (br_orig_unadjusted[0], br_orig_unadjusted[1] + Y_AXIS_ADJUSTMENT)
            cv2.rectangle(current_frame, tl_adjusted, br_adjusted, color, 3)
            label = f"T{template_id+1} Conf:{confidence:.2f}"
            cv2.putText(current_frame, label, (tl_adjusted[0], tl_adjusted[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            roi_x1, roi_y1 = max(0, tl_adjusted[0]), max(0, tl_adjusted[1])
            roi_x2, roi_y2 = min(current_frame.shape[1], br_adjusted[0]), min(current_frame.shape[0], br_adjusted[1])
            
            if roi_x2 > roi_x1 and roi_y2 > roi_y1:
                roi = current_frame[roi_y1:roi_y2, roi_x1:roi_x2]
                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                roi_pink_mask = create_pink_mask(roi_rgb)
                contours, _ = cv2.findContours(roi_pink_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if contours:
                    main_contour = max(contours, key=cv2.contourArea)
                    x_real, y_real, w_real, h_real = cv2.boundingRect(main_contour)
                    
                    real_top_left = (roi_x1 + x_real, roi_y1 + y_real)
                    real_bottom_right = (roi_x1 + x_real + w_real, roi_y1 + y_real + h_real)
                    cv2.rectangle(current_frame, real_top_left, real_bottom_right, (0, 255, 255), 2)
                    target_center_x = (real_top_left[0] + real_bottom_right[0]) / 2
                    target_center_y = (real_top_left[1] + real_bottom_right[1]) / 2
                    err_x = center_x_orig - target_center_x
                    err_y = center_y_orig - target_center_y
                    target_found = True
                    cv2.circle(current_frame, (int(target_center_x), int(target_center_y)), 5, (0, 0, 255), -1)
                    
                    current_kx, current_ky = K_VALUES[template_id]
                    distance_z_x = (current_ky * REAL_HEIGHT_CM) / w_real if w_real > 0 else 0
                    distance_z_y = (current_kx * REAL_WIDTH_CM) / h_real if h_real > 0 else 0
                    
                    if distance_z_x > 0 and distance_z_y > 0: final_distance = (distance_z_x + distance_z_y) / 2
                    elif distance_z_x > 0: final_distance = distance_z_x
                    else: final_distance = distance_z_y
                    
                    if final_distance > 0:
                        kf_distance.predict()
                        kf_distance.update(final_distance)

        kalman_distance = kf_distance.get_state() if target_found else 0

        # --- ส่วนคำนวณ PID และ Speed ---
        current_time = time.time()
        if target_found:
            delta_time = current_time - prev_time
            if delta_time == 0: delta_time = 0.001
            
            if not is_currently_tracking:
                D_term_x, D_term_y = 0, 0
                accumulate_err_x, accumulate_err_y = 0, 0
                is_currently_tracking = True
            else:
                D_term_x = D_GAIN * ((err_x - prev_err_x) / delta_time)
                D_term_y = D_GAIN * ((err_y - prev_err_y) / delta_time)
                accumulate_err_x += err_x * delta_time
                accumulate_err_y += err_y * delta_time

            speed_x = (P_GAIN * err_x) + D_term_x + (I_GAIN * accumulate_err_x)
            speed_y = (P_GAIN * err_y) + D_term_y + (I_GAIN * accumulate_err_y)
            
            prev_err_x, prev_err_y = err_x, err_y
            prev_time = current_time
        else:
            speed_x, speed_y = 0, 0
            is_currently_tracking = False
            accumulate_err_x, accumulate_err_y, prev_err_x, prev_err_y = 0, 0, 0, 0
        
        # --- ส่วนแสดงผลบน Frame (สำหรับ UI) ---
        cv2.putText(current_frame, f"e_x: {err_x:.2f}, e_y: {err_y:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(current_frame, f"Sp_x: {speed_x:.2f}, Sp_y: {speed_y:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(current_frame, f"Raw Dist: {final_distance:.2f} cm", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(current_frame, f"Kalman Dist: {kalman_distance:.2f} cm", (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.circle(current_frame, (int(center_x_orig), int(center_y_orig)), 5, (255, 0, 255), -1)

        # --- อัปเดตผลลัพธ์สุดท้ายให้ Main Thread ---
        with output_lock:
            processed_output["annotated_frame"] = current_frame
            processed_output["speed_x"] = speed_x
            processed_output["speed_y"] = speed_y
            
    print("Processing thread stopped.")


# ---------------------------------------------------
# ส่วนที่ 3: Main Thread (Controller & UI)
# ---------------------------------------------------
def main():
    global processed_output

    # --- ค่าคงที่และ Parameters ---
    PROCESSING_WIDTH = 640.0
    TEMPLATE_FILES = [
        "image/template/use/template_night_pic1_x_557_y_278_w_120_h_293.jpg",
        "image/template/use/template_new_pic2_x_552_y_246_w_90_h_212.jpg",
        # "image/template/use/template_new_pic3_x_577_y_258_w_45_h_117.jpg"
    ]
    
    # --- รวม Parameters ทั้งหมดไว้ใน Dictionary เพื่อส่งให้ Thread ---
    params = {
        "PROCESSING_WIDTH": PROCESSING_WIDTH,
        "pid": (-0.607, 0, -0.00135),
        "k_values": [
            (600.01, 371.126),  # T1
            (591.768, 395.899), # T2
            (614.324, 409.928)  # T3
        ],
        "real_dims": (23.9, 13.9), # REAL_WIDTH_CM, REAL_HEIGHT_CM
        "detection": (0.55, 0.1, 25) # MATCH_THRESHOLD, IOU_THRESHOLD, Y_AXIS_ADJUSTMENT
    }

    # --- การเชื่อมต่อหุ่นยนต์ (ทำใน Main Thread) ---
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")
    ep_camera = ep_robot.camera
    ep_gimbal = ep_robot.gimbal
    
    ep_camera.start_video_stream(display=False, resolution=camera.STREAM_720P)
    ep_gimbal.recenter(pitch_speed=200, yaw_speed=200).wait_for_completed()
    time.sleep(1)

    # --- เตรียมข้อมูลสำหรับ Threads ---
    frame_for_scale = ep_camera.read_cv2_image(timeout=5)
    if frame_for_scale is None:
        print("Error: Could not get frame from camera.")
        ep_robot.close()
        return
        
    h_orig, w_orig, _ = frame_for_scale.shape
    processing_scale = PROCESSING_WIDTH / w_orig
    h_proc = int(h_orig * processing_scale)
    params["dims"] = (h_proc, w_orig / 2, h_orig / 2, processing_scale)

    templates_masked = []
    try:
        for f in TEMPLATE_FILES:
            tmpl = cv2.imread(f)
            if tmpl is None: raise FileNotFoundError(f"Template file not found: {f}")
            w_tmpl = int(tmpl.shape[1] * processing_scale)
            h_tmpl = int(tmpl.shape[0] * processing_scale)
            tmpl_resized = cv2.resize(tmpl, (w_tmpl, h_tmpl), interpolation=cv2.INTER_AREA)
            tmpl_rgb = cv2.cvtColor(tmpl_resized, cv2.COLOR_BGR2RGB)
            tmpl_pink_mask = create_pink_mask(tmpl_rgb)
            tmpl_gray = cv2.cvtColor(tmpl_rgb, cv2.COLOR_RGB2GRAY)
            tmpl_masked = cv2.bitwise_and(tmpl_gray, tmpl_gray, mask=tmpl_pink_mask)
            templates_masked.append(tmpl_masked)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        ep_robot.close()
        return

    # --- เริ่มการทำงานของ Threads ---
    capture_thread = threading.Thread(target=capture_thread_func, args=(ep_camera,))
    processing_thread = threading.Thread(target=processing_thread_func, args=(templates_masked, params))
    
    capture_thread.start()
    processing_thread.start()

    try:
        # --- Loop หลักของ Main Thread ---
        # ทำหน้าที่แค่ควบคุมหุ่นยนต์และแสดงผลเท่านั้น
        while True:
            local_speed_x, local_speed_y = 0.0, 0.0
            annotated_frame_copy = None

            with output_lock:
                local_speed_x = processed_output["speed_x"]
                local_speed_y = processed_output["speed_y"]
                if processed_output["annotated_frame"] is not None:
                    annotated_frame_copy = processed_output["annotated_frame"].copy()

            # ส่งคำสั่งควบคุม Gimbal
            if local_speed_x != 0 or local_speed_y != 0:
                 ep_gimbal.drive_speed(pitch_speed=-local_speed_y, yaw_speed=local_speed_x)
            else:
                 # ถ้าไม่มีเป้าหมาย ให้หยุด gimbal เพื่อความเสถียร
                 ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=0)

            # แสดงผลภาพ
            if annotated_frame_copy is not None:
                cv2.imshow("Robomaster Pink Cup Detection (Multithreaded)", annotated_frame_copy)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # เช็คว่า Thread ลูกยังทำงานอยู่หรือไม่
            if not processing_thread.is_alive() or not capture_thread.is_alive():
                print("A thread has unexpectedly stopped. Exiting.")
                break

    finally:
        print("Stopping threads and releasing resources...")
        # --- ขั้นตอนการปิดโปรแกรมอย่างถูกต้อง ---
        stop_event.set() # ส่งสัญญาณให้ทุก Thread หยุดการทำงาน
        
        capture_thread.join() # รอให้ capture_thread ทำงานจนจบ
        processing_thread.join() # รอให้ processing_thread ทำงานจนจบ
        
        ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
        cv2.destroyAllWindows()
        ep_camera.stop_video_stream()
        ep_robot.close()
        print("Program terminated gracefully.")

if __name__ == '__main__':
    main()