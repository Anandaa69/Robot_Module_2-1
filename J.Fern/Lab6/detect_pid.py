import cv2
import numpy as np
import time
from robomaster import robot
from robomaster import camera
from robomaster import blaster

# --- ส่วนของฟังก์ชัน (ไม่มีการเปลี่ยนแปลง) ---
# ... (โค้ดของฟังก์ชัน create_pink_mask, match_template_masked, calculate_iou, advanced_prioritized_nms) ...

# ---------------------------------------------------
# ส่วนที่ 2: การทำงานหลัก (Main Program) - แก้ไขแล้ว
# ---------------------------------------------------

def main():
    """ฟังก์ชันหลักสำหรับเชื่อมต่อหุ่นยนต์และเริ่มการตรวจจับพร้อม PID"""
    
    PROCESSING_WIDTH = 640.0
    
    TEMPLATE_FILES = [
        "image/template/use/template_night_pic1_x_557_y_278_w_120_h_293.jpg",
        "image/template/use/template_night_pic2_x_609_y_290_w_57_h_138.jpg",
        "image/template/use/template_night_pic3_x_622_y_293_w_40_h_93.jpg"
    ]
    MATCH_THRESHOLD = 0.55
    IOU_THRESHOLD = 0.1
    Y_AXIS_ADJUSTMENT = 25
    
    P_GAIN = -0.607
    I_GAIN = 0
    D_GAIN = -0.00135 

    print("Loading and processing templates...")
    templates_original = []
    try:
        for f in TEMPLATE_FILES:
            tmpl = cv2.imread(f)
            if tmpl is None: raise FileNotFoundError(f"Template file not found: {f}")
            templates_original.append(tmpl)
    except FileNotFoundError as e:
        print(f"Error: {e}"); return

    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")
    ep_camera = ep_robot.camera
    ep_gimbal = ep_robot.gimbal
    ep_blaster = ep_robot.blaster

    ep_camera.start_video_stream(display=False, resolution=camera.STREAM_720P)
    ep_gimbal.recenter(pitch_speed=200, yaw_speed=200).wait_for_completed()
    time.sleep(1)

    print("Waiting for first frame to calculate scale...")
    frame_for_scale = ep_camera.read_cv2_image(timeout=5)
    if frame_for_scale is None:
        print("Error: Could not get frame from camera.")
        ep_robot.close()
        return
        
    h_orig, w_orig, _ = frame_for_scale.shape
    processing_scale = PROCESSING_WIDTH / w_orig
    h_proc = int(h_orig * processing_scale)
    
    print(f"Original frame: {w_orig}x{h_orig}, Processing at: {int(PROCESSING_WIDTH)}x{h_proc}")
    
    center_x_orig = w_orig / 2 
    center_y_orig = h_orig / 2 

    templates_masked = []
    for tmpl in templates_original:
        w_tmpl = int(tmpl.shape[1] * processing_scale)
        h_tmpl = int(tmpl.shape[0] * processing_scale)
        tmpl_resized = cv2.resize(tmpl, (w_tmpl, h_tmpl), interpolation=cv2.INTER_AREA)
        
        tmpl_rgb = cv2.cvtColor(tmpl_resized, cv2.COLOR_BGR2RGB)
        tmpl_pink_mask = create_pink_mask(tmpl_rgb)
        tmpl_gray = cv2.cvtColor(tmpl_rgb, cv2.COLOR_RGB2GRAY)
        tmpl_masked = cv2.bitwise_and(tmpl_gray, tmpl_gray, mask=tmpl_pink_mask)
        templates_masked.append(tmpl_masked)
        
    print("Robot connected. Starting real-time detection and PID control...")

    prev_err_x = 0
    prev_err_y = 0
    accumulate_err_x = 0
    accumulate_err_y = 0
    prev_time = time.time()
    count = 0

    # ประกาศตัวแปรที่ยังไม่ได้ประกาศก่อนเข้า loop
    err_x, err_y = 0, 0
    speed_x, speed_y = 0, 0
    target_found = False

    try:
        last_display_time = time.time()
        while True:
            frame = ep_camera.read_cv2_image()
            if frame is None: continue

            current_time = time.time()

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

                tl_orig_unadjusted = (int(top_left_proc[0] / processing_scale), int(top_left_proc[1] / processing_scale))
                br_orig_unadjusted = (int(bottom_right_proc[0] / processing_scale), int(bottom_right_proc[1] / processing_scale))
                
                tl_adjusted = (tl_orig_unadjusted[0], tl_orig_unadjusted[1] + Y_AXIS_ADJUSTMENT)
                br_adjusted = (br_orig_unadjusted[0], br_orig_unadjusted[1] + Y_AXIS_ADJUSTMENT)
                
                target_center_x = (tl_adjusted[0] + br_adjusted[0]) / 2
                target_center_y = (tl_adjusted[1] + br_adjusted[1]) / 2

                err_x = center_x_orig - target_center_x
                err_y = center_y_orig - target_center_y
                target_found = True

                cv2.rectangle(frame, tl_adjusted, br_adjusted, color, 3)
                label = f"T{template_id+1} Conf:{confidence:.2f}"
                cv2.putText(frame, label, (tl_adjusted[0], tl_adjusted[1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.circle(frame, (int(target_center_x), int(target_center_y)), 5, (0, 255, 255), -1)
            else:
                target_found = False

            if target_found:
                delta_time = current_time - prev_time
                if delta_time == 0:
                    delta_time = 0.001 

                accumulate_err_x += err_x * delta_time
                accumulate_err_y += err_y * delta_time

                speed_x = (P_GAIN * err_x) + (D_GAIN * ((prev_err_x - err_x) / delta_time)) + (I_GAIN * accumulate_err_x)
                speed_y = (P_GAIN * err_y) + (D_GAIN * ((prev_err_y - err_y) / delta_time)) + (I_GAIN * accumulate_err_y)
                
                ep_gimbal.drive_speed(pitch_speed=-speed_y, yaw_speed=speed_x) 

                prev_err_x = err_x
                prev_err_y = err_y
                prev_time = current_time
                count += 1
            else:
                ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
                accumulate_err_x = 0 
                accumulate_err_y = 0
                prev_err_x = 0
                prev_err_y = 0
                err_x, err_y = 0, 0 # รีเซ็ต error ที่แสดงผลด้วย

            # ==========================[ จุดที่แก้ไข ]===========================
            delta_display_time = current_time - last_display_time
            # ป้องกันการหารด้วยศูนย์ หากเฟรมเรตเร็วมากๆ
            if delta_display_time > 0:
                display_fps = 1 / delta_display_time
            else:
                display_fps = 999.0 # แสดงค่าสูงๆ แทน
            
            last_display_time = current_time
            # ===================================================================

            cv2.putText(frame, f"FPS: {display_fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"e_x: {err_x:.2f}, e_y: {err_y:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Sp_x: {speed_x:.2f}, Sp_y: {speed_y:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.circle(frame, (int(center_x_orig), int(center_y_orig)), 5, (255, 0, 255), -1)

            cv2.imshow("Robomaster Pink Cup Detection with PID", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'): break
    finally:
        print("Stopping detection and releasing resources.")
        ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
        cv2.destroyAllWindows()
        ep_camera.stop_video_stream()
        ep_robot.close()

# ส่วนที่เหลือของโค้ด (ฟังก์ชันต่างๆ) ไม่ต้องเปลี่ยนแปลง
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


if __name__ == '__main__':
    main()