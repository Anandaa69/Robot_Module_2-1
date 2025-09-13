# calibrate_k.py (เวอร์ชันอัปเกรดสมบูรณ์)
import cv2
import numpy as np
from robomaster import robot
from robomaster import camera
import time

# --- [เพิ่มใหม่] นำฟังก์ชันทั้งหมดมาจากโค้ดหลัก ---
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

# --- ค่าคงที่สำหรับการปรับเทียบ ---
# 1. [สำคัญ!] วัดระยะทางจริง แล้วใส่ค่าที่นี่ (หน่วยเป็น cm)
#    - สำหรับ K1 (ใกล้) แนะนำ 50.0
#    - สำหรับ K2 (กลาง) แนะนำ 150.0
#    - สำหรับ K3 (ไกล) แนะนำ 300.0
Z_KNOWN_DISTANCE_CM = 50.0 

# 2. ใส่ขนาดจริงของวัตถุ (หน่วยเป็น cm)
REAL_WIDTH_CM = 23.9
REAL_HEIGHT_CM = 13.9

# 3. [สำคัญ!] ใส่รายชื่อ Template ทั้ง 3 ไฟล์
TEMPLATE_FILES = [
    "image/template/use/template_night_pic1_x_557_y_278_w_120_h_293.jpg",
    "image/template/use/template_new3_pic2_x_552_y_271_w_90_h_192.jpg",
    "image/template/use/template_new1_pic3_x_607_y_286_w_70_h_142.jpg"
]
MATCH_THRESHOLD = 0.55
IOU_THRESHOLD = 0.1
Y_AXIS_ADJUSTMENT = 25
PROCESSING_WIDTH = 640.0

# --- ส่วนการทำงานหลัก ---
def main():
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")
    ep_robot.set_robot_mode(mode=robot.FREE)
    ep_camera = ep_robot.camera
    ep_gimbal = ep_robot.gimbal

    ep_camera.start_video_stream(display=False, resolution=camera.STREAM_720P)
    
    print("Recentering gimbal...")
    ep_gimbal.recenter(pitch_speed=150, yaw_speed=150).wait_for_completed()
    time.sleep(1) 

    # --- [ปรับแก้] โหลดและเตรียม Template ทั้ง 3 อัน ---
    frame_for_scale = ep_camera.read_cv2_image(timeout=5)
    if frame_for_scale is None:
        print("Error: Could not get frame from camera."); ep_robot.close(); return
        
    h_orig, w_orig, _ = frame_for_scale.shape
    processing_scale = PROCESSING_WIDTH / w_orig
    
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
        print(f"Error: {e}"); return
    
    print(f"\n!!! Calibration Mode: Find Best Template !!!")
    print(f"Place object at a known distance of {Z_KNOWN_DISTANCE_CM} cm.")
    print("Press 'c' to capture pixel dimensions and calculate K.")
    print("Press 'q' to quit.")

    try:
        while True:
            frame = ep_camera.read_cv2_image()
            if frame is None: continue

            # --- [ปรับแก้] ใช้วิธีตรวจจับแบบเดียวกับโค้ดหลัก ---
            frame_proc = cv2.resize(frame, (int(PROCESSING_WIDTH), int(h_orig * processing_scale)), interpolation=cv2.INTER_AREA)
            frame_rgb = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2RGB)
            main_pink_mask = create_pink_mask(frame_rgb)
            main_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
            main_masked = cv2.bitwise_and(main_gray, main_gray, mask=main_pink_mask)

            all_detections = []
            for i, tmpl_masked in enumerate(templates_masked):
                boxes = match_template_masked(main_masked, tmpl_masked, threshold=MATCH_THRESHOLD)
                for top_left, bottom_right, confidence in boxes:
                    all_detections.append((top_left, bottom_right, confidence, i, (0,255,0)))
            
            final_detections = advanced_prioritized_nms(all_detections, IOU_THRESHOLD)
            
            key_pressed = cv2.waitKey(1) & 0xFF

            if final_detections:
                best_detection = final_detections[0]
                top_left_proc, bottom_right_proc, confidence, template_id, _ = best_detection

                tl_orig_unadjusted = (int(top_left_proc[0] / processing_scale), int(top_left_proc[1] / processing_scale))
                br_orig_unadjusted = (int(bottom_right_proc[0] / processing_scale), int(bottom_right_proc[1] / processing_scale))
                
                tl_adjusted = (tl_orig_unadjusted[0], tl_orig_unadjusted[1] + Y_AXIS_ADJUSTMENT)
                br_adjusted = (br_orig_unadjusted[0], br_orig_unadjusted[1] + Y_AXIS_ADJUSTMENT)
                cv2.rectangle(frame, tl_adjusted, br_adjusted, (0, 255, 0), 2)
                label = f"Detected T{template_id+1}"
                cv2.putText(frame, label, (tl_adjusted[0], tl_adjusted[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

                roi_x1, roi_y1 = max(0, tl_adjusted[0]), max(0, tl_adjusted[1])
                roi_x2, roi_y2 = min(frame.shape[1], br_adjusted[0]), min(frame.shape[0], br_adjusted[1])

                if roi_x2 > roi_x1 and roi_y2 > roi_y1:
                    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
                    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                    roi_pink_mask = create_pink_mask(roi_rgb)
                    contours, _ = cv2.findContours(roi_pink_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    if contours:
                        main_contour = max(contours, key=cv2.contourArea)
                        x, y, w, h = cv2.boundingRect(main_contour)

                        real_top_left = (roi_x1 + x, roi_y1 + y)
                        real_bottom_right = (roi_x1 + x + w, roi_y1 + y + h)
                        cv2.rectangle(frame, real_top_left, real_bottom_right, (0, 255, 255), 2)
                        
                        cv2.putText(frame, f"Width (px): {w}", (real_top_left[0], real_top_left[1] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                        cv2.putText(frame, f"Height (px): {h}", (real_top_left[0], real_top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                        if key_pressed == ord('c'):
                            if w > 0 and h > 0:
                                calculated_kx = (Z_KNOWN_DISTANCE_CM * w) / REAL_WIDTH_CM
                                calculated_ky = (Z_KNOWN_DISTANCE_CM * h) / REAL_HEIGHT_CM
                                
                                print("\n" + "="*30)
                                print(f"!!! CALIBRATION RESULTS (for T{template_id+1} at {Z_KNOWN_DISTANCE_CM}cm) !!!")
                                print(f"  - Measured Pixel Width (P_x): {w} px")
                                print(f"  - Measured Pixel Height (P_y): {h} px")
                                print("-" * 30)
                                print(f"==> Calculated K_X = {calculated_kx:.4f}")
                                print(f"==> Calculated K_Y = {calculated_ky:.4f}")
                                print("="*30 + "\n")
            
            cv2.line(frame, (0, h_orig // 2), (w_orig, h_orig // 2), (0, 255, 0), 1)
            cv2.line(frame, (w_orig // 2, 0), (w_orig // 2, h_orig), (0, 255, 0), 1)

            cv2.imshow("Calibration", frame)
            if key_pressed == ord('q'):
                break
    finally:
        print("Stopping calibration and releasing resources.")
        cv2.destroyAllWindows()
        ep_camera.stop_video_stream()
        ep_robot.close()

if __name__ == '__main__':
    main()