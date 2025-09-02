import cv2
import numpy as np
import time
from robomaster import robot
from robomaster import camera

# --------------------------------------------------------------------
# ส่วนที่ 1: ฟังก์ชัน (เพิ่มฟังก์ชันใหม่)
# --------------------------------------------------------------------

def create_pink_mask(img_rgb):
    # ... (ไม่มีการเปลี่ยนแปลง)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower_pink = np.array([120, 10, 100])
    upper_pink = np.array([170, 100, 200])
    mask = cv2.inRange(hsv, lower_pink, upper_pink)
    mask = cv2.medianBlur(mask, 5)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

# ==========================[ ฟังก์ชันใหม่ ]===========================
def refine_box_by_white_edges(full_frame, original_tl, original_br, padding=5):
    """
    ปรับแก้ Bounding Box ให้แม่นยำขึ้นโดยหาขอบสีขาวภายใน Box เดิม
    :param full_frame: เฟรมภาพต้นฉบับขนาดเต็ม
    :param original_tl: พิกัด Top-Left ของ Box เดิม (x, y)
    :param original_br: พิกัด Bottom-Right ของ Box เดิม (x, y)
    :param padding: จำนวน pixel ที่จะขยายกรอบสุดท้ายออกไปเพื่อความสวยงาม
    :return: พิกัด (new_tl, new_br) ของ Box ที่ปรับแก้แล้ว
    """
    # ป้องกัน error หากพิกัดอยู่นอกขอบเขตภาพ
    h_frame, w_frame, _ = full_frame.shape
    x1, y1 = max(0, original_tl[0]), max(0, original_tl[1])
    x2, y2 = min(w_frame, original_br[0]), min(h_frame, original_br[1])

    # ถ้ากรอบไม่มีพื้นที่ ให้คืนค่าเดิม
    if x1 >= x2 or y1 >= y2:
        return original_tl, original_br

    # ตัดภาพเฉพาะส่วนที่สนใจ (ROI)
    roi = full_frame[y1:y2, x1:x2]

    # แปลง ROI เป็น HSV เพื่อหาขอบสีขาว
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # กำหนดช่วงสีขาวในระบบ HSV
    # Hue สามารถเป็นค่าใดก็ได้, Saturation ต่ำ, Value สูง
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 50, 255])
    white_mask = cv2.inRange(hsv_roi, lower_white, upper_white)

    # หาตำแหน่งของ pixel ที่เป็นสีขาวทั้งหมด
    white_pixels = np.where(white_mask > 0)
    
    # ถ้าไม่เจอ pixel สีขาวเลย ให้คืนค่ากรอบเดิม
    if not white_pixels[0].size > 0:
        return original_tl, original_br

    # หาขอบเขต (ซ้ายสุด, ขวาสุด, บนสุด, ล่างสุด) ของ pixel สีขาว
    min_y = np.min(white_pixels[0])
    max_y = np.max(white_pixels[0])
    min_x = np.min(white_pixels[1])
    max_x = np.max(white_pixels[1])

    # แปลงพิกัดที่ได้ (ซึ่งเป็นพิกัดภายใน ROI) กลับเป็นพิกัดของเฟรมเต็ม
    # และเพิ่ม padding เข้าไป
    new_tl_x = max(0, x1 + min_x - padding)
    new_tl_y = max(0, y1 + min_y - padding)
    new_br_x = min(w_frame, x1 + max_x + padding)
    new_br_y = min(h_frame, y1 + max_y + padding)
    
    return (new_tl_x, new_tl_y), (new_br_x, new_br_y)
# =====================================================================

def match_template_masked(img_masked, tmpl_masked, threshold=0.7):
    # ... (ไม่มีการเปลี่ยนแปลง)
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
    # ... (ไม่มีการเปลี่ยนแปลง)
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
    # ... (ไม่มีการเปลี่ยนแปลง)
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

# ---------------------------------------------------
# ส่วนที่ 2: การทำงานหลัก (Main Program) - ปรับปรุงเพื่อเรียกใช้ฟังก์ชันใหม่
# ---------------------------------------------------

def main():
    """ฟังก์ชันหลักสำหรับเชื่อมต่อหุ่นยนต์และเริ่มการตรวจจับ"""
    
    PROCESSING_WIDTH = 640.0
    
    TEMPLATE_FILES = [
        "image/template/use/template_night_pic1_x_557_y_278_w_120_h_293.jpg", # T1
        "image/template/use/template_night_pic2_x_609_y_290_w_57_h_138.jpg",  # T2
        "image/template/use/template_night_pic3_x_622_y_293_w_40_h_93.jpg"   # T3
    ]
    MATCH_THRESHOLD = 0.55
    IOU_THRESHOLD = 0.1
    # ตัวแปรสำหรับชดเชยตำแหน่งแกน Y (ยังคงไว้เผื่อต้องใช้ปรับแก้ขั้นสุดท้าย)
    Y_AXIS_ADJUSTMENT = 0 

    # (ส่วนของการโหลด template และเชื่อมต่อหุ่นยนต์ เหมือนเดิมทั้งหมด)
    # ...
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
    ep_camera.start_video_stream(display=False, resolution=camera.STREAM_720P)
    
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
        
    print("Robot connected. Starting real-time detection...")

    try:
        last_time = time.time()
        while True:
            frame = ep_camera.read_cv2_image()
            if frame is None: continue

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
            
            current_time = time.time()
            fps = 1 / (current_time - last_time)
            last_time = current_time
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            for top_left, bottom_right, confidence, template_id, color in final_detections:
                # แปลงพิกัดกลับไปเป็นขนาดของเฟรมดั้งเดิม
                tl_orig = (int(top_left[0] / processing_scale), int(top_left[1] / processing_scale))
                br_orig = (int(bottom_right[0] / processing_scale), int(bottom_right[1] / processing_scale))
                
                # ==========================[ จุดปรับแก้ ]===========================
                # เรียกใช้ฟังก์ชันใหม่เพื่อปรับแก้กรอบให้แม่นยำขึ้น
                refined_tl, refined_br = refine_box_by_white_edges(frame, tl_orig, br_orig, padding=5)
                
                # นำค่า Y_AXIS_ADJUSTMENT มาใช้กับกรอบที่ปรับแก้แล้ว (ถ้าจำเป็น)
                tl_adjusted = (refined_tl[0], refined_tl[1] + Y_AXIS_ADJUSTMENT)
                br_adjusted = (refined_br[0], refined_br[1] + Y_AXIS_ADJUSTMENT)
                # =================================================================

                # ใช้พิกัดที่ปรับแก้แล้ว (adjusted) ในการวาด
                cv2.rectangle(frame, tl_adjusted, br_adjusted, color, 3)
                label = f"T{template_id+1} (Refined Fit)" # เปลี่ยนป้ายชื่อ
                cv2.putText(frame, label, (tl_adjusted[0], tl_adjusted[1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            cv2.imshow("Robomaster Pink Cup Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'): break
    finally:
        print("Stopping detection and releasing resources.")
        cv2.destroyAllWindows()
        ep_camera.stop_video_stream()
        ep_robot.close()

if __name__ == '__main__':
    main()