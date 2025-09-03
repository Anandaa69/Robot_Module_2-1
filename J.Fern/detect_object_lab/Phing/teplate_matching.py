import cv2
import numpy as np
import robomaster
from robomaster import robot
from robomaster import camera


class PinkObjectDetector:
    def __init__(self, template_paths, confidence_threshold=0.4, iou_threshold=0.3):
        """
        Pink Object Detector สำหรับ RoboMaster EP S1 (ปรับปรุง)
        """
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.templates_masked = []
        self.templates_shapes = []
        self.templates_original = []  # เก็บ template ต้นฉบับ
        
        # โหลดและเตรียม templates
        self._load_templates(template_paths)
        
        # ตัวแปรสำหรับ debugging
        self.debug_mode = True
    
    def create_pink_mask_improved(self, img_rgb):
        """สร้าง mask สำหรับสีชมพู ที่ปรับปรุงแล้ว"""
        # แปลงเป็น HSV
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        
        # ขยายช่วงสีชมพูให้กว้างขึ้น (รวมถึงสีม่วงอ่อนและแดงอ่อน)
        # ช่วงที่ 1: ชมพู-ม่วง
        lower_pink1 = np.array([140, 30, 50])
        upper_pink1 = np.array([180, 255, 255])
        
        # ช่วงที่ 2: แดงชมพู
        lower_pink2 = np.array([0, 30, 50])
        upper_pink2 = np.array([20, 255, 255])
        
        # รวม mask ทั้ง 2 ช่วง
        mask1 = cv2.inRange(hsv, lower_pink1, upper_pink1)
        mask2 = cv2.inRange(hsv, lower_pink2, upper_pink2)
        mask = cv2.bitwise_or(mask1, mask2)
        
        # ลด noise แต่อย่าเยอะเกินไป
        mask = cv2.medianBlur(mask, 3)  # ลดจาก 5
        kernel = np.ones((3, 3), np.uint8)  # ลดจาก 7x7
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
    
    def _load_templates(self, template_paths):
        """โหลดและเตรียม templates"""
        for template_path in template_paths:
            # โหลด template
            template = cv2.imread(template_path)
            if template is None:
                print(f"Warning: Could not load template {template_path}")
                continue
            
            # แปลงเป็น RGB
            template_rgb = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
            self.templates_original.append(template_rgb)
            
            # สร้าง pink mask
            template_mask = self.create_pink_mask_improved(template_rgb)
            
            # แปลงเป็น grayscale และใช้ mask
            template_gray = cv2.cvtColor(template_rgb, cv2.COLOR_RGB2GRAY)
            template_masked = cv2.bitwise_and(template_gray, template_gray, mask=template_mask)
            
            # เก็บเฉพาะ template ที่มีเนื้อหา
            if np.sum(template_mask) > 100:  # มี pixel สีชมพูอย่างน้อย 100 pixels
                self.templates_masked.append(template_masked)
                self.templates_shapes.append(template_masked.shape)
                print(f"Loaded template: {template_path}, Pink pixels: {np.sum(template_mask)}")
            else:
                print(f"Warning: Template {template_path} has insufficient pink content")
    
    def detect_using_contours(self, frame_bgr):
        """ตรวจจับโดยใช้ contour detection เป็นทางเลือก"""
        # แปลงเป็น RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        # สร้าง pink mask
        mask = self.create_pink_mask_improved(frame_rgb)
        
        if self.debug_mode:
            cv2.imshow("Pink Mask", mask)
        
        # หา contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # กรองขนาดขั้นต่ำ
                x, y, w, h = cv2.boundingRect(contour)
                # คำนวณ confidence จากอัตราส่วนพื้นที่
                confidence = min(area / (w * h), 1.0)
                if confidence > 0.3:  # threshold สำหรับ shape matching
                    detections.append([x, y, w, h, confidence, 0])  # template_id = 0
        
        return detections
    
    def match_template_masked(self, img_masked, template_masked):
        """Template matching บน masked grayscale images"""
        if img_masked.size == 0 or template_masked.size == 0:
            return []
            
        result = cv2.matchTemplate(img_masked, template_masked, cv2.TM_CCOEFF_NORMED)
        
        # หา location ที่ score >= threshold
        locations = np.where(result >= self.confidence_threshold)
        
        boxes_with_scores = []
        h, w = template_masked.shape
        
        for pt in zip(*locations[::-1]):  # แปลง (row, col) -> (x, y)
            top_left = pt
            bottom_right = (pt[0] + w, pt[1] + h)
            confidence = result[pt[1], pt[0]]
            boxes_with_scores.append((top_left, bottom_right, confidence))
        
        return boxes_with_scores
    
    def calculate_iou(self, box1, box2):
        """คำนวณ Intersection over Union (IoU) ระหว่าง 2 กรอบ"""
        x1_1, y1_1 = box1[0]
        x2_1, y2_1 = box1[1]
        x1_2, y1_2 = box2[0]
        x2_2, y2_2 = box2[1]
        
        # หาพื้นที่ทับซ้อน
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
        
        # หาพื้นที่รวม
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0
    
    def non_maximum_suppression(self, boxes_with_scores):
        """Non-Maximum Suppression เพื่อกำจัดกรอบที่ทับซ้อน"""
        if len(boxes_with_scores) == 0:
            return []
        
        # เรียงตาม confidence score (จากมากไปน้อย)
        boxes_with_scores.sort(key=lambda x: x[2], reverse=True)
        
        selected_boxes = []
        
        while len(boxes_with_scores) > 0:
            # เลือกกรอบที่มี confidence สูงสุด
            current_box = boxes_with_scores.pop(0)
            selected_boxes.append(current_box)
            
            # กำจัดกรอบที่มี IoU สูงเกินไปกับกรอบที่เลือก
            boxes_with_scores = [
                box for box in boxes_with_scores
                if self.calculate_iou(current_box[:2], box[:2]) < self.iou_threshold
            ]
        
        return selected_boxes
    
    def detect_from_camera_frame(self, frame_bgr):
        """
        ตรวจจับจาก frame ที่ได้จากกล้อง (ปรับปรุง)
        """
        # วิธีที่ 1: Template Matching
        template_detections = self._detect_with_template_matching(frame_bgr)
        
        # วิธีที่ 2: Contour Detection (เป็นทางเลือก)
        contour_detections = self.detect_using_contours(frame_bgr)
        
        # รวมผลลัพธ์
        all_detections = template_detections + contour_detections
        
        # ใช้ NMS
        if all_detections:
            # แปลงเป็นรูปแบบที่ NMS ต้องการ
            boxes_with_scores = []
            for detection in all_detections:
                x, y, w, h, confidence, template_id = detection
                top_left = (x, y)
                bottom_right = (x + w, y + h)
                boxes_with_scores.append((top_left, bottom_right, confidence, template_id))
            
            nms_detections = self.non_maximum_suppression(boxes_with_scores)
            
            # แปลงกลับ
            final_results = []
            for detection in nms_detections:
                top_left, bottom_right, confidence, template_id = detection
                x, y = top_left
                w = bottom_right[0] - top_left[0]
                h = bottom_right[1] - top_left[1]
                final_results.append([x, y, w, h, confidence, template_id])
            
            return final_results
        
        return []
    
    def _detect_with_template_matching(self, frame_bgr):
        """Template matching detection"""
        # แปลงจาก BGR เป็น RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        # สร้าง pink mask สำหรับภาพหลัก
        main_pink_mask = self.create_pink_mask_improved(frame_rgb)
        
        if self.debug_mode and np.sum(main_pink_mask) > 0:
            print(f"Pink pixels found: {np.sum(main_pink_mask)}")
        
        # แปลงเป็น grayscale และใช้ mask
        main_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        main_masked = cv2.bitwise_and(main_gray, main_gray, mask=main_pink_mask)
        
        # รวบรวมการตรวจจับทั้งหมด
        all_detections = []
        
        for template_id, template_masked in enumerate(self.templates_masked):
            if template_id < len(template_paths):  # ใช้ template_id จริง
                boxes_with_scores = self.match_template_masked(main_masked, template_masked)
                
                for detection in boxes_with_scores:
                    top_left, bottom_right, confidence = detection
                    x, y = top_left
                    w = bottom_right[0] - top_left[0]
                    h = bottom_right[1] - top_left[1]
                    all_detections.append([x, y, w, h, confidence, template_id])
        
        return all_detections


class PinkObjectInfo:
    """คลาสสำหรับเก็บข้อมูล pink object ที่ตรวจพบ"""
    
    def __init__(self, x, y, w, h, confidence, template_id, img_width=1280, img_height=720):
        self.img_width = img_width
        self.img_height = img_height
        
        # แปลงจากพิกัด pixel เป็น normalized coordinates (0-1)
        self._x = x / img_width + w / (2 * img_width)  # center x
        self._y = y / img_height + h / (2 * img_height)    # center y
        self._w = w / img_width
        self._h = h / img_height
        self._confidence = confidence
        self._template_id = template_id

    @property
    def pt1(self):
        return int((self._x - self._w / 2) * self.img_width), int((self._y - self._h / 2) * self.img_height)

    @property
    def pt2(self):
        return int((self._x + self._w / 2) * self.img_width), int((self._y + self._h / 2) * self.img_height)

    @property
    def center(self):
        return int(self._x * self.img_width), int(self._y * self.img_height)

    @property
    def text(self):
        return f"T{self._template_id+1}:{self._confidence:.2f}"


# ตัวแปรสำหรับเก็บผลลัพธ์การตรวจจับ
pink_objects = []


def detect_pink_objects_in_frame(detector, frame):
    """ตรวจจับ pink objects ใน frame และอัพเดท pink_objects list"""
    global pink_objects
    
    # ตรวจจับ pink objects
    detections = detector.detect_from_camera_frame(frame)
    
    # ล้างข้อมูลเดิม
    pink_objects.clear()
    
    # ตรวจสอบขนาดภาพจริง
    img_height, img_width = frame.shape[:2]
    
    # สร้าง PinkObjectInfo objects
    for detection in detections:
        x, y, w, h, confidence, template_id = detection
        pink_objects.append(PinkObjectInfo(x, y, w, h, confidence, template_id, img_width, img_height))
        print(f"Pink Object - x:{x}, y:{y}, w:{w}, h:{h}, confidence:{confidence:.3f}")


if __name__ == '__main__':
    # กำหนด template paths (แก้ไข path ให้ตรงกับไฟล์จริง)
    template_paths = [
        "template/template_pic1_x_573_y_276_w_115_h_312.jpg",
        "template/template_pic2_x_634_y_291_w_50_h_134.jpg",
        "template/template_pic3_x_629_y_291_w_35_h_92.jpg"
    ]
    
    # สร้าง Pink Object Detector
    detector = PinkObjectDetector(
        template_paths=template_paths,
        confidence_threshold=0.3,  # ลดลงให้ detect ง่ายขึ้น
        iou_threshold=0.4
    )
    
    # เชื่อมต่อ RoboMaster
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")
    
    ep_camera = ep_robot.camera
    
    # เริ่ม video stream ด้วย resolution ที่เหมาะสม
    ep_camera.start_video_stream(display=False, resolution=camera.STREAM_720P)
    
    print("Starting pink object detection...")
    print("Press 'q' to quit, 'c' to capture screenshot, 'd' to toggle debug")
    
    try:
        frame_count = 0
        while True:
            # อ่านภาพจากกล้อง
            img = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
            
            if img is not None:
                frame_count += 1
                
                # ตรวจจับ pink objects
                detect_pink_objects_in_frame(detector, img)
                
                # วาดกรอบและข้อความ
                for obj in pink_objects:
                    # วาดกรอบสีเขียว
                    cv2.rectangle(img, obj.pt1, obj.pt2, (0, 255, 0), 2)
                    
                    # วาดข้อความ (confidence)
                    cv2.putText(img, obj.text, 
                            (obj.center[0] - 50, obj.center[1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # วาดจุดกึ่งกลาง
                    cv2.circle(img, obj.center, 5, (0, 0, 255), -1)
                
                # แสดงจำนวน objects ที่พบ
                cv2.putText(img, f"Pink Objects: {len(pink_objects)}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # แสดง frame count
                cv2.putText(img, f"Frame: {frame_count}", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # แสดงผล
                cv2.imshow("Pink Objects Detection", img)
                
                # ตรวจสอบ key press
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Quitting...")
                    break
                elif key == ord('c'):
                    # บันทึกภาพ
                    filename = f"pink_detection_frame_{frame_count}.jpg"
                    cv2.imwrite(filename, img)
                    print(f"Screenshot saved: {filename}")
                elif key == ord('d'):
                    # เปิด/ปิด debug mode
                    detector.debug_mode = not detector.debug_mode
                    print(f"Debug mode: {'ON' if detector.debug_mode else 'OFF'}")
                
                # แสดง progress ทุก 50 frames
                if frame_count % 50 == 0:
                    print(f"Processed {frame_count} frames, Found {len(pink_objects)} pink objects")
            else:
                print("Failed to read frame from camera")
    
    except KeyboardInterrupt:
        print("Interrupted by user")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # ทำความสะอาด
        cv2.destroyAllWindows()
        ep_camera.stop_video_stream()
        ep_robot.close()
        print("Cleanup completed")