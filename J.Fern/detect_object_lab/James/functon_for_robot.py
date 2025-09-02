import cv2
import numpy as np

class PinkObjectDetector:
    def __init__(self, template_paths, confidence_threshold=0.7, iou_threshold=0.3):
        """
        Pink Object Detector สำหรับ RoboMaster EP S1
        
        Args:
            template_paths (list): รายการ path ของ template images
            confidence_threshold (float): threshold สำหรับ template matching
            iou_threshold (float): threshold สำหรับ NMS
        """
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.templates_masked = []
        self.templates_shapes = []
        
        # โหลดและเตรียม templates
        self._load_templates(template_paths)
    
    def create_pink_mask(self, img_rgb):
        """สร้าง mask สำหรับสีชมพู โดยใช้ HSV color space"""
        # แปลงเป็น HSV เพื่อจับสีชมพูได้ดีกว่า
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        
        # ช่วงค่าสีชมพู
        lower_pink = np.array([120, 20, 80])
        upper_pink = np.array([170, 100, 200])
        
        # สร้าง mask
        mask = cv2.inRange(hsv, lower_pink, upper_pink)
        
        # ลด noise ด้วย morphology operations
        mask = cv2.medianBlur(mask, 5)
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
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
            
            # สร้าง pink mask
            template_mask = self.create_pink_mask(template_rgb)
            
            # แปลงเป็น grayscale และใช้ mask
            template_gray = cv2.cvtColor(template_rgb, cv2.COLOR_RGB2GRAY)
            template_masked = cv2.bitwise_and(template_gray, template_gray, mask=template_mask)
            
            self.templates_masked.append(template_masked)
            self.templates_shapes.append(template_masked.shape)
    
    def match_template_masked(self, img_masked, template_masked):
        """Template matching บน masked grayscale images"""
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
    
    def detect(self, image_rgb):
        """
        ตรวจจับ pink objects ในภาพ
        
        Args:
            image_rgb (numpy.ndarray): ภาพ input ในรูปแบบ RGB
            
        Returns:
            list: รายการของ detection results ในรูปแบบ [x, y, w, h, confidence, template_id]
        """
        # สร้าง pink mask สำหรับภาพหลัก
        main_pink_mask = self.create_pink_mask(image_rgb)
        
        # แปลงเป็น grayscale และใช้ mask
        main_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        main_masked = cv2.bitwise_and(main_gray, main_gray, mask=main_pink_mask)
        
        # รวบรวมการตรวจจับทั้งหมด
        all_detections = []
        
        for template_id, template_masked in enumerate(self.templates_masked):
            boxes_with_scores = self.match_template_masked(main_masked, template_masked)
            
            for detection in boxes_with_scores:
                top_left, bottom_right, confidence = detection
                # เพิ่ม template_id เข้าไป
                all_detections.append((top_left, bottom_right, confidence, template_id))
        
        # ใช้ NMS กับการตรวจจับทั้งหมด
        nms_detections = self.non_maximum_suppression(all_detections)
        
        # แปลงผลลัพธ์เป็น format [x, y, w, h, confidence, template_id]
        final_results = []
        for detection in nms_detections:
            top_left, bottom_right, confidence, template_id = detection
            x, y = top_left
            w = bottom_right[0] - top_left[0]
            h = bottom_right[1] - top_left[1]
            
            final_results.append([x, y, w, h, confidence, template_id])
        
        return final_results
    
    def detect_from_camera_frame(self, frame_bgr):
        """
        ตรวจจับจาก frame ที่ได้จากกล้อง (BGR format)
        
        Args:
            frame_bgr (numpy.ndarray): frame จากกล้อง (BGR format)
            
        Returns:
            list: รายการของ detection results ในรูปแบบ [x, y, w, h, confidence, template_id]
        """
        # แปลงจาก BGR เป็น RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return self.detect(frame_rgb)


# ฟังก์ชันสำหรับใช้งานง่าย
def detect_pink_objects(image_or_frame, template_paths, confidence_threshold=0.7, iou_threshold=0.3, is_bgr=False):
    """
    ฟังก์ชันสำหรับตรวจจับ pink objects แบบง่าย
    
    Args:
        image_or_frame: ภาพ input (RGB หรือ BGR ขึ้นอยู่กับ is_bgr parameter)
        template_paths (list): รายการ path ของ template images
        confidence_threshold (float): threshold สำหรับ template matching
        iou_threshold (float): threshold สำหรับ NMS
        is_bgr (bool): True ถ้า input เป็น BGR format (จากกล้อง), False ถ้าเป็น RGB
        
    Returns:
        list: รายการของ detection results ในรูปแบบ [x, y, w, h, confidence, template_id]
    """
    detector = PinkObjectDetector(template_paths, confidence_threshold, iou_threshold)
    
    if is_bgr:
        return detector.detect_from_camera_frame(image_or_frame)
    else:
        return detector.detect(image_or_frame)


# ตัวอย่างการใช้งานกับ RoboMaster EP S1
def robomaster_detection_example():
    """ตัวอย่างการใช้งานกับ RoboMaster EP S1"""
    
    # กำหนด template paths
    template_paths = [
        "template/template_pic1_x_573_y_276_w_115_h_312.jpg",
        "template/template_pic2_x_634_y_291_w_50_h_134.jpg",
        "template/template_pic3_x_629_y_291_w_35_h_92.jpg"
    ]
    
    # สร้าง detector
    detector = PinkObjectDetector(
        template_paths=template_paths,
        confidence_threshold=0.7,
        iou_threshold=0.3
    )
    
    # สำหรับการใช้งานกับกล้อง RoboMaster
    # โดยทั่วไป frame จากกล้องจะเป็น BGR format
    
    # สมมติว่ามี frame จากกล้อง
    # frame = ... # ได้จากกล้อง RoboMaster
    
    # ตรวจจับ objects
    # detections = detector.detect_from_camera_frame(frame)
    
    # ผลลัพธ์ที่ได้จะเป็น list ของ [x, y, w, h, confidence, template_id]
    # for detection in detections:
    #     x, y, w, h, confidence, template_id = detection
    #     print(f"Template {template_id+1}: x={x}, y={y}, w={w}, h={h}, confidence={confidence:.3f}")
    
    return detector


if __name__ == "__main__":
    # ตัวอย่างการใช้งาน
    print("Pink Object Detector for RoboMaster EP S1")
    print("Functions available:")
    print("1. PinkObjectDetector class - สำหรับใช้งานแบบ object-oriented")
    print("2. detect_pink_objects() - สำหรับใช้งานแบบง่าย")
    print("3. robomaster_detection_example() - ตัวอย่างการใช้งาน")