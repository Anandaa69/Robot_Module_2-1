import cv2
import os

# โหลดภาพต้นฉบับ
image_path = "./long.jpg"

image = cv2.imread(image_path)

# ให้เลือก ROI ด้วยการลากเมาส์
roi = cv2.selectROI("Select Object", image, showCrosshair=True, fromCenter=False)

# roi จะได้เป็น (x, y, w, h)
x, y, w, h = roi
print(f"ROI ที่เลือก: x={x}, y={y}, w={w}, h={h}")

# ตัดภาพตาม ROI
template = image[y:y+h, x:x+w]

save_dir = "./image/template/use"

# สร้าง path สำหรับบันทึกไฟล์
save_path = os.path.join(save_dir, f"template_new1_pic3_x_{x}_y_{y}_w_{w}_h_{h}.jpg")
print(f"จะบันทึกไฟล์ที่: {save_path}")

# บันทึกไฟล์
try:
    success = cv2.imwrite(save_path, template)
    if success:
        print(f"บันทึกสำเร็จ: {save_path}")
    else:
        print(f"บันทึกไม่สำเร็จ: {save_path}")
except Exception as e:
    print(f"Error บันทึกไฟล์: {e}")

cv2.imshow("Template", template)

cv2.waitKey(0)
cv2.destroyAllWindows()