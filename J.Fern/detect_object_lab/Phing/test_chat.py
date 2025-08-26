import cv2

image_folder_path

# โหลดภาพต้นฉบับ
image = cv2.imread(".\image\capture_1756180004.jpg.jpg")

# ให้เลือก ROI ด้วยการลากเมาส์
roi = cv2.selectROI("Select Object", image, showCrosshair=True, fromCenter=False)

# roi จะได้เป็น (x, y, w, h)
x, y, w, h = roi
template = image[y:y+h, x:x+w]

# บันทึกและแสดง template
cv2.imwrite("template.jpg", template)
cv2.imshow("Template", template)
cv2.waitKey(0)
cv2.destroyAllWindows()

