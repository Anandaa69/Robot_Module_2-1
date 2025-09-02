# import numpy as np
# import cv2
# import matplotlib.pyplot as plt

# def sliding_window(image, template, step=1):
#     ih, iw = image.shape
#     th, tw = template.shape

#     result = np.zeros((ih - th + 1, iw - tw + 1))

#     for y in range(0, ih - th + 1, step):
#         for x in range(0, iw - tw + 1, step):
#             patch = image[y:y+th, x:x+tw]
#             ssd = np.sum((patch - template) ** 2)  # SSD
#             result[y, x] = ssd
#     return result

# # โหลดภาพและเทมเพลต (เป็น grayscale)
# image = cv2.imread("./image/capture_1756180004.jpg", cv2.IMREAD_GRAYSCALE)
# template = cv2.imread("./image/template/template_pic1_x_573_y_276_w_115_h_312.jpg", cv2.IMREAD_GRAYSCALE)

# # ทำ sliding window
# result = sliding_window(image, template)

# # หา match ที่ดีที่สุด (ค่าน้อยสุดสำหรับ SSD)
# min_val = np.min(result)
# min_loc = np.unravel_index(np.argmin(result), result.shape)
# print(f"Best match at {min_loc} with SSD={min_val}")

# # วาดสี่เหลี่ยม
# top_left = (min_loc[1], min_loc[0])
# bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])
# detected = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
# cv2.rectangle(detected, top_left, bottom_right, (0, 255, 0), 2)

# cv2.imshow("Detected", detected)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # แสดง heatmap
# plt.imshow(result, cmap='hot')
# plt.colorbar()
# plt.title("SSD Heatmap")
# plt.show()
