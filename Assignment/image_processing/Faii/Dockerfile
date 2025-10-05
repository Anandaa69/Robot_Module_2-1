# ใช้ Python 3.9 บน amd64 (แก้ปัญหาพวก wheel ไม่รองรับ arm64)
FROM --platform=linux/amd64 python:3.9-slim

# ทำงานในโฟลเดอร์ /app
WORKDIR /app

# ติดตั้ง dependency เบื้องต้น
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# คัดลอก requirements.txt เข้าไป
COPY requirements.txt 

# ติดตั้ง lib หลักก่อน (ที่ไม่งอแง)
RUN pip install --no-cache-dir numpy opencv-python pandas matplotlib pillow scipy netaddr netifaces

# พยายามติดตั้ง robomaster แยก (ถ้าไม่ได้ก็ไม่พังทั้งก้อน)
RUN pip install --no-cache-dir robomaster==0.1.1.68 || echo "robomaster install failed, skipping..."

# คัดลอกไฟล์โปรเจกต์ทั้งหมดเข้าไป
COPY . .

# คำสั่งเริ่มต้น
CMD ["python", "cope_camera.py"]
