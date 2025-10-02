# Dockerfile สำหรับ Mac M1/M2
# ใช้ platform amd64 เพื่อให้ robomaster ติดตั้งได้
FROM --platform=linux/amd64 python:3.9-slim

# ตั้ง working directory ใน container
WORKDIR /app

# 1. ติดตั้งไลบรารีของระบบที่จำเป็นสำหรับ OpenCV
# ไลบรารีเหล่านี้จำเป็นเพื่อให้ไบนารีของ OpenCV ทำงานได้ในสภาพแวดล้อม Linux
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 && \
    rm -rf /var/lib/apt/lists/*

# 2. คัดลอก requirements.txt และติดตั้ง dependency ของ Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. คัดลอกไฟล์โปรเจกต์ทั้งหมด
COPY . .

# 4. รัน cope_camera.py เมื่อ container start
CMD ["python", "cope_camera.py"]
