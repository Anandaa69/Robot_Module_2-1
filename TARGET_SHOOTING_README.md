# Target Shooting Mission - round2.py

## ภาพรวม
ไฟล์ `round2.py` ได้รับการแก้ไขให้เปลี่ยนจากการสำรวจแมพเป็นการเดินไปยิง target ที่กำหนด โดยอ่านค่าจาก JSON ไฟล์ที่เซฟไว้

## ฟีเจอร์หลัก

### 1. การโหลดข้อมูลจาก JSON
- **Mapping_Top.json**: โหลดแมพและข้อมูล wall
- **Detected_Objects.json**: โหลดข้อมูล target ที่ต้องยิง
- **Robot_Position_Timestamps.json**: โหลดตำแหน่งเริ่มต้นของ robot

### 2. การนำทางไปยัง Target
- คำนวณเส้นทางจากจุดเริ่มต้นไปยังโหนดก่อนหน้าที่เจอ target
- ใช้ BFS algorithm สำหรับการหาเส้นทาง
- รองรับการตรวจสอบ wall และคำนวณเส้นทางใหม่

### 3. ระบบการยิง Target
- ใช้ PID tracking สำหรับการยิง target
- รองรับการยิงหลาย target ตามลำดับ
- มีระบบ timeout และ error handling

## ฟังก์ชันใหม่ที่เพิ่มเข้ามา

### `load_target_data_from_json()`
โหลดข้อมูล target จาก Detected_Objects.json

### `load_robot_position_from_json()`
โหลดตำแหน่งเริ่มต้นของ robot จาก Robot_Position_Timestamps.json

### `find_target_shooting_position(target_node, occupancy_map)`
หาตำแหน่งที่เหมาะสมสำหรับยิง target โดยอยู่โหนดก่อนหน้าที่เจอ target

### `execute_target_shooting_mission()`
ดำเนินการยิง target ตามลำดับ

### `check_wall_discrepancy()`
ตรวจสอบว่า wall ที่พบจริงต่างจากข้อมูลใน JSON หรือไม่

## การใช้งาน

1. วางไฟล์ JSON ทั้ง 3 ไฟล์ในโฟลเดอร์ `Assignment/dude/James_path/`
2. รันโปรแกรม `round2.py`
3. โปรแกรมจะโหลดข้อมูลจาก JSON และเริ่มการยิง target อัตโนมัติ

## ข้อมูล JSON ที่ต้องการ

### Mapping_Top.json
- ข้อมูลแมพและ wall probabilities
- ข้อมูล target ในแต่ละโหนด

### Detected_Objects.json
- รายการ target ที่ต้องยิง
- ข้อมูลตำแหน่งที่เจอ target

### Robot_Position_Timestamps.json
- ตำแหน่งเริ่มต้นของ robot
- ทิศทางเริ่มต้น

## การปรับแต่ง

สามารถปรับแต่งพารามิเตอร์ต่างๆ ได้ในส่วน Configuration:
- `FIRE_SHOTS_COUNT`: จำนวนการยิงต่อ target
- `PID_KP`, `PID_KI`, `PID_KD`: พารามิเตอร์ PID
- `TARGET_SHAPE`, `TARGET_COLOR`: รูปแบบและสีของ target
