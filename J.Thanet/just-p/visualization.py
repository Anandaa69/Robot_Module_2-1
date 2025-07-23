import csv
import matplotlib.pyplot as plt

times = []
x_values = []

# อ่านข้อมูลจาก CSV
with open("project_2\kp_2_5.csv", mode="r") as csvfile:
    csv_reader = csv.DictReader(csvfile)
    for row in csv_reader:
        times.append(float(row["elapsed_time"]))
        x_values.append(float(row["current_x"]))

# พล็อตกราฟ
plt.figure(figsize=(10, 5))
plt.plot(times, x_values, marker='o', linestyle='-')
plt.xlabel("time (sec)")
plt.ylabel("x-position (meter)")
plt.grid(True)
plt.show()
