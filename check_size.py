import cv2
import csv

video = "test_video.mp4"
csv_path = r"prediction\test_video_ball.csv"  # ✅ 用 prediction 路徑

cap = cv2.VideoCapture(video)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
cap.release()

x_min = float("inf"); x_max = float("-inf")
y_min = float("inf"); y_max = float("-inf")
row_count = 0

with open(csv_path, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            x = float(row["X"])
            y = float(row["Y"])
        except Exception:
            continue
        x_min = min(x_min, x); x_max = max(x_max, x)
        y_min = min(y_min, y); y_max = max(y_max, y)
        row_count += 1

print("OpenCV video size:", w, "x", h, "fps:", fps)
print("CSV rows read:", row_count)
print("CSV X range:", x_min, "to", x_max)
print("CSV Y range:", y_min, "to", y_max)