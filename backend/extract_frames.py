import cv2
import os


video_folder = "videos"   # folder where your .mp4 files are
output_real = "dataset/real"
output_fake = "dataset/fake"

os.makedirs(output_real, exist_ok=True)
os.makedirs(output_fake, exist_ok=True)

def extract(video_path, output_folder, label):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("ERROR: Cannot open", video_path)

    count = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % 10 == 0:  # take every 10th frame
            filename = f"{label}_{saved}.jpg"
            cv2.imwrite(os.path.join(output_folder, filename), frame)
            saved += 1

        count += 1

    cap.release()


# ----------- CHANGE THESE FOLDERS -----------
real_videos = "videos/real"
fake_videos = "videos/fake"
# -------------------------------------------

for vid in os.listdir(real_videos):
    extract(os.path.join(real_videos, vid), output_real, "real")

for vid in os.listdir(fake_videos):
    extract(os.path.join(fake_videos, vid), output_fake, "fake")

print("Frames extracted successfully!")