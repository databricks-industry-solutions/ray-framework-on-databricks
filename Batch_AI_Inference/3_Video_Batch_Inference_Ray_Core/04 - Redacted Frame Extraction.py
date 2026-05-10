# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — Redacted Frame Extraction (for talk visuals)
# MAGIC
# MAGIC Extract a single frame from a CONFIRMED shoplifting video, detect any human faces in
# MAGIC it, and pixelate those face regions before saving to a UC Volume. The redacted frame
# MAGIC is intended for visual reference (e.g. slides, blog posts) where the source video is
# MAGIC public domain but the individuals depicted did not consent to public display.
# MAGIC
# MAGIC **Why this matters.** The Kaggle CCTV anomaly dataset is publicly downloadable, but
# MAGIC the people captured in surveillance footage almost certainly did not consent to being
# MAGIC shown in a conference talk or recorded blog. Even when the model's classification
# MAGIC happens to be correct, displaying an identifiable face alongside a "shoplifting
# MAGIC CONFIRMED" label creates a public association between that person's face and a
# MAGIC criminal allegation — without their consent or due process.
# MAGIC
# MAGIC This notebook does the redaction step explicitly so the pattern is auditable: anyone
# MAGIC reproducing the demo gets a redacted frame the same way.
# MAGIC
# MAGIC **Output:** a single PNG written to a UC Volume, with face regions pixelated.

# COMMAND ----------

# DBTITLE 1,Install ffmpeg (system) + opencv (Python)
# MAGIC %sh
# MAGIC apt-get update -qq && apt-get install -y -qq ffmpeg
# MAGIC ffprobe -version | head -1

# COMMAND ----------

# MAGIC %pip install --quiet "numpy<2" "opencv-python-headless==4.10.0.84" "mediapipe==0.10.18"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Widgets
dbutils.widgets.text("CATALOG", "samantha_wise", label="CATALOG")
dbutils.widgets.text("SCHEMA", "dais_2026", label="SCHEMA")
dbutils.widgets.text("VOLUME", "video_data", label="VOLUME")
dbutils.widgets.text("VIDEO_FILE", "Shoplifting028_x264.mp4", label="VIDEO_FILE")
dbutils.widgets.text("VIDEO_CATEGORY", "shoplifting", label="VIDEO_CATEGORY")
dbutils.widgets.text("FRAME_AT_SECONDS", "20", label="FRAME_AT_SECONDS")
dbutils.widgets.text("OUTPUT_FILENAME", "Shoplifting028_redacted_frame.png", label="OUTPUT_FILENAME")

CATALOG = dbutils.widgets.get("CATALOG")
SCHEMA = dbutils.widgets.get("SCHEMA")
VOLUME = dbutils.widgets.get("VOLUME")
VIDEO_FILE = dbutils.widgets.get("VIDEO_FILE")
VIDEO_CATEGORY = dbutils.widgets.get("VIDEO_CATEGORY")
FRAME_AT_SECONDS = float(dbutils.widgets.get("FRAME_AT_SECONDS"))
OUTPUT_FILENAME = dbutils.widgets.get("OUTPUT_FILENAME")

video_path = (
    f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/raw_video/CCTV/data/{VIDEO_CATEGORY}/{VIDEO_FILE}"
)
output_dir = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/redacted_frames"
output_path = f"{output_dir}/{OUTPUT_FILENAME}"

import os
os.makedirs(output_dir, exist_ok=True)
print(f"Source video:  {video_path}")
print(f"Output:        {output_path}")
print(f"Frame at:      {FRAME_AT_SECONDS}s")

# COMMAND ----------

# DBTITLE 1,Extract frame via ffmpeg
import subprocess

raw_frame_path = f"/tmp/{VIDEO_FILE}.raw_frame.png"
result = subprocess.run(
    [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", str(FRAME_AT_SECONDS),
        "-i", video_path,
        "-frames:v", "1",
        "-update", "1",
        raw_frame_path,
    ],
    capture_output=True, text=True,
)
if result.returncode != 0:
    raise RuntimeError(f"ffmpeg failed: {result.stderr}")
print(f"Frame extracted to {raw_frame_path}")

# COMMAND ----------

# DBTITLE 1,Detect faces and pixelate
import cv2
import mediapipe as mp
import numpy as np

img = cv2.imread(raw_frame_path)
if img is None:
    raise RuntimeError(f"Could not read frame from {raw_frame_path}")
h, w = img.shape[:2]
print(f"Frame size: {w}x{h}")

# --- Pass 1: MediaPipe ---
mp_face = mp.solutions.face_detection
faces_mp = []
with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.3) as fd:
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = fd.process(rgb)
    if res.detections:
        for d in res.detections:
            box = d.location_data.relative_bounding_box
            x, y, bw, bh = int(box.xmin * w), int(box.ymin * h), int(box.width * w), int(box.height * h)
            faces_mp.append((x, y, bw, bh))
print(f"MediaPipe found {len(faces_mp)} face(s)")

# --- Pass 2: OpenCV Haar (frontal + profile) ---
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cascade_paths = [
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml",
    cv2.data.haarcascades + "haarcascade_profileface.xml",
]
faces_cv = []
for path in cascade_paths:
    cascade = cv2.CascadeClassifier(path)
    detections = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))
    faces_cv.extend(list(detections))
print(f"OpenCV Haar found {len(faces_cv)} face(s)")

all_boxes = list(faces_mp) + [tuple(b) for b in faces_cv]
print(f"Total face regions to pixelate: {len(all_boxes)}")

def pixelate_region(image, x, y, bw, bh, mosaic_size=12):
    x = max(0, x); y = max(0, y)
    bw = min(image.shape[1] - x, bw); bh = min(image.shape[0] - y, bh)
    if bw <= 0 or bh <= 0:
        return
    region = image[y:y+bh, x:x+bw]
    small = cv2.resize(region, (max(1, bw // mosaic_size), max(1, bh // mosaic_size)), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(small, (bw, bh), interpolation=cv2.INTER_NEAREST)
    image[y:y+bh, x:x+bw] = pixelated

margin = 0.3
for (x, y, bw, bh) in all_boxes:
    cx, cy = x + bw / 2, y + bh / 2
    new_bw = int(bw * (1 + margin))
    new_bh = int(bh * (1 + margin))
    new_x = int(cx - new_bw / 2)
    new_y = int(cy - new_bh / 2)
    pixelate_region(img, new_x, new_y, new_bw, new_bh, mosaic_size=14)

# COMMAND ----------

# DBTITLE 1,Safety: if no face was detected, blur upper third as fallback
if not all_boxes:
    print("WARNING: no face detected — applying conservative blur to upper third of frame")
    upper_h = h // 3
    upper = img[0:upper_h, :]
    blurred = cv2.GaussianBlur(upper, (51, 51), 0)
    img[0:upper_h, :] = blurred

# COMMAND ----------

# DBTITLE 1,Add small "Face redacted" annotation
ann_text = "Face redacted (auto-detected, pixelated)"
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.55
thickness = 1
(text_w, text_h), baseline = cv2.getTextSize(ann_text, font, font_scale, thickness)
margin_px = 8
x = margin_px
y = h - margin_px
overlay = img.copy()
cv2.rectangle(overlay, (x - 4, y - text_h - 4), (x + text_w + 4, y + baseline), (0, 0, 0), -1)
img = cv2.addWeighted(overlay, 0.55, img, 0.45, 0)
cv2.putText(img, ann_text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

# COMMAND ----------

# DBTITLE 1,Save redacted frame to UC Volume
ok = cv2.imwrite(output_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 6])
if not ok:
    raise RuntimeError(f"Failed to write {output_path}")
size = os.path.getsize(output_path)
print(f"Wrote redacted frame to {output_path} ({size:,} bytes, {w}x{h})")

# COMMAND ----------

display(spark.read.format("binaryFile").load(output_path).select("path", "length"))
