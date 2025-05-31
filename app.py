from flask import Flask, request, render_template, send_from_directory
import cv2
import os
import torch
from ultralytics import YOLO
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Folders
UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load and prepare model once at startup
model = YOLO("best.pt")
model.fuse()  # Optional but helps optimize model
model.predict(source="assets/dummy.jpg", imgsz=640, save=False, verbose=False)  # Warm-up with dummy image

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        file = request.files.get("file")
        if file:
            img_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(img_path)

            # Run inference with no_grad to save memory
            with torch.no_grad():
                results = model(img_path)

            # Plot and save output image
            output_img = results[0].plot()
            output_path = os.path.join(RESULT_FOLDER, file.filename)
            cv2.imwrite(output_path, output_img)

            # Extract detections
            detections = []
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                class_name = results[0].names[class_id]
                confidence = round(box.conf[0].item() * 100, 2)
                detections.append({"class": class_name, "confidence": confidence})

            return render_template("result.html", image_file=file.filename, detections=detections)

    return render_template("index.html")

@app.route("/results/<filename>")
def get_result_image(filename):
    return send_from_directory(RESULT_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)
