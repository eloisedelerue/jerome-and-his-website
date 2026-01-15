from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
import os
import traceback
import time
import numpy as np
import torch
from werkzeug.utils import secure_filename
from utils.yolov8_object_detector import YOLOv8ObjectDetector
from models.gradcam import YOLOV8GradCAM

app = Flask(__name__)

app.secret_key = ''

@app.route('/images/<path:filename>')
def serve_images(filename):
    return send_from_directory('static/images', filename)

@app.route('/css/<path:filename>')
def serve_css(filename):
    return send_from_directory('static/css', filename)

@app.route('/js/<path:filename>')
def serve_js(filename):
    return send_from_directory('static/js', filename)

@app.route('/')
def home():
    return render_template('project.html')

UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

IMG_SIZE = (640, 640)
TARGET_LAYER = 'SPPF'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

TARGET_CLASSES_INTERNAL = ['person', 'human']
DISPLAY_LABEL = "Person"

print(f"Loading models on {DEVICE}...")

model_base_path = os.path.join('models', 'yolov8n.pt')
model_jerome_path = os.path.join('models', 'jerome.pt')

model_base = YOLOv8ObjectDetector(
    model_path=model_base_path, 
    device=DEVICE, 
    img_size=IMG_SIZE, 
    confidence=0.3
)
cam_base = YOLOV8GradCAM(model=model_base, layer_name=TARGET_LAYER, img_size=IMG_SIZE)

model_jerome = YOLOv8ObjectDetector(
    model_path=model_jerome_path, 
    device=DEVICE, 
    img_size=IMG_SIZE, 
    confidence=0.3
)
cam_jerome = YOLOV8GradCAM(model=model_jerome, layer_name=TARGET_LAYER, img_size=IMG_SIZE)


def process_and_save(model, cam_wrapper, img_path, filename_prefix, forced_label=None):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise ValueError("Unable to read image")

    h, w = img_bgr.shape[:2]
    original_shape = (h, w)

    torch_img, _ = model.preprocessing(img_bgr[..., ::-1]) 

    masks, _, out = cam_wrapper(torch_img, original_shape=original_shape)
    boxes, classes, class_names, confs = out

    res_img = img_bgr.copy()
    valid_masks = []
    valid_boxes = []
    valid_labels = []

    if boxes and len(boxes[0]) > 0:
        for i, cls_name_internal in enumerate(class_names[0]):
            is_target = any(target.lower() in cls_name_internal.lower() for target in TARGET_CLASSES_INTERNAL)
            
            if is_target:
                if i < len(masks):
                    resized_mask = cv2.resize(masks[i], (w, h))
                    valid_masks.append(resized_mask)
                
                valid_boxes.append(boxes[0][i])
                
                lbl_txt = forced_label if forced_label else cls_name_internal
                valid_labels.append(f"{lbl_txt} {confs[0][i]:.2f}") 

    if valid_masks:
        combined_mask = np.max(np.array(valid_masks), axis=0)
        heatmap = cv2.applyColorMap(np.uint8(255 * combined_mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        
        res_img_float = np.float32(res_img) / 255
        cam = heatmap + res_img_float
        cam = cam / np.max(cam)
        res_img = np.uint8(255 * cam)

        for bbox, label in zip(valid_boxes, valid_labels):
            x1, y1, x2, y2 = bbox
            cv2.rectangle(res_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            (w_txt, h_txt), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(res_img, (x1, y1 - h_txt - 5), (x1 + w_txt, y1), (0, 0, 0), -1)
            cv2.putText(res_img, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    output_filename = f"result_{filename_prefix}_{os.path.basename(img_path)}"
    output_path = os.path.join(RESULT_FOLDER, output_filename)
    cv2.imwrite(output_path, res_img)
    
    detection_count = len(valid_boxes)
    return output_filename, detection_count

@app.route('/compare', methods=['POST'])
def compare():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        start = time.time()
        res_base, count_base = process_and_save(
            model_base, cam_base, filepath, "yolov8n", forced_label=DISPLAY_LABEL
        )
        time_base = time.time() - start

        start = time.time()
        res_jerome, count_jerome = process_and_save(
            model_jerome, cam_jerome, filepath, "jerome", forced_label=DISPLAY_LABEL
        )
        time_jerome = time.time() - start

        return jsonify({
            'result_jerome': f"/results/{res_jerome}",
            'result_yolov8n': f"/results/{res_base}",
            'time_jerome': time_jerome,
            'time_yolov8n': time_base,
            'num_detections_jerome': count_jerome,
            'num_detections_yolov8n': count_base,
            'original_image': f"/uploads/{filename}"
        })

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/results/<filename>')
def send_result(filename):
    return send_from_directory(RESULT_FOLDER, filename)

@app.route('/uploads/<filename>')
def send_upload(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000)