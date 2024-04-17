from flask import Flask, request, jsonify, render_template
import cv2
import base64
from io import BytesIO
import numpy as np
import os
from ultralytics import YOLO
import math

app = Flask(__name__)

class_dict = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 
        6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 
        12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 
        20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 
        28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 
        35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 
        41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 
        50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 
        58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 
        66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 
        73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

classsname_list = list(class_dict.values())

def yolo_detection(image_path,filter_item):
    # Load image
    img = cv2.imread(image_path)

    # Load YOLO model
    model = YOLO("yolov8n.pt")
    for id_,class_ in class_dict.items():
        if class_ == filter_item:
            filter_id = id_

    # Perform detection
    results = model.predict(img,classes=[filter_id])
    class_name = []

    # Process detection results
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            class_name = class_dict[cls]

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            label = f"{class_name} - {conf}"
            t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=1)[0]
            c2 = x1 + t_size[0], y1 - t_size[1] - 3
            cv2.rectangle(img, (x1, y1), c2, [255, 255, 255], -1, cv2.LINE_AA)
            cv2.putText(img, label, (x1, y1 - 2), 0, 1, [0, 255, 0], thickness=2, lineType=cv2.LINE_AA)

    return img, class_name

@app.route('/',methods=['GET'])
def home():
    return render_template('combined.html',classsname_list=classsname_list)

@app.route('/process_image', methods=['POST'])
def process_image():
    # Get image path and selected filter item from request
    image_path = request.form['image_path']
    filter_item = request.form['filter_item']  

    # Perform YOLO detection
    processed_image, class_name = yolo_detection(image_path,filter_item)

    # Convert processed image to base64
    _, buffer = cv2.imencode('.jpg', processed_image)
    encoded_image = base64.b64encode(buffer).decode('utf-8')

    return jsonify({'image': encoded_image})

@app.route('/relevancy_filter',methods=['POST'])
def checkRelevancy():
    # Get image path and selected filter item from request
    image_path = request.form['image_path']
    filter_item = request.form['filter_item']  

    # Perform YOLO detection
    processed_image,class_name = yolo_detection(image_path,filter_item)
    if class_name == []:
        relevancy_ = "NOT RELEVANT"
    else:
        relevancy_ = "RELEVANT"
    
    # Convert processed image to base64
    _, buffer = cv2.imencode('.jpg', processed_image)
    encoded_image = base64.b64encode(buffer).decode('utf-8')

    return jsonify({'image': encoded_image,'Relevancy Status':relevancy_})

if __name__ == '__main__':
    app.run(debug=True)
