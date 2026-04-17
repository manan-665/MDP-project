from flask import Flask, render_template, request, send_from_directory
import cv2
import numpy as np
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def analyze_for_fire(image_path):
    img = cv2.imread(image_path)
    if img is None: return False, None
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    lower_red1 = np.array([0, 150, 100]); upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 150, 100]); upper_red2 = np.array([180, 255, 255])
    lower_core = np.array([20, 100, 200]); upper_core = np.array([35, 255, 255])

    mask = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1), 
             cv2.bitwise_or(cv2.inRange(hsv, lower_red2, upper_red2),
             cv2.inRange(hsv, lower_core, upper_core)))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9,9), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    fire_found = False
    for c in contours:
        area = cv2.contourArea(c)
        
        if area > 40: 
            fire_found = True
            x, y, w, h = cv2.boundingRect(c)
            
            if area > 1000:
                # LARGE FIRE: Solid Red, Thick Border
                color = (50, 50, 255) 
                thickness = 3
            elif area > 250:
                # MEDIUM SPREAD: Bright Orange
                color = (0, 165, 255) 
                thickness = 2
            else:
                # SMALL/SUBTLE SPOT: Yellow
                color = (0, 255, 255) 
                thickness = 1
            
            cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)

    res_name = "result_" + os.path.basename(image_path)
    cv2.imwrite(os.path.join(UPLOAD_FOLDER, res_name), img)
    return fire_found, res_name

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename != '':
            path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(path)
            fire, image = analyze_for_fire(path)
            return render_template('index.html', image=image, original=file.filename, fire=fire)
    return render_template('index.html', image=None, original=None, fire=False)

@app.route('/uploads/<filename>')
def display_image(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)