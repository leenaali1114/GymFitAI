from flask import Flask, render_template, request, redirect, url_for, session, Response, jsonify
import os
import numpy as np
import cv2
import tensorflow as tf
import requests
import json
import time
import sys
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import markdown

sys.stdout.reconfigure(encoding='utf-8')

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
app.secret_key = os.urandom(24)

UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# loading the model
# model is saved in static folder
model = tf.keras.models.load_model('static/Age_sex_detection.h5')

# gender mapping
# if gender prediction = 0, the predition is MALE
# else FEMAL
gender_dict = {0: 'Male', 1: 'Female'}

# Groq API configuration
GROQ_API_KEY = "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"  # Replace with your actual Groq API key
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Function to extract face from uploaded image
# The dataset on which model was trained contained only face images
# This function finds the face area using
# a haar cascade designed by OpenCV to detect the frontal face
def extract_face():
    
    img = cv2.imread('static/uploaded_img.jpg') # reading uploaded image
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # converted to grayscale
    
    # OpenCV's CascadeClassifier to load a pre-trained Haar cascade for detecting frontal faces
    haar_cascade = cv2.CascadeClassifier('static/haarcascade_frontalface_default.xml')
    faces_rect = haar_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=9)
    extracted_faces = []

    if len(faces_rect) == 0:
        cv2.imwrite('static/extracted.jpg', img) # if no coordinates are detected, its only face image
    else:
        extracted_faces = []
        for (x, y, w, h) in faces_rect:
            face = img[y:y+h, x:x+w]
            extracted_faces.append(face)
        concatenated_faces = cv2.hconcat(extracted_faces)
        cv2.imwrite('static/extracted.jpg', concatenated_faces) # face extracted image - saved as extracted.jpg

    if len(faces_rect) != 0:
        for (x, y, w, h) in faces_rect:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)
        cv2.imwrite('static/uploaded_img.jpg', img) # original uploaded image where face is marked

    if extracted_faces:
        # Save the first detected face
        cv2.imwrite('static/extracted.jpg', extracted_faces[0])
        
        # Draw rectangles on original image
        for (x, y, w, h) in faces_rect:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)
        cv2.imwrite('static/uploaded_img.jpg', img)
        
        return True
    return False

# Function to extract features from the image
def extract_features(image_path):
    img = cv2.imread(image_path)  # Read image in color
    img = cv2.resize(img, (48, 48))  # Resize to 48x48 (what the model expects)
    img = np.array(img)  # Convert image to numpy array
    return img

# Function to predict gender and age
def predict_result(image_path='static/extracted.jpg'):
    X = extract_features(image_path)
    X = X/255.0  # Normalize
    # Reshape to match the model's expected input shape (None, 48, 48, 3)
    pred = model.predict(np.array([X]))
    pred_gender = gender_dict[round(pred[0][0][0])]
    pred_age = round(pred[1][0][0])
    return pred_gender, pred_age

# Function to get recommendations from Groq API
def get_recommendations(age, gender):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    prompt = f"""
    As a fitness expert, provide personalized workout and diet recommendations for a {age}-year-old {gender.lower()}.
    
    Format your response in clean, structured markdown tables with the following sections:
    
    1. A weekly workout plan with specific exercises (5 days)
    2. A daily diet plan with meal suggestions
    3. 3 fitness tips tailored to their age and gender
    
    Make sure all tables have clear headers and are well-formatted for easy reading.
    """
    
    data = {
        "model": "llama3-8b-8192",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 1024
    }
    
    try:
        response = requests.post(GROQ_API_URL, headers=headers, data=json.dumps(data))
        if response.status_code == 200:
            markdown_content = response.json()["choices"][0]["message"]["content"]
            # Convert markdown to HTML
            html_content = markdown.markdown(markdown_content, extensions=['tables'])
            return html_content
        else:
            return f"<h3>Error getting recommendations: {response.status_code}</h3><p>{response.text}</p>"
    except Exception as e:
        return f"<h3>Error connecting to Groq API</h3><p>{str(e)}</p>"

# Routes
@app.route('/prediction')
def prediction():
    if os.path.exists('static/extracted.jpg'):
        gender, age = predict_result()
        recommendations = get_recommendations(age, gender)
        return render_template('prediction_page.html', age=age, gender=gender, recommendations=recommendations)
    else:
        return render_template('not_found.html')

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST' and 'face_image' in request.files:
        face_image = request.files['face_image']
        if face_image.filename != '':
            image_filename = 'uploaded_img.jpg'
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
            face_image.save(image_path)
            session['image_filename'] = image_filename
            if extract_face():
                return redirect(url_for('prediction'))
            else:
                return render_template('not_found.html')
    return render_template('index.html')

# Live video capture and prediction
camera = None

def generate_frames():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            haar_cascade = cv2.CascadeClassifier('static/haarcascade_frontalface_default.xml')
            faces = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            
            # Process each face
            for (x, y, w, h) in faces:
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Extract and process face for prediction
                face = frame[y:y+h, x:x+w]
                if face.size > 0:
                    # Save face temporarily
                    temp_path = 'static/temp_face.jpg'
                    cv2.imwrite(temp_path, face)
                    
                    # Predict age and gender
                    try:
                        gender, age = predict_result(temp_path)
                        # Display prediction on frame
                        cv2.putText(frame, f"Age: {age}, Gender: {gender}", (x, y-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    except Exception as e:
                        print(f"Prediction error: {e}")
            
            # Convert frame to JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            # Yield frame for streaming
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
            # Add a small delay to reduce CPU usage
            time.sleep(0.1)

@app.route('/live_prediction')
def live_prediction():
    return render_template('live_prediction.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_camera')
def stop_camera():
    global camera
    if camera is not None:
        camera.release()
        camera = None
    return jsonify({"status": "success"})

@app.route('/not_found')
def not_found():
    return render_template('not_found.html')

@app.route('/static/images/<filename>')
def serve_placeholder_image(filename):
    # Check if the requested image exists
    image_path = os.path.join('static/images', filename)
    if os.path.exists(image_path):
        return redirect(url_for('static', filename=f'images/{filename}'))
    
    # If not, generate a placeholder
    from PIL import Image, ImageDraw, ImageFont
    from io import BytesIO
    
    # Create a colored background
    if filename == 'fitness-hero.jpg':
        color = (13, 110, 253)  # Blue
        text = "Fitness Hero"
        size = (800, 400)
    else:
        color = (108, 117, 125)  # Gray
        text = "Camera Off"
        size = (400, 300)
    
    img = Image.new('RGB', size, color)
    draw = ImageDraw.Draw(img)
    
    # Add text
    text_color = (255, 255, 255)
    text_position = (size[0]//4, size[1]//2)
    draw.text(text_position, text, fill=text_color)
    
    # Save to BytesIO
    img_io = BytesIO()
    img.save(img_io, 'JPEG')
    img_io.seek(0)
    
    return Response(img_io.getvalue(), mimetype='image/jpeg')

@app.context_processor
def inject_now():
    return {'now': datetime.now}

if __name__ == "__main__":
    app.run(debug=True, port=8000)
