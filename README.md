# FaceFacts AI: Predicting Age and Gender from Images

FaceFacts AI is a web application that uses deep learning to predict a person's age and gender from facial images. The application then provides personalized fitness recommendations based on these predictions.

## Project Overview

This project combines computer vision and natural language processing to create a personalized fitness recommendation system:

1. **Age and Gender Detection**: Using a convolutional neural network trained on the UTKFace dataset
2. **Personalized Recommendations**: Generating customized workout and diet plans using the Groq API
3. **Interactive Web Interface**: A user-friendly Flask web application with both image upload and live camera analysis

## Dataset

The model was trained on the UTKFace dataset, which contains over 20,000 face images with annotations of age, gender, and ethnicity. The dataset has a wide age range from 0 to 116 years old.

- **UTKFace Dataset**: [Download from Kaggle](https://www.kaggle.com/datasets/jangedoo/utkface-new)
- **Pre-processed Data Files**: [Download from Google Drive](https://drive.google.com/drive/folders/1tgxsX_6WC35vsgdl8rdzpFAcL9U2XYgP?usp=sharing)

The Google Drive link contains the following pre-processed NumPy files:
- `UTKFaceage.npy`: Age labels
- `UTKFacegender.npy`: Gender labels
- `UTKFaceimage.npy`: Processed face images

## Model Architecture

The model uses a multi-output convolutional neural network architecture:
- Input: 48x48x3 RGB images
- Shared convolutional layers for feature extraction
- Two separate output branches:
  - Gender classification (binary)
  - Age regression (continuous value)

## Implementation Details

### Project Structure

```
FaceFacts-AI/
├── Model/
│   └── Age&SexDetection2.ipynb  (Model training notebook)
├── Web App/
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css
│   │   ├── images/
│   │   │   ├── fitness-hero.jpg
│   │   │   └── camera-off.jpg
│   │   ├── Age_sex_detection.h5  (Trained model)
│   │   └── haarcascade_frontalface_default.xml  (Face detection)
│   ├── templates/
│   │   ├── index.html
│   │   ├── prediction_page.html
│   │   ├── live_prediction.html
│   │   └── not_found.html
│   ├── app.py  (Flask application)
│   └── requirements.txt
└── README.md
```

### Technologies Used

- **Backend**: Python, Flask
- **Frontend**: HTML, CSS, JavaScript, Bootstrap 5
- **Machine Learning**: TensorFlow, OpenCV
- **API Integration**: Groq API for generating fitness recommendations

## Installation and Setup

1. **Clone the repository**:
   ```
   git clone https://github.com/yourusername/FaceFacts-AI.git
   cd FaceFacts-AI
   ```

2. **Set up a virtual environment**:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```
   cd "Web App"
   pip install -r requirements.txt
   ```

4. **Download required files**:
   - Download the trained model (`Age_sex_detection.h5`) from the Google Drive link
   - Download the face detection cascade file (`haarcascade_frontalface_default.xml`) from OpenCV's GitHub repository
   - Place both files in the `Web App/static` directory

5. **Set up Groq API key**:
   - Get an API key from [Groq](https://console.groq.com/)
   - Replace the placeholder API key in `app.py` with your actual key

6. **Run the application**:
   ```
   python app.py
   ```

7. **Access the web interface**:
   - Open your browser and navigate to `http://localhost:8000`

## Features

### 1. Image Upload Analysis
- Upload a photo to detect age and gender
- Receive personalized fitness and diet recommendations

### 2. Live Camera Analysis
- Real-time age and gender detection using your webcam
- Interactive and user-friendly interface

### 3. Personalized Recommendations
- Age and gender-specific workout routines
- Customized diet plans
- Fitness tips tailored to the user's profile

## Model Training

To train the model yourself:
1. Download the UTKFace dataset
2. Run the `Age&SexDetection2.ipynb` notebook in Google Colab or locally
3. Save the trained model as `Age_sex_detection.h5`

## Future Improvements

- Add ethnicity prediction as a third output
- Implement user accounts to track progress
- Expand recommendation system with more detailed fitness plans
- Add mobile responsiveness for better experience on smartphones

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- UTKFace dataset for providing the training data
- OpenCV for face detection capabilities
- Groq for the AI-powered recommendation engine