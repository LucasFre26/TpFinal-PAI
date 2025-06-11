# predict.py (updated)
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

class PneumoniaPredictor:
    def __init__(self, model_path):
        self.model = load_model(model_path)
        self.img_size = (224, 224)
        self.class_names = ['NORMAL', 'PNEUMONIA']
        
    def preprocess_image(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Could not load image")
        
        # Resize with padding to make square
        h, w = img.shape
        if h != w:
            size = max(h, w)
            pad_h = (size - h) // 2
            pad_w = (size - w) // 2
            img = np.pad(img, ((pad_h, size-h-pad_h), (pad_w, size-w-pad_w)), 
                         mode='constant', constant_values=0)
        
        img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_AREA)
        
        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img = clahe.apply(img)
        
        # Gaussian Blur
        img = cv2.GaussianBlur(img, (3,3), 0)
        
        # Canny Edge Detection
        img = cv2.Canny((img * 1).astype(np.uint8), 50, 150)
        
        # Normalization and channel expansion
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=-1)
        
        return img
    
    def predict(self, img_path):
        img = self.preprocess_image(img_path)
        prediction = self.model.predict(img[np.newaxis, ...])[0][0]
        label = self.class_names[1] if prediction > 0.5 else self.class_names[0]
        confidence = prediction * 100 if prediction > 0.5 else (1 - prediction) * 100
        return label, confidence

# Initialize predictor once
predictor = PneumoniaPredictor(os.path.join(os.path.dirname(__file__), '..', 'models', 'pneumonia_model.keras'))

def predict_pneumonia(img_path):
    return predictor.predict(img_path)