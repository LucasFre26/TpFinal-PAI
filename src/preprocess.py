import os
import cv2
import numpy as np
from tqdm import tqdm

def load_and_preprocess_images(directory, img_size=(224, 224)):
    images = []
    labels = []
    class_names = ['NORMAL', 'PNEUMONIA']
    
    for class_name in class_names:
        class_dir = os.path.join(directory, class_name)
        if not os.path.exists(class_dir):
            continue
            
        for img_name in tqdm(os.listdir(class_dir), desc=f'Processing {class_name}'):
            img_path = os.path.join(class_dir, img_name)
            
            try:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                
                # Resize com padding para quadrado
                h, w = img.shape
                if h != w:
                    size = max(h, w)
                    pad_h = (size - h) // 2
                    pad_w = (size - w) // 2
                    img = np.pad(img, ((pad_h, size-h-pad_h), (pad_w, size-w-pad_w)), 
                                 mode='constant', constant_values=0)
                
                img = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
                
                # CLAHE
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                img = clahe.apply(img)
                
                # Desfoque Gaussiano
                img = cv2.GaussianBlur(img, (3,3), 0)
                
                # Canny Edge Detection
                img = cv2.Canny((img * 1).astype(np.uint8), 50, 150)
                
                # Normalização e expansão do canal
                img = img.astype(np.float32) / 255.0
                img = np.expand_dims(img, axis=-1)
                
                images.append(img)
                labels.append(class_names.index(class_name))
                
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
                continue
    
    images = np.array(images)
    labels = np.array(labels)
    
    return images, labels, class_names
