import cv2
import numpy as np
import sys
import importlib.util
import os

# Check if TensorFlow is installed
if importlib.util.find_spec("tensorflow") is None:
    print("Error: TensorFlow is not installed. Please install it using: pip install tensorflow")
    sys.exit(1)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

def create_subfolders():
    base_dir = 'dataset/'
    subfolders = ['train/happy', 'train/sad', 'train/angry', 'train/surprise', 'train/neutral',
                  'validation/happy', 'validation/sad', 'validation/angry', 'validation/surprise', 'validation/neutral']
    for folder in subfolders:
        path = os.path.join(base_dir, folder)
        os.makedirs(path, exist_ok=True)

def check_dataset():
    dataset_path = 'dataset/train'
    if not os.path.exists(dataset_path) or not any(os.listdir(dataset_path)):
        raise FileNotFoundError("Error: Dataset directory is empty or not found. Please add images to 'dataset/train/'")

def prepare_data():
    create_subfolders()
    check_dataset()
    
    data_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    train_data = data_gen.flow_from_directory('dataset/train', target_size=(48, 48), batch_size=32, class_mode='categorical')
    val_data = data_gen.flow_from_directory('dataset/validation', target_size=(48, 48), batch_size=32, class_mode='categorical')
    
    if train_data.samples == 0:
        raise ValueError("Error: No images found in training dataset.")
    
    print("Classes detected:", train_data.class_indices)
    return train_data, val_data

def build_model(num_classes):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(48,48,3)),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')  # Dynamically set number of classes
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, train_data, val_data):
    history = model.fit(train_data, validation_data=val_data, epochs=10)
    model.save('emotion_model.h5')
    return history

def plot_history(history):
    plt.plot(history.history['accuracy'], label='train accuracy')
    plt.plot(history.history['val_accuracy'], label='val accuracy')
    plt.legend()
    plt.show()

def detect_emotion(train_data):
    if not os.path.exists('emotion_model.h5'):
        raise FileNotFoundError("Error: Model file 'emotion_model.h5' not found. Please train the model first.")
    
    model = tf.keras.models.load_model('emotion_model.h5')
    emotion_labels = list(train_data.class_indices.keys())  # Dynamically get emotion labels
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Error: Could not access the webcam.")
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x,y,w,h) in faces:
            face = frame[y:y+h, x:x+w]
            face = cv2.resize(face, (48,48))
            face = np.expand_dims(face, axis=0)
            face = np.array(face, dtype=np.float32) / 255.0
            pred = model.predict(face)
            emotion = emotion_labels[np.argmax(pred)]
            cv2.putText(frame, emotion, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
        
        cv2.imshow('Facial Expression Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Execution Steps
if __name__ == '__main__':
    try:
        train_data, val_data = prepare_data()
        num_classes = train_data.num_classes  # Fetch number of classes dynamically
        model = build_model(num_classes)  # Pass num_classes to model
        history = train_model(model, train_data, val_data)
        plot_history(history)
        detect_emotion(train_data)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)
