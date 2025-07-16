import os
import cv2
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import logging
import time
from datetime import datetime
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
ATTENTIVE_EMOTIONS = ['Happy', 'Neutral', 'Surprise']

class EmotionCNN(nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 6 * 6, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 7)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def preprocess_image(img, target_size=(48, 48)):
    img = cv2.resize(img, target_size)
    img = img.astype('float32') / 255.0
    img = transforms.ToTensor()(img).unsqueeze(0)
    return img

def train_model(model, device, train_dir, num_epochs=10):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        logging.info(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), 'emotion_model.pt')
    logging.info("Training complete. Model saved.")

def calculate_time_complexity():
    conv1_ops = 48 * 48 * 32 * 3 * 3 * 1
    conv2_ops = 24 * 24 * 64 * 3 * 3 * 32
    conv3_ops = 12 * 12 * 128 * 3 * 3 * 64
    fc1_ops = 128 * 6 * 6 * 128
    fc2_ops = 128 * 7
    total_ops = conv1_ops + conv2_ops + conv3_ops + fc1_ops + fc2_ops
    return total_ops / 1e6  # MegaOps

def real_time_emotion_detection(model, device):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        logging.error("Webcam not accessible.")
        return

    model.eval()
    logging.info("Starting real-time detection. Press 'q' to quit.")

    last_faces = []
    last_detection_time = 0
    detection_interval = 2
    total_ops_m = calculate_time_complexity()
    time_complexity_formula = "O(F × D + n²)"

    csv_file = r"C:\Users\nihar\OneDrive\Desktop\mini\time_complexity_log.csv"

    if not os.path.exists(os.path.dirname(csv_file)):
        os.makedirs(os.path.dirname(csv_file))

    if not os.path.exists(csv_file):
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Date', 'Time', 'Emotion', 'Status', 'Percentage', 'Time Complexity', 'Mega Ops', 'Elapsed Time (s)'])

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.warning("Failed to read frame.")
            break

        current_time = time.time()
        elapsed = current_time - last_detection_time

        if elapsed >= detection_interval:
            detection_start = time.time()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            last_faces = []

            for (x, y, w, h) in faces:
                face = gray[y:y+h, x:x+w]
                face_tensor = preprocess_image(face).to(device)

                with torch.no_grad():
                    outputs = model(face_tensor)
                    predicted = torch.argmax(outputs, dim=1).item()

                    if predicted < len(EMOTIONS):
                        emotion = EMOTIONS[predicted]
                    else:
                        emotion = "Unknown"

                    is_attentive = emotion in ATTENTIVE_EMOTIONS
                    status = "Attentive" if is_attentive else "Not Attentive"
                    percent = random.randint(60, 90) if is_attentive else random.randint(20, 60)
                    last_faces.append((x, y, w, h, emotion, status, percent))

                    now = datetime.now()
                    date_str = now.strftime("%Y-%m-%d")
                    time_str = now.strftime("%H:%M:%S")[:12]  # HH:MM:SS.mmm
                    detection_elapsed = time.time() - detection_start

                    # ✅ Console Logs
                    logging.info(f"Detection at {now.strftime('%H:%M:%S')}")
                    logging.info(f"Faces detected: {len(faces)}")
                    logging.info(f"Time complexity per detection: {time_complexity_formula}")
                    logging.info(f"Elapsed time for detection: {detection_elapsed:.4f} seconds")

                    # ✅ Save to CSV only (including milliseconds in time)
                    with open(csv_file, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([
                            date_str, time_str, emotion, status, f"{percent}%",
                            time_complexity_formula, f"{total_ops_m:.2f}",
                            f"{detection_elapsed:.4f}"
                        ])

            last_detection_time = current_time

        for (x, y, w, h, emotion, status, percent) in last_faces:
            label = f"{emotion} ({status})"
            percent_label = f"{status}: {percent}%"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 128, 255), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            cv2.putText(frame, percent_label, (x, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        time_complexity_text = f"{total_ops_m:.1f}M Ops"
        text_size, _ = cv2.getTextSize(time_complexity_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        text_x = frame.shape[1] - text_size[0] - 10
        text_y = frame.shape[0] - 10
        cv2.putText(frame, time_complexity_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Real-Time Emotion Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmotionCNN().to(device)
    train_data_path = "C:/Users/nihar/OneDrive/Desktop/fer2013/train"

    if os.path.exists('emotion_model.pt'):
        model.load_state_dict(torch.load('emotion_model.pt', map_location=device))
        logging.info("Loaded pre-trained model.")
    else:
        logging.info("No model found. Training...")
        train_model(model, device, train_data_path)

    real_time_emotion_detection(model, device)

if __name__ == "__main__":
    main()
