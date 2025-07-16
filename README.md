# Student-attentive-analysis
Real-time emotion detection using CNN (PyTorch) and OpenCV. Classifies 7 emotions from webcam input, detects attentiveness, logs time complexity, and stores results with timestamps in a CSV. Trained on FER-2013. Includes live face detection, logging, and CSV export.
🧠 Real-Time Emotion Detection using CNN and OpenCV
This project implements a real-time emotion recognition system using a Convolutional Neural Network (CNN) in PyTorch, integrated with OpenCV for live webcam input.

🚀 Features
🎥 Real-time face detection using Haar cascades

😃 Emotion classification into 7 categories:
Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral

✅ Attentiveness detection based on emotion

📊 Time complexity logging per detection

📁 CSV logging of:

Timestamp (with milliseconds)

Emotion and attentiveness status

Confidence percentage

Time complexity and detection time

📌 Technologies Used
Python

PyTorch

OpenCV

Torchvision

Haar Cascade Classifier

CSV Logging

🧪 Dataset
Trained on the FER-2013 dataset (grayscale 48x48 emotion-labeled images)

📂 Output Example
✅ Terminal Log (printed every 2s):
less
Copy
Edit
[INFO] Detection at 14:31:05
[INFO] Faces detected: 1
[INFO] Time complexity per detection: O(F × D + n²)
[INFO] Elapsed time for detection: 0.0342 seconds
📄 CSV Log (time_complexity_log.csv):
Date	Time	Emotion	Status	Percentage	Time Complexity	Mega Ops	Elapsed Time (s)
2025-07-16	14:31:05.423	Happy	Attentive	87%	O(F × D + n²)	17.31	0.0342

💡 How to Run
Clone the repo and place your FER-2013 training dataset in the correct path.

Run the script:

bash
Copy
Edit
python demo.py
Press q to exit the real-time webcam detection.

📎 Future Improvements
Add confidence scores from the model

Visual plots from the log file

Real-time dashboard using Streamlit or Dash




























## 🚀 How to Run the Project

Follow these steps to run the real-time emotion detection system:

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/emotion-detection.git
cd emotion-detection
2️⃣ Install Requirements
Ensure you have Python 3.7+ installed. Then install the dependencies:

bash
Copy
Edit
pip install -r requirements.txt
3️⃣ Prepare the Dataset
Download the FER2013 dataset and place the train folder inside:

swift
Copy
Edit
C:/Users/nihar/OneDrive/Desktop/mini/
4️⃣ Run the Application
Start the real-time webcam emotion detection:

bash
Copy
Edit
python demo.py
5️⃣ Output
The webcam window will show detected faces with predicted emotion and attention status.

Real-time logs will appear in the terminal.

Detection logs are saved in:

swift
Copy
Edit
C:/Users/nihar/OneDrive/Desktop/mini/time_complexity_log.csv
Press q to quit the webcam window.
