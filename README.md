#Face Recognition and Attendance Tracking System

This project is a real-time face recognition and attendance tracking system designed to detect faces, recognize employees, and mark attendance using a YOLOv8-based face detection model. Additionally, it tracks employee presence at specified intervals to provide detailed insights.

Features

	•	Real-time face detection using YOLOv8.
	•	Automatic attendance marking based on employee face recognition.
	•	Interval-based tracking of employee presence.
	•	Stores attendance and interval tracking data in CSV files.
	•	Thread-safe implementation to handle simultaneous attendance marking.
	•	Configurable parameters for attendance cooldown and interval duration.

Requirements

	•	Python 3.8 or later
	•	OpenCV
	•	NumPy
	•	face_recognition
	•	YOLO model from Ultralytics
	•	SciPy
	•	Concurrent futures (for threading)

Project Structure

Face-Recognition/
│
├── LICENSE                     # License file
├── README.md                   # Project documentation
├── yolov8n-face.pt             # YOLOv8 model file for face detection
├── empdata/                    # Folder containing employee images for encoding
├── encodings.pkl               # Pickled file for storing face encodings
├── Attendance.csv              # CSV file for attendance logs
├── IntervalTracking.csv        # CSV file for interval presence tracking
└── face-recognition-script.py  # Main script for face recognition and tracking

Setup Instructions

	1.	Clone the repository:

git clone <repository_url>
cd Face-Recognition


	2.	Install dependencies:
Install the required Python packages:

pip install -r requirements.txt


	3.	Prepare employee images:
	•	Place images of employees in subfolders under empdata/, with each folder named after the respective employee.
	•	Ensure images are clear and in formats like .png, .jpg, or .jpeg.
	4.	Download YOLOv8 model:
Place the YOLOv8 face detection model (yolov8n-face.pt) in the root directory.

Usage

Running the Script

	1.	Start Attendance and Interval Tracking:

python face-recognition-script.py


	2.	Real-time Attendance:
	•	The script will load the YOLOv8 model, detect faces in the video, and mark attendance when a recognized employee is detected.
	•	Employee presence is also tracked every specified interval, marking them as present or absent in IntervalTracking.csv.
	3.	End Process:
	•	Press q to terminate the video processing window.

Viewing Logs

Attendance and interval logs are stored in:

	•	Attendance.csv: Records each attendance marking with a timestamp.
	•	IntervalTracking.csv: Logs interval-based presence status for each employee.

Customization

Modify the following constants in face-recognition-script.py to suit your needs:

	•	YOLO_CONFIDENCE_THRESHOLD: Confidence level for face detection.
	•	ATTENDANCE_COOLDOWN: Minimum duration between consecutive attendance markings.
	•	INTERVAL_DURATION: Frequency of presence check (default is 1 minute).

Troubleshooting

	•	No Face Detected:
Ensure the model file (yolov8n-face.pt) is correctly placed and configured.
	•	Encoding Issues:
If encodings are not loaded properly, delete encodings.pkl and rerun the script to regenerate encodings.
	•	Performance Lag:
Lower the resolution of video input or reduce YOLO_CONFIDENCE_THRESHOLD for faster processing.

Acknowledgments

This project uses Ultralytics’ YOLO for face detection and face_recognition for encoding and recognition.

