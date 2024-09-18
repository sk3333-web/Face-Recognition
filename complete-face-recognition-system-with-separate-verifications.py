import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime, timedelta
import pickle
from collections import defaultdict
from scipy.spatial.distance import cosine
import logging
from concurrent.futures import ThreadPoolExecutor
import threading
from ultralytics import YOLO
import csv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
BASE_PATH = r" empdata"
ENCODINGS_FILE = r"encodings.pkl"
ATTENDANCE_FILE = 'Attendance.csv'
INTERVAL_TRACKING_FILE = 'IntervalTracking.csv'
ATTENDANCE_VERIFICATION_FRAMES = 10  # Changed to 10 for attendance marking
INTERVAL_VERIFICATION_FRAMES = 5  # New constant for interval tracking
SIMILARITY_THRESHOLD = 0.75
ATTENDANCE_COOLDOWN = timedelta(hours=10)
INTERVAL_DURATION = timedelta(minutes=1)
VIDEO_PATH = r" "
YOLO_MODEL_PATH = r"yolov8n-face.pt"
YOLO_CONFIDENCE_THRESHOLD = 0.3

# Initialize variables
employee_encodings = {}
classNames = []
last_attendance = defaultdict(lambda: datetime.min)
attendance_frame_count = defaultdict(int)
interval_frame_count = defaultdict(int)
last_interval_check = datetime.now()
interval_employee = None
face_detected_in_interval = False

# Thread-safe attendance marking
attendance_lock = threading.Lock()

# Load YOLOv8n face detection model
yolo_model = YOLO(YOLO_MODEL_PATH)

def load_encodings():
    global employee_encodings, classNames
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, 'rb') as f:
            data = pickle.load(f)
            if isinstance(data, tuple) and len(data) == 2:
                old_encodings, classNames = data
                # Convert old structure to new structure
                employee_encodings = {}
                if isinstance(old_encodings, list):
                    for name, encoding in zip(classNames, old_encodings):
                        employee_encodings[name] = {'encodings': [encoding], 'average': encoding}
                elif isinstance(old_encodings, dict):
                    employee_encodings = old_encodings
                logging.info('Encodings Loaded and Converted')
            else:
                logging.warning('Invalid encoding file format. Starting fresh.')
                employee_encodings = {}
                classNames = []
    else:
        logging.info('No existing encodings found. Starting fresh.')
        employee_encodings = {}
        classNames = []

def process_employee_images():
    global employee_encodings, classNames
    for folder_name in os.listdir(BASE_PATH):
        folder_path = os.path.join(BASE_PATH, folder_name)
        if os.path.isdir(folder_path):
            if folder_name not in employee_encodings:
                employee_encodings[folder_name] = {'encodings': [], 'average': None}
            for file_name in os.listdir(folder_path):
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(folder_path, file_name)
                    process_image(img_path, folder_name)
            
            if employee_encodings[folder_name]['encodings']:
                employee_encodings[folder_name]['average'] = np.mean(employee_encodings[folder_name]['encodings'], axis=0)
                if folder_name not in classNames:
                    classNames.append(folder_name)
            else:
                employee_encodings.pop(folder_name, None)
                if folder_name in classNames:
                    classNames.remove(folder_name)
    
    save_encodings()

def process_image(img_path, folder_name):
    try:
        curImg = cv2.imread(img_path)
        if curImg is not None:
            curImg = cv2.cvtColor(curImg, cv2.COLOR_BGR2RGB)
            results = yolo_model(curImg, conf=YOLO_CONFIDENCE_THRESHOLD)
            if len(results[0].boxes) > 0:
                box = results[0].boxes[0].xyxy[0].cpu().numpy().astype(int)
                face_image = curImg[box[1]:box[3], box[0]:box[2]]
                encodes = face_recognition.face_encodings(face_image)
                if encodes:
                    employee_encodings[folder_name]['encodings'].append(encodes[0])
                else:
                    logging.warning(f"No face encoding generated for {img_path}")
            else:
                logging.warning(f"No face detected in {img_path}")
        else:
            logging.error(f"Failed to load image: {img_path}")
    except Exception as e:
        logging.error(f"Error processing {img_path}: {str(e)}")

def save_encodings():
    with open(ENCODINGS_FILE, 'wb') as f:
        pickle.dump((employee_encodings, classNames), f)
    logging.info('Encodings saved')

def initialize_interval_tracking():
    if not os.path.exists(INTERVAL_TRACKING_FILE):
        with open(INTERVAL_TRACKING_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'Employee', 'Status'])

def update_interval_tracking():
    global last_interval_check, interval_employee, face_detected_in_interval
    current_time = datetime.now()
    if current_time - last_interval_check >= INTERVAL_DURATION:
        status = 'Present' if face_detected_in_interval else 'Absent'
        with open(INTERVAL_TRACKING_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            row = [current_time.strftime('%Y-%m-%d %H:%M:%S'), interval_employee or 'None', status]
            writer.writerow(row)
        
        # Reset for next interval
        interval_employee = None
        face_detected_in_interval = False
        last_interval_check = current_time

def mark_attendance(name):
    global interval_employee
    now = datetime.now()
    with attendance_lock:
        if now - last_attendance[name] >= ATTENDANCE_COOLDOWN:
            with open(ATTENDANCE_FILE, 'a+', encoding='utf-8') as f:
                dtString = now.strftime('%Y-%m-%d %H:%M:%S')
                f.write(f'{name},{dtString}\n')
            logging.info(f'Attendance marked for {name} at {dtString}')
            last_attendance[name] = now
            
            # Update interval tracking with employee name
            if interval_employee is None:
                interval_employee = name
        else:
            logging.info(f'Attendance for {name} already marked recently.')

def process_frame(img):
    global face_detected_in_interval, interval_employee
    results = yolo_model(img, conf=YOLO_CONFIDENCE_THRESHOLD)
    
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy().astype(int)
        for box in boxes:
            face_image = img[box[1]:box[3], box[0]:box[2]]
            face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            encodeFace = face_recognition.face_encodings(face_image_rgb)
            
            if encodeFace:
                best_match = None
                best_similarity = -1

                for name, data in employee_encodings.items():
                    avg_encoding = data['average']
                    if avg_encoding is not None:
                        similarity = 1 - cosine(encodeFace[0], avg_encoding)
                        
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match = name

                if best_similarity > SIMILARITY_THRESHOLD:
                    name = best_match.upper()
                    attendance_frame_count[name] += 1
                    interval_frame_count[name] += 1
                    
                    # Check for attendance marking
                    if attendance_frame_count[name] >= ATTENDANCE_VERIFICATION_FRAMES:
                        mark_attendance(name)
                        attendance_frame_count[name] = 0
                    
                    # Check for interval tracking
                    if interval_frame_count[name] >= INTERVAL_VERIFICATION_FRAMES:
                        face_detected_in_interval = True
                        if interval_employee is None:
                            interval_employee = name
                    
                    color = (0, 255, 0)  # Green for recognized faces
                else:
                    name = 'Unknown'
                    attendance_frame_count[name] = 0
                    interval_frame_count[name] = 0
                    color = (0, 0, 255)  # Red for unknown faces
            else:
                name = 'Unknown'
                color = (0, 0, 255)  # Red for unknown faces
            
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.rectangle(img, (box[0], box[3]-35), (box[2], box[3]), color, cv2.FILLED)
            cv2.putText(img, name, (box[0]+6, box[3]-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    update_interval_tracking()
    return img

def process_video():
    load_encodings()
    process_employee_images()
    initialize_interval_tracking()

    cap = cv2.VideoCapture(VIDEO_PATH)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    frame_number = 0

    with ThreadPoolExecutor(max_workers=4) as executor:
        while True:
            success, img = cap.read()
            if not success:
                logging.info("End of video reached")
                break

            frame_number += 1
            current_time = datetime.now()

            try:
                processed_img = executor.submit(process_frame, img).result()
                
                # Display progress
                progress = (frame_number / total_frames) * 100
                logging.info(f"Processing: {progress:.2f}% complete")

                cv2.imshow('Video', processed_img)
            except Exception as e:
                logging.error(f"Error processing frame {frame_number}: {str(e)}")

            # Control playback speed (adjust as needed)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_video()
