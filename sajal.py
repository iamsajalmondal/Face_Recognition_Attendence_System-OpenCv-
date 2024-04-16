import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

video_capture = cv2.VideoCapture(0)

image_paths = [
    "images/adani.jpg",
    "images/bill.jpg",
    "images/elon.jpg",
    "images/jobs.jpg",
    "images/mukesh.jpg",
    "images/srk.jpg",
    "images/sajal.jpg"
]

known_face_names = [
    'adani',
    'bill',
    'elon',
    'jobs',
    'mukesh',
    'srk',
    'sajal'
]
known_designations = [
    'CEO',
    'Co-founder',
    'Manager',
    'CTO',
    'Chairman',
    'Actor',
    'Software Engineer'
]
known_departments = [
    'Management',
    'Finance',
    'Marketing',
    'HR',
    'Management',
    'Entertainment',
    'Technical'
]

known_face_encodings = []

for image_path, name, designation, department in zip(image_paths, known_face_names, known_designations, known_departments):
    image = face_recognition.load_image_file(image_path)
    encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(encoding)

employees = known_face_names.copy()
recognized_names = []

while True:
    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    file_name = f'{current_date}.csv'

    with open(file_name, 'a', newline='') as f:
        lnwriter = csv.writer(f)
        if f.tell() == 0:
            lnwriter.writerow(['Name', 'Designation', 'Department', 'Time'])

        _, frame = video_capture.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)

        if len(face_locations) > 0:
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            print("Detected faces:", len(face_locations))  # Print number of detected faces

            for face_encoding, face_location in zip(face_encodings, face_locations):
                top, right, bottom, left = face_location
                face_image = rgb_small_frame[top:bottom, left:right]

                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = ""
                designation = ""
                department = ""

                face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distance)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    designation = known_designations[best_match_index]
                    department = known_departments[best_match_index]

                    print("Name:", name)
                    print("Designation:", designation)
                    print("Department:", department)

                    if name in employees and name not in recognized_names:
                        recognized_names.append(name)  # Add name to recognized list
                        current_time = datetime.now().strftime("%H:%M:%S")
                        lnwriter.writerow([name, designation, department, current_time])
        else:
            print("No faces detected")

        cv2.imshow("Attendance System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

video_capture.release()
cv2.destroyAllWindows()
