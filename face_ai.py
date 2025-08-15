# Face Recognition Project
# Requirements: pip install opencv-python face-recognition pillow numpy

import mahotas as mh # type: ignore
import face_recognition
import numpy as np # type: ignore
import os
import pickle
from PIL import Image, ImageDraw, ImageFont # type: ignore
import json
from datetime import datetime

class FaceRecognitionSystem:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.process_this_frame = True
        
    def load_known_faces(self, faces_dir="known_faces"):
        """Load known faces from a directory"""
        if not os.path.exists(faces_dir):
            os.makedirs(faces_dir)
            print(f"Created directory: {faces_dir}")
            print("Please add face images to this directory and run again.")
            return
        
        print("Loading known faces...")
        for filename in os.listdir(faces_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Get person name from filename (without extension)
                name = os.path.splitext(filename)[0]
                
                # Load image
                image_path = os.path.join(faces_dir, filename)
                image = face_recognition.load_image_file(image_path)
                
                # Get face encoding
                encodings = face_recognition.face_encodings(image)
                
                if encodings:
                    encoding = encodings[0]  # Take the first face found
                    self.known_face_encodings.append(encoding)
                    self.known_face_names.append(name)
                    print(f"Loaded face: {name}")
                else:
                    print(f"No face found in {filename}")
        
        print(f"Loaded {len(self.known_face_names)} known faces")
    
    def save_encodings(self, filename="face_encodings.pkl"):
        """Save face encodings to file"""
        data = {
            'encodings': self.known_face_encodings,
            'names': self.known_face_names
        }
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"Face encodings saved to {filename}")
    
    def load_encodings(self, filename="face_encodings.pkl"):
        """Load face encodings from file"""
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.known_face_encodings = data['encodings']
                self.known_face_names = data['names']
            print(f"Face encodings loaded from {filename}")
            return True
        except FileNotFoundError:
            print(f"Encoding file {filename} not found")
            return False
    
    def recognize_faces_in_image(self, image_path, save_result=True):
        """Recognize faces in a single image"""
        # Load image
        image = face_recognition.load_image_file(image_path)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Find faces
        face_locations = face_recognition.face_locations(rgb_image)
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
        
        # Convert to PIL for drawing
        pil_image = Image.fromarray(rgb_image)
        draw = ImageDraw.Draw(pil_image)
        
        results = []
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Check if face matches known faces
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"
            confidence = 0
            
            # Use the known face with the smallest distance
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
                confidence = 1 - face_distances[best_match_index]
            
            results.append({
                'name': name,
                'confidence': confidence,
                'location': (top, right, bottom, left)
            })
            
            # Draw rectangle and label
            draw.rectangle([left, top, right, bottom], outline=(0, 255, 0), width=3)
            text = f"{name} ({confidence:.2f})"
            draw.text((left, top - 20), text, fill=(0, 255, 0))
        
        # Save result if requested
        if save_result:
            output_path = f"result_{os.path.basename(image_path)}"
            pil_image.save(output_path)
            print(f"Result saved to {output_path}")
        
        return results, pil_image
    
    def start_video_recognition(self, camera_index=0):
        """Start real-time face recognition from camera"""
        video_capture = cv2.VideoCapture(camera_index)
        
        print("Starting video recognition. Press 'q' to quit, 's' to save current frame")
        
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            
            # Resize frame for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Process every other frame for better performance
            if self.process_this_frame:
                self.face_locations = face_recognition.face_locations(rgb_small_frame)
                self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)
                
                self.face_names = []
                for face_encoding in self.face_encodings:
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    name = "Unknown"
                    confidence = 0
                    
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        confidence = 1 - face_distances[best_match_index]
                    
                    self.face_names.append(f"{name} ({confidence:.2f})")
            
            self.process_this_frame = not self.process_this_frame
            
            # Display results
            for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                # Scale back up face locations
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                
                # Draw rectangle
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                
                # Draw label
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
            
            cv2.imshow('Face Recognition', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"capture_{timestamp}.jpg", frame)
                print(f"Frame saved as capture_{timestamp}.jpg")
        
        video_capture.release()
        cv2.destroyAllWindows()
    
    def add_new_face(self, image_path, name):
        """Add a new face to the known faces"""
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        
        if encodings:
            self.known_face_encodings.append(encodings[0])
            self.known_face_names.append(name)
            print(f"Added new face: {name}")
            return True
        else:
            print("No face found in the image")
            return False
    
    def remove_face(self, name):
        """Remove a face from known faces"""
        if name in self.known_face_names:
            index = self.known_face_names.index(name)
            del self.known_face_names[index]
            del self.known_face_encodings[index]
            print(f"Removed face: {name}")
            return True
        else:
            print(f"Face {name} not found")
            return False
    
    def list_known_faces(self):
        """List all known faces"""
        print("Known faces:")
        for i, name in enumerate(self.known_face_names):
            print(f"{i+1}. {name}")
    
    def create_attendance_log(self, image_path):
        """Create attendance log from an image"""
        results, _ = self.recognize_faces_in_image(image_path, save_result=False)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = {
            'timestamp': timestamp,
            'image': image_path,
            'detected_faces': []
        }
        
        for result in results:
            if result['name'] != "Unknown" and result['confidence'] > 0.6:
                log_entry['detected_faces'].append({
                    'name': result['name'],
                    'confidence': result['confidence']
                })
        
        # Save to JSON log file
        log_file = "attendance_log.json"
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                logs = json.load(f)
        else:
            logs = []
        
        logs.append(log_entry)
        
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)
        
        print(f"Attendance logged: {len(log_entry['detected_faces'])} people detected")
        return log_entry


def main():
    # Initialize face recognition system
    fr_system = FaceRecognitionSystem()
    
    print("=== Face Recognition System ===")
    print("1. Load faces from directory")
    print("2. Load saved encodings")
    print("3. Start video recognition")
    print("4. Recognize faces in image")
    print("5. Add new face")
    print("6. Remove face")
    print("7. List known faces")
    print("8. Save encodings")
    print("9. Create attendance log")
    print("0. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice: ").strip()
            
            if choice == '1':
                faces_dir = input("Enter faces directory (press Enter for 'known_faces'): ").strip()
                if not faces_dir:
                    faces_dir = "known_faces"
                fr_system.load_known_faces(faces_dir)
            
            elif choice == '2':
                filename = input("Enter encoding file (press Enter for 'face_encodings.pkl'): ").strip()
                if not filename:
                    filename = "face_encodings.pkl"
                fr_system.load_encodings(filename)
            
            elif choice == '3':
                if not fr_system.known_face_encodings:
                    print("Please load known faces first!")
                    continue
                camera_index = input("Enter camera index (press Enter for 0): ").strip()
                camera_index = int(camera_index) if camera_index else 0
                fr_system.start_video_recognition(camera_index)
            
            elif choice == '4':
                if not fr_system.known_face_encodings:
                    print("Please load known faces first!")
                    continue
                image_path = input("Enter image path: ").strip()
                if os.path.exists(image_path):
                    results, _ = fr_system.recognize_faces_in_image(image_path)
                    print(f"Detected {len(results)} faces:")
                    for result in results:
                        print(f"- {result['name']} (confidence: {result['confidence']:.2f})")
                else:
                    print("Image file not found!")
            
            elif choice == '5':
                image_path = input("Enter image path: ").strip()
                name = input("Enter person name: ").strip()
                if os.path.exists(image_path) and name:
                    fr_system.add_new_face(image_path, name)
                else:
                    print("Invalid input!")
            
            elif choice == '6':
                name = input("Enter name to remove: ").strip()
                if name:
                    fr_system.remove_face(name)
            
            elif choice == '7':
                fr_system.list_known_faces()
            
            elif choice == '8':
                filename = input("Enter filename (press Enter for 'face_encodings.pkl'): ").strip()
                if not filename:
                    filename = "face_encodings.pkl"
                fr_system.save_encodings(filename)
            
            elif choice == '9':
                if not fr_system.known_face_encodings:
                    print("Please load known faces first!")
                    continue
                image_path = input("Enter image path: ").strip()
                if os.path.exists(image_path):
                    fr_system.create_attendance_log(image_path)
                else:
                    print("Image file not found!")
            
            elif choice == '0':
                print("Goodbye!")
                break
            
            else:
                print("Invalid choice!")
        
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()