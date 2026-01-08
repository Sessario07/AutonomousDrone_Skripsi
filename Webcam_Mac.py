import os
import cv2
import numpy as np
import face_recognition
from tkinter import Tk, Label, Button, Frame, StringVar, OptionMenu
from PIL import Image, ImageTk
import mediapipe as mp
import time
import platform


def load_known_faces(directory):
    known_face_encodings = []
    known_face_names = []

    for file_name in os.listdir(directory):
        if file_name.endswith((".jpg", ".png")):
            image_path = os.path.join(directory, file_name)
            image = face_recognition.load_image_file(image_path)
            face_encodings = face_recognition.face_encodings(image)
            if face_encodings:
                known_face_encodings.append(face_encodings[0])
                known_face_names.append(os.path.splitext(file_name)[0])

    return known_face_encodings, known_face_names


# Function to calculate confidence from face distance
def calculate_confidence(face_distance, face_match_threshold=0.6):
    if face_distance > face_match_threshold:
        linear_val = (1.0 - face_distance) / (0.1 - face_match_threshold)
        return max(0.0, min(1.0, linear_val)) * 100
    else:
        linear_val = (1.0 - face_distance) / (face_match_threshold - 0.1)
        return max(0.0, min(1.0, linear_val)) * 100


# Function to perform face recognition on a single frame
def recognize_faces(frame, known_face_encodings, known_face_names):
    # Use smaller resize factor on macOS for better performance
    is_mac = platform.system() == 'Darwin'
    scale_factor = 0.5 if is_mac else 0.25
    
    small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    
    # Use faster model on macOS
    face_locations = face_recognition.face_locations(rgb_small_frame, model='hog')
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    
    face_names = []
    face_confidences = []
    face_distances = []

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
        name = "Unknown"
        confidence = 0.0
        distance = 0.0

        face_distances_current = face_recognition.face_distance(known_face_encodings, face_encoding)

        best_match_index = np.argmin(face_distances_current)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            confidence = calculate_confidence(face_distances_current[best_match_index])
            distance = face_distances_current[best_match_index]

        face_names.append(name)
        face_confidences.append(confidence)
        face_distances.append(distance)

    # Adjust scale factor for coordinates
    scale_multiplier = int(1 / scale_factor)
    return face_locations, face_names, face_confidences, face_distances, scale_multiplier


# Class for controlling the webcam video stream via Tkinter GUI
class WebcamController:
    def __init__(self):
        self.root = Tk()
        self.root.title("Webcam Controller - Tkinter")
        self.root.minsize(800, 600)

        # Detect if running on macOS
        self.is_mac = platform.system() == 'Darwin'
        
        self.input_frame = Frame(self.root)
        
        # Initialize camera with macOS-specific settings
        if self.is_mac:
            # Try different camera backends on macOS
            self.cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        else:
            self.cap = cv2.VideoCapture(0)
        
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Add a small delay to allow camera to initialize on macOS
        if self.is_mac:
            time.sleep(0.5)

        faces_dir = "faces"
        # Create faces directory if it doesn't exist
        if not os.path.exists(faces_dir):
            os.makedirs(faces_dir)
            print(f"Created '{faces_dir}' directory. Please add face images here.")
        
        self.known_face_encodings, self.known_face_names = load_known_faces(faces_dir)

        self.cap_lbl = Label(self.root)
        self.button_frame = Frame(self.root)

        self.demo_button = Button(self.button_frame, text="Demo Button", command=self.demo_function)

        self.face_detection_var = StringVar(self.root)
        self.face_detection_var.set("Disable")
        self.face_detection_menu = OptionMenu(self.button_frame, self.face_detection_var, 
                                              "Disable", *self.known_face_names)

        # Camera and face parameters
        self.FOCAL_LENGTH = 800
        self.KNOWN_FACE_WIDTH = 16  # cm

        self.prev_frame_gray = None
        self.tracked_points = None
        self.current_name = "Disable"
        self.old_distance = 0.0
        self.isPositionRight = 0
        self.timeLostTrack = None

        self.movement = []

        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.5)
        
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # Frame processing flag to prevent concurrent processing
        self.processing = False

    def demo_function(self):
        print("Button clicked!")

    def calculate_distance(self, face_width_pixels):
        if face_width_pixels == 0:
            return 0.0
        return (self.KNOWN_FACE_WIDTH * self.FOCAL_LENGTH) / face_width_pixels

    def run_app(self):
        try:
            self.input_frame.pack()
            self.input_frame.focus_set()
            self.cap_lbl.pack(anchor="center", pady=15)
            self.demo_button.pack(side='left', padx=10)
            self.face_detection_menu.pack(side='left')
            self.button_frame.pack(anchor="center", pady=10)
            
            # Start video stream after GUI is set up
            self.root.after(100, self.video_stream)
            self.root.mainloop()
        except Exception as e:
            print(f"Error running the application: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()

    def video_stream(self):
        # Skip if already processing (prevent concurrent calls)
        if self.processing:
            self.cap_lbl.after(10, self.video_stream)
            return
        
        self.processing = True
        
        try:
            h, w = 480, 720
            
            # Read frame directly without threading (macOS Tkinter doesn't like threads)
            ret, frame = self.cap.read()
            
            if not ret:
                print("Failed to grab frame")
                self.processing = False
                self.cap_lbl.after(100, self.video_stream)
                return
            
            # Initialize grayscale frame
            if self.prev_frame_gray is None:
                self.prev_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            frame = cv2.resize(frame, (w, h))
            cur_gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            selected_name = self.face_detection_var.get()

            # Reset tracking if name changed
            if selected_name != self.current_name:
                self.current_name = selected_name
                self.tracked_points = None
                self.movement.clear()
            
            # Print movements
            if self.movement:
                print("Movement: ", end="")
                for movement in self.movement:
                    print(f"{movement}, ", end="")
                print()
                self.movement.clear()
            
            # Process face detection
            if selected_name == self.current_name and selected_name != "Disable":
                if self.tracked_points is None:
                    # Initialize tracking
                    if selected_name in self.known_face_names:
                        idx = self.known_face_names.index(selected_name)
                        target_encodings = [self.known_face_encodings[idx]]
                        target_names = [self.known_face_names[idx]]
                    else:
                        target_encodings = self.known_face_encodings
                        target_names = self.known_face_names

                    # Perform face recognition (without threading on macOS)
                    face_locations, face_names, face_confidences, face_distances, scale_multiplier = \
                        recognize_faces(frame, target_encodings, target_names)

                    for (top, right, bottom, left), name, confidence, distance in \
                            zip(face_locations, face_names, face_confidences, face_distances):
                        top *= scale_multiplier
                        right *= scale_multiplier
                        bottom *= scale_multiplier
                        left *= scale_multiplier

                        if name != "Unknown":
                            face_width_pixels = right - left
                            distance = self.calculate_distance(face_width_pixels)
                            self.old_distance = distance
                            self.tracked_points = np.array([[[(left + right) // 2, (top + bottom) // 2]]], 
                                                          dtype=np.float32)
                            x, y = self.tracked_points[0].ravel()
                            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

                            # Determine movement
                            if distance > 85:
                                self.movement.append("Move forward")
                            elif distance < 40:
                                self.movement.append("Move backward")

                            face_center_x = (left + right) // 2
                            frame_center_x = w // 2
                            if face_center_x < frame_center_x - 50:
                                self.movement.append("Move left")
                            elif face_center_x > frame_center_x + 50:
                                self.movement.append("Move right")

                            face_center_y = (top + bottom) // 2
                            frame_center_y = h // 2
                            if face_center_y < frame_center_y - 50:
                                self.movement.append("Move upward")
                            elif face_center_y > frame_center_y + 50:
                                self.movement.append("Move downward")

                        # Draw bounding box
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                        font = cv2.FONT_HERSHEY_DUPLEX
                        label = f"{name} ({confidence:.2f}%) Distance: {distance:.2f} cm"
                        cv2.putText(frame, label, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
                
                elif self.tracked_points is not None:
                    # Track existing points
                    next_points, status, _ = cv2.calcOpticalFlowPyrLK(
                        self.prev_frame_gray, cur_gray_frame, self.tracked_points, None, **self.lk_params
                    )

                    if next_points is not None and len(next_points) > 0:
                        x, y = next_points[0].ravel()
                        
                        # Use MediaPipe for face detection
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = self.face_detection.process(rgb_frame)

                        faces = []
                        if results.detections:
                            for detection in results.detections:
                                bboxC = detection.location_data.relative_bounding_box
                                ih, iw, _ = frame.shape
                                xmin, ymin = int(bboxC.xmin * iw), int(bboxC.ymin * ih)
                                width, height = int(bboxC.width * iw), int(bboxC.height * ih)
                                faces.append((xmin, ymin, xmin + width, ymin + height))

                        matched_face = None
                        for face in faces:
                            left, top, right, bottom = face
                            if left <= x <= right and top <= y <= bottom:
                                matched_face = face
                                break
                        
                        if matched_face:
                            left, top, right, bottom = matched_face
                            face_width = right - left
                            estimated_distance = self.calculate_distance(face_width)
                            
                            if estimated_distance > 85:
                                self.movement.append("Move forward")
                                self.isPositionRight = 1
                            elif estimated_distance < 40:
                                self.movement.append("Move backward")
                                self.isPositionRight = 1

                            face_center_x = (left + right) // 2
                            frame_center_x = w // 2
                            if x < frame_center_x - 120:
                                self.movement.append("Move left")
                                self.isPositionRight = 1
                            elif x > frame_center_x + 120:
                                self.movement.append("Move right")
                                self.isPositionRight = 1

                            face_center_y = (top + bottom) // 2
                            frame_center_y = h // 2
                            if y < frame_center_y - 120:
                                self.movement.append("Move upward")
                                self.isPositionRight = 1
                            elif y > frame_center_y + 120:
                                self.movement.append("Move downward")
                                self.isPositionRight = 1

                            if self.isPositionRight == 0:
                                self.movement.clear()
                            
                            self.isPositionRight = 0
                            self.tracked_points = next_points
                        else:
                            print("Tracked point does not match any face. Resetting tracking.")
                            self.tracked_points = None
                            self.movement.clear()

                        # Check bounds
                        if 0 <= x < w and 0 <= y < h:
                            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
                        else:
                            print("Tracked point out of bounds. Resetting tracking.")
                            self.tracked_points = None
                            self.movement.clear()
                    else:
                        print("Optical flow returned invalid points. Resetting tracking.")
                        self.tracked_points = None
                        self.movement.clear()

            self.prev_frame_gray = cur_gray_frame.copy()
            
            # Update GUI (convert to PIL Image for Tkinter)
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            self.cap_lbl.imgtk = imgtk
            self.cap_lbl.configure(image=imgtk)
        
        except Exception as e:
            print(f"Error in video stream: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.processing = False
            # Schedule next frame (use longer delay on macOS for stability)
            delay = 30 if self.is_mac else 10
            self.cap_lbl.after(delay, self.video_stream)

    def cleanup(self):
        try:
            print("Cleaning up resources...")
            if self.cap is not None and self.cap.isOpened():
                self.cap.release()
            if self.face_detection is not None:
                self.face_detection.close()
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Error performing cleanup: {e}")


if __name__ == "__main__":
    gui = WebcamController()
    gui.run_app()