from tkinter import Tk, Label, Button, Frame, StringVar, OptionMenu
import traceback
import os
import sys
import platform
import time
import threading

os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'
os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'

import cv2
if platform.system() == 'Darwin':
    cv2.setNumThreads(0)

import face_recognition
import pygame
from PIL import Image, ImageTk
from djitellopy import tello
import numpy as np
import mediapipe as mp


class FaceRecognitionSystem:
    
    def __init__(self, faces_directory):
        self.known_face_encodings = []
        self.known_face_names = []
        self._load_known_faces(faces_directory)
        
    def _load_known_faces(self, directory):
        if not os.path.exists(directory):
            print(f"[WARNING] Faces directory '{directory}' not found")
            return
            
        for file_name in os.listdir(directory):
            if file_name.endswith((".jpg", ".png", ".jpeg")):
                image_path = os.path.join(directory, file_name)
                try:
                    image = face_recognition.load_image_file(image_path)
                    face_encodings = face_recognition.face_encodings(image)
                    
                    if face_encodings:
                        self.known_face_encodings.append(face_encodings[0])
                        self.known_face_names.append(os.path.splitext(file_name)[0])
                        print(f"[INFO] Loaded face: {file_name}")
                    else:
                        print(f"[WARNING] No face found in {file_name}")
                except Exception as e:
                    print(f"[ERROR] Failed to load {file_name}: {e}")
                    
        print(f"[INFO] Loaded {len(self.known_face_names)} known faces")
    
    def get_known_names(self):
        return self.known_face_names.copy()
    
    def get_encodings_for_name(self, name):
        if name in self.known_face_names:
            idx = self.known_face_names.index(name)
            return [self.known_face_encodings[idx]], [self.known_face_names[idx]]
        return self.known_face_encodings, self.known_face_names
    
    def calculate_confidence(self, face_distance, face_match_threshold=0.6):
        if face_distance > face_match_threshold:
            linear_val = (1.0 - face_distance) / (0.1 - face_match_threshold)
            return max(0.0, min(1.0, linear_val)) * 100
        else:
            linear_val = (1.0 - face_distance) / (face_match_threshold - 0.1)
            return max(0.0, min(1.0, linear_val)) * 100
    
    def recognize_faces(self, frame, target_name=None):
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        if not rgb_small_frame.flags['C_CONTIGUOUS']:
            rgb_small_frame = np.ascontiguousarray(rgb_small_frame)
        
        if target_name:
            target_encodings, target_names = self.get_encodings_for_name(target_name)
        else:
            target_encodings = self.known_face_encodings
            target_names = self.known_face_names
        
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        face_names = []
        face_confidences = []
        face_distances = []
        
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(target_encodings, face_encoding)
            name = "Unknown"
            confidence = 0.0
            distance = 0.0
            
            face_distances_current = face_recognition.face_distance(
                target_encodings, face_encoding
            )
            
            if len(face_distances_current) > 0:
                best_match_index = np.argmin(face_distances_current)
                if matches[best_match_index]:
                    name = target_names[best_match_index]
                    confidence = self.calculate_confidence(
                        face_distances_current[best_match_index]
                    )
                    distance = face_distances_current[best_match_index]
            
            face_names.append(name)
            face_confidences.append(confidence)
            face_distances.append(distance)
        
        return face_locations, face_names, face_confidences, face_distances


class DroneController:
    
    def __init__(self):
        self.drone = None
        self.frame_reader = None
        self.is_flying = False
        self.is_landing = False
        
    def connect(self):
        self.drone = tello.Tello()
        self.drone.connect()
        self.drone.speed = 100
        
    def start_stream(self):
        if self.drone:
            self.drone.streamon()
            
    def stop_stream(self):
        try:
            if self.drone:
                self.drone.streamoff()
        except Exception as e:
            print(f"[WARNING] Error stopping stream: {e}")
            
    def get_frame_reader(self):
        if not self.frame_reader and self.drone:
            self.frame_reader = self.drone.get_frame_read()
        return self.frame_reader
    
    def get_battery(self):
        if self.drone:
            return self.drone.get_battery()
        return 0
    
    def takeoff(self):
        if self.drone and not self.is_flying:
            self.drone.takeoff()
            self.is_flying = True
            self.is_landing = False
            
    def land(self):
        if self.drone and self.is_flying:
            self.drone.land()
            self.is_landing = True
            self.is_flying = False
            
    def emergency(self):
        if self.drone:
            self.drone.emergency()
            self.is_flying = False
            self.is_landing = False
            
    def send_rc_control(self, left_right, forward_backward, up_down, yaw):
        if self.drone:
            self.drone.send_rc_control(left_right, forward_backward, up_down, yaw)
            
    def send_control_command(self, command):
        if self.drone:
            self.drone.send_control_command(command)


def start_flying(event, direction, drone_controller, speed):
    direction_map = {
        'forward': (0, speed, 0, 0),
        'backward': (0, -speed, 0, 0),
        'left': (-speed, 0, 0, 0),
        'right': (speed, 0, 0, 0),
        'upward': (0, 0, speed, 0),
        'downward': (0, 0, -speed, 0),
        'yaw_left': (0, 0, 0, -speed),
        'yaw_right': (0, 0, 0, speed),
    }
    
    if direction in direction_map:
        left_right, forward_backward, up_down, yaw = direction_map[direction]
        drone_controller.send_rc_control(left_right, forward_backward, up_down, yaw)


def stop_flying(event, drone_controller):
    drone_controller.send_rc_control(0, 0, 0, 0)


class KeyboardController:
    
    def __init__(self, drone_controller, input_frame):
        self.drone_controller = drone_controller
        self.input_frame = input_frame
        
    def bind_keys(self):
        frame = self.input_frame
        drone = self.drone_controller
        
        frame.bind('<KeyPress-w>', lambda event: start_flying(event, 'forward', drone, 100))
        frame.bind('<KeyRelease-w>', lambda event: stop_flying(event, drone))
        
        frame.bind('<KeyPress-s>', lambda event: start_flying(event, 'backward', drone, 100))
        frame.bind('<KeyRelease-s>', lambda event: stop_flying(event, drone))
        
        frame.bind('<KeyPress-a>', lambda event: start_flying(event, 'left', drone, 100))
        frame.bind('<KeyRelease-a>', lambda event: stop_flying(event, drone))
        
        frame.bind('<KeyPress-d>', lambda event: start_flying(event, 'right', drone, 100))
        frame.bind('<KeyRelease-d>', lambda event: stop_flying(event, drone))
        
        frame.bind('<KeyPress-Up>', lambda event: start_flying(event, 'upward', drone, 100))
        frame.bind('<KeyRelease-Up>', lambda event: stop_flying(event, drone))
        
        frame.bind('<KeyPress-Down>', lambda event: start_flying(event, 'downward', drone, 100))
        frame.bind('<KeyRelease-Down>', lambda event: stop_flying(event, drone))
        
        frame.bind('<KeyPress-q>', lambda event: start_flying(event, 'yaw_left', drone, 100))
        frame.bind('<KeyRelease-q>', lambda event: stop_flying(event, drone))
        
        frame.bind('<KeyPress-e>', lambda event: start_flying(event, 'yaw_right', drone, 100))
        frame.bind('<KeyRelease-e>', lambda event: stop_flying(event, drone))
        
        print("[INFO] Keyboard controls bound")


class JoystickController:
    
    def __init__(self, drone_controller):
        self.drone_controller = drone_controller
        self.joystick_running = False
        self.available = False
        self.joystick = None
        self.pygame_initialized = False
        self._control_thread = None
        
    def initialize(self):
        if self.pygame_initialized:
            return
        
        try:
            pygame.init()
            pygame.joystick.init()
            self.pygame_initialized = True
            
            if pygame.joystick.get_count() == 0:
                print("[INFO] No joystick connected.")
                return
            
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            self.available = True
            print(f"[INFO] Joystick connected: {self.joystick.get_name()}")
            
        except Exception as e:
            print(f"[WARNING] Joystick init failed: {e}")
    
    def scale_axis(self, value, deadzone=0.1):
        if abs(value) < deadzone:
            return 0
        return int(value * 100)
    
    def _joystick_control_loop(self):
        last_roll, last_pitch, last_throttle, last_yaw = 0, 0, 0, 0
        self.joystick_running = True
        
        print("[INFO] Joystick control loop started")
        
        while self.joystick_running:
            try:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                    
                    if event.type == pygame.JOYBUTTONDOWN:
                        self._handle_button_press(event.button)
                
                roll = self.scale_axis(self.joystick.get_axis(2))
                pitch = -self.scale_axis(self.joystick.get_axis(3))
                throttle = -self.scale_axis(self.joystick.get_axis(1))
                yaw = self.scale_axis(self.joystick.get_axis(0))
                
                if roll != 0 or pitch != 0 or throttle != 0 or yaw != 0:
                    self.drone_controller.send_rc_control(roll, pitch, throttle, yaw)
                    last_roll, last_pitch, last_throttle, last_yaw = roll, pitch, throttle, yaw
                    
                elif (roll == pitch == throttle == yaw == 0 and 
                      (last_roll != 0 or last_pitch != 0 or 
                       last_throttle != 0 or last_yaw != 0)):
                    self.drone_controller.send_rc_control(0, 0, 0, 0)
                    last_roll, last_pitch, last_throttle, last_yaw = 0, 0, 0, 0
                    print("Stopping movement")
                
                time.sleep(0.1)
                
            except Exception as e:
                print(f"[ERROR] Joystick control loop: {e}")
                time.sleep(0.5)
    
    def _handle_button_press(self, button):
        if button == 7 and not self.drone_controller.is_flying:
            print("Takeoff (Joystick)")
            self.drone_controller.takeoff()
        
        elif button == 6 and self.drone_controller.is_flying and not self.drone_controller.is_landing:
            print("Landing (Joystick)")
            self.drone_controller.land()
        
        elif button == 0:
            print("EMERGENCY STOP! (Joystick)")
            self.drone_controller.emergency()
    
    def start(self):
        if not self.available:
            print("[INFO] Joystick not available.")
            return
        
        if not self.joystick_running:
            self._control_thread = threading.Thread(
                target=self._joystick_control_loop, 
                daemon=True
            )
            self._control_thread.start()
            print("[INFO] Joystick control thread started.")
    
    def stop(self):
        self.joystick_running = False
        if self._control_thread:
            self._control_thread.join(timeout=1.0)
        print("[INFO] Joystick control thread stopped.")


class AutonomousController:
    
    FOCAL_LENGTH = 800
    KNOWN_FACE_WIDTH = 16
    DISTANCE_CLOSE_MIN = 50
    DISTANCE_CLOSE_MAX = 60
    DISTANCE_FAR_MIN = 85
    DISTANCE_FAR_MAX = 95
    CENTER_THRESHOLD_X = 120
    CENTER_THRESHOLD_Y = 90
    
    def __init__(self, drone_controller, face_recognition, is_mac=False):
        self.drone_controller = drone_controller
        self.face_recognition = face_recognition
        self.is_mac = is_mac
        
        self.prev_frame_gray = None
        self.tracked_points = None
        self.current_target_name = "Disable"
        self.movement_queue = []
        self.prev_movement = ""
        self.position_state = 2
        
        self.total_frames = 0
        self.success_frames = 0
        self.prev_point = None
        self.point_motion = []
        
        self.tracking_start_time = None
        self.is_tracking_active = False
        self.evaluation_active = False
        self.tracking_attempt_active = False
        self.autonomous_start_time = None
        self.autonomous_duration = 0.0
        self.autonomous_active = False
        
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = None
        
        self.lk_params = dict(
            winSize=(25, 25),
            maxLevel=4,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
    
    def initialize_mediapipe(self):
        if self.face_detection is None:
            try:
                self.face_detection = self.mp_face_detection.FaceDetection(
                    min_detection_confidence=0.5
                )
                print("[INFO] MediaPipe initialized")
            except Exception as e:
                print(f"[WARNING] MediaPipe init failed: {e}")
    
    def calculate_distance(self, face_width_pixels):
        if face_width_pixels == 0:
            return 0.0
        return (self.KNOWN_FACE_WIDTH * self.FOCAL_LENGTH) / face_width_pixels
    
    def process_frame(self, frame, gray_frame, target_name, frame_width, frame_height):
        if self.face_detection is None:
            self.initialize_mediapipe()
        
        self._autonomous_logic(frame, gray_frame, target_name, frame_width, frame_height)
        self.prev_frame_gray = gray_frame.copy()
    
    def _autonomous_logic(self, frame, cur_gray_frame, selected_name, w, h):
        if self.tracking_attempt_active:
            self.total_frames += 1
        
        self._execute_movements()
        
        if self.tracked_points is None:
            self._search_for_face(frame, selected_name, w, h)
        else:
            self._track_face(frame, cur_gray_frame, w, h)
    
    def _search_for_face(self, frame, selected_name, w, h):
        face_locations, face_names, face_confidences, face_distances = \
            self.face_recognition.recognize_faces(frame, selected_name)
        
        for (top, right, bottom, left), name, confidence, distance in zip(
                face_locations, face_names, face_confidences, face_distances):
            
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            
            if name != "Unknown":
                self._initialize_tracking(left, right, top, bottom, w, h)
                
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            label = f"{name} ({confidence:.2f}%) Dist: {distance:.2f} cm"
            cv2.putText(frame, label, (left + 6, bottom - 6),
                       cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
    
    def _initialize_tracking(self, left, right, top, bottom, w, h):
        self.tracking_attempt_active = True
        
        if not self.is_tracking_active:
            self.tracking_start_time = time.time()
            self.is_tracking_active = True
        
        if not self.evaluation_active:
            self.evaluation_active = True
        
        face_width_pixels = right - left
        distance = self.calculate_distance(face_width_pixels)
        
        self.tracked_points = np.array(
            [[[(left + right) // 2, (top + bottom) // 2]]], 
            dtype=np.float32
        )
        
        self._generate_movement_commands(
            left, right, top, bottom, distance, w, h
        )
    
    def _track_face(self, frame, cur_gray_frame, w, h):
        if self.evaluation_active:
            self.total_frames += 1
            self.success_frames += 1
        
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_frame_gray, cur_gray_frame, self.tracked_points, 
            None, **self.lk_params
        )
        
        if next_points is not None and len(next_points) > 0:
            x, y = next_points[0].ravel()
            self.tracked_points = next_points
            self.success_frames += 1
            
            if self.prev_point is not None:
                dist = np.linalg.norm([x - self.prev_point[0], y - self.prev_point[1]])
                self.point_motion.append(dist)
            self.prev_point = (x, y)
            
            self._update_tracking_commands(frame, x, y, w, h)
            
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
            else:
                self._stop_tracking()
        else:
            self._stop_tracking()
    
    def _update_tracking_commands(self, frame, x, y, w, h):
        if self.face_detection is None:
            return
        
        try:
            results = self.face_detection.process(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            )
            
            faces = []
            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    h_img, w_img, _ = frame.shape
                    xmin = int(bboxC.xmin * w_img)
                    ymin = int(bboxC.ymin * h_img)
                    width = int(bboxC.width * w_img)
                    height = int(bboxC.height * h_img)
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
                distance = self.calculate_distance(face_width)
                
                print(f"Estimated Distance: {distance:.2f} cm")
                self._generate_movement_commands(left, right, top, bottom, distance, w, h)
                
        except Exception as e:
            pass
    
    def _generate_movement_commands(self, left, right, top, bottom, distance, w, h):
        self.movement_queue.clear()
        self.position_state = 1
        
        if self.DISTANCE_FAR_MIN < distance < self.DISTANCE_FAR_MAX:
            self.movement_queue.append("Move forward1")
        elif distance > self.DISTANCE_FAR_MAX:
            self.movement_queue.append("Move forward2")
        elif self.DISTANCE_CLOSE_MAX > distance > self.DISTANCE_CLOSE_MIN:
            self.movement_queue.append("Move backward1")
        elif distance <= self.DISTANCE_CLOSE_MIN:
            self.movement_queue.append("Move backward2")
        
        face_center_x = (left + right) // 2
        frame_center_x = w // 2
        
        if face_center_x < frame_center_x - self.CENTER_THRESHOLD_X:
            if face_center_x > frame_center_x - 150:
                self.movement_queue.append("Move left1")
            else:
                self.movement_queue.append("Move left2")
        elif face_center_x > frame_center_x + self.CENTER_THRESHOLD_X:
            if face_center_x < frame_center_x + 150:
                self.movement_queue.append("Move right1")
            else:
                self.movement_queue.append("Move right2")
        
        face_center_y = (top + bottom) // 2
        frame_center_y = h // 2
        
        if face_center_y < frame_center_y - self.CENTER_THRESHOLD_Y:
            self.movement_queue.append("Move upward")
        elif face_center_y > frame_center_y + self.CENTER_THRESHOLD_Y:
            self.movement_queue.append("Move downward")
        
        if self.position_state == 1:
            self.position_state = 2
        elif self.position_state == 2:
            self.position_state = 0
        elif self.position_state == 0:
            self.position_state = 2
    
    def _execute_movements(self):
        if not self.movement_queue:
            return
        
        for movement in self.movement_queue:
            if self.position_state == 0:
                self._execute_correction(movement)
                break
            
            self._execute_movement(movement)
            self.prev_movement = movement
        
        self.movement_queue.clear()
    
    def _execute_correction(self, movement):
        correction_map = {
            "Move forward": ('backward', 10),
            "Move backward": ('forward', 10),
            "Move left": ('right', 10),
            "Move right": ('left', 10),
            "Move upward": ('downward', 10),
            "Move downward": ('upward', 10),
        }
        
        if self.prev_movement in correction_map:
            direction, speed = correction_map[self.prev_movement]
            start_flying(None, direction, self.drone_controller, speed)
            self.drone_controller.send_control_command("rc 0 0 0 0")
        
        self.prev_movement = ""
    
    def _execute_movement(self, movement):
        movement_map = {
            "Move forward1": ('forward', 20),
            "Move forward2": ('forward', 30),
            "Move backward1": ('backward', 20),
            "Move backward2": ('backward', 30),
            "Move left1": ('left', 20),
            "Move left2": ('left', 30),
            "Move right1": ('right', 20),
            "Move right2": ('right', 30),
            "Move upward": ('upward', 30),
            "Move downward": ('downward', 30),
        }
        
        if movement in movement_map:
            direction, speed = movement_map[movement]
            start_flying(None, direction, self.drone_controller, speed)
    
    def _stop_tracking(self):
        self.tracked_points = None
        self.movement_queue.clear()
        
        if self.is_tracking_active:
            self.is_tracking_active = False
            self.evaluation_active = False
    
    def get_metrics(self):
        if self.autonomous_active and self.autonomous_start_time is not None:
            self.autonomous_duration += time.time() - self.autonomous_start_time
            self.autonomous_active = False
        
        return {
            'total_frames': self.total_frames,
            'success_frames': self.success_frames,
            'point_motion': self.point_motion,
            'duration': self.autonomous_duration
        }
    
    def cleanup(self):
        if self.face_detection is not None:
            try:
                self.face_detection.close()
            except:
                pass


class VideoStream:
    
    def __init__(self, cap, cap_lbl, face_detection_var, joystick_controller, 
                 autonomous_controller, drone_controller, is_mac=False):
        self.cap = cap
        self.cap_lbl = cap_lbl
        self.face_detection_var = face_detection_var
        self.joystick_controller = joystick_controller
        self.autonomous_controller = autonomous_controller
        self.drone_controller = drone_controller
        self.is_mac = is_mac
        
        self.is_running = False
        self.stream_initialized = False
        self.frame_count = 0
        
        self.prev_time = time.time()
        self.fps = 0.0
        self.fps_list = []
        
        self.frame_width = 720
        self.frame_height = 480
    
    def start(self):
        self.is_running = True
        delay = 500 if self.is_mac else 100
        self.cap_lbl.after(delay, self._video_stream_loop)
    
    def stop(self):
        self.is_running = False
    
    def get_fps_list(self):
        return self.fps_list.copy()
    
    def _video_stream_loop(self):
        try:
            if not self.is_running:
                return
            
            frame = self._read_frame()
            
            if frame is None:
                delay = 500 if not self.stream_initialized else (50 if self.is_mac else 10)
                self.cap_lbl.after(delay, self._video_stream_loop)
                return
            
            if not self.stream_initialized:
                self.stream_initialized = True
                print("[INFO] Video stream successfully initialized in GUI loop")
            
            self.frame_count += 1
            
            self._process_frame(frame)
            self._update_fps(frame)
            self._display_frame(frame)
            
            self.cap_lbl.after(1, self._video_stream_loop)
            
        except Exception as e:
            print(f"[ERROR] Video stream: {e}")
            traceback.print_exc()
            self.cap_lbl.after(100, self._video_stream_loop)
    
    def _read_frame(self):
        try:
            frame = self.cap.frame
            if frame is None or frame.size == 0:
                return None
            return frame
        except Exception as e:
            if self.frame_count < 10:
                print(f"[WARNING] Frame read error (attempt {self.frame_count}): {e}")
            return None
    
    def _process_frame(self, frame):
        frame = cv2.resize(frame, (self.frame_width, self.frame_height))
        
        if self.autonomous_controller.prev_frame_gray is None:
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                self.autonomous_controller.prev_frame_gray = gray
            except Exception as e:
                print(f"[ERROR] Gray conversion: {e}")
                return
        
        cur_gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        selected_name = self.face_detection_var.get()
        
        if selected_name == "Disable":
            self._handle_manual_mode()
        else:
            self._handle_autonomous_mode(frame, cur_gray_frame, selected_name)
    
    def _handle_manual_mode(self):
        if self.autonomous_controller.autonomous_active:
            duration = time.time() - self.autonomous_controller.autonomous_start_time
            self.autonomous_controller.autonomous_duration += duration
            self.autonomous_controller.autonomous_active = False
            self.autonomous_controller.autonomous_start_time = None
        
        if self.joystick_controller.available:
            if not self.joystick_controller.joystick_running:
                self.joystick_controller.start()
    
    def _handle_autonomous_mode(self, frame, cur_gray_frame, selected_name):
        if not self.autonomous_controller.autonomous_active:
            self.autonomous_controller.autonomous_start_time = time.time()
            self.autonomous_controller.autonomous_active = True
        
        if selected_name != self.autonomous_controller.current_target_name:
            self.autonomous_controller.current_target_name = selected_name
            self.autonomous_controller.tracked_points = None
            self.autonomous_controller.movement_queue.clear()
        
        self.autonomous_controller.process_frame(
            frame, cur_gray_frame, selected_name, 
            self.frame_width, self.frame_height
        )
    
    def _update_fps(self, frame):
        current_time = time.time()
        elapsed = current_time - self.prev_time
        
        if elapsed > 0:
            self.fps = 1 / elapsed
        
        self.prev_time = current_time
        
        is_actively_tracking = (
            self.autonomous_controller.is_tracking_active and 
            self.autonomous_controller.tracked_points is not None
        )
        is_flying = (
            self.drone_controller.is_flying and 
            not self.drone_controller.is_landing
        )
        
        if is_actively_tracking and is_flying:
            self.fps_list.append(self.fps)
        
        cv2.putText(
            frame, 
            f"FPS: {self.fps:.2f}", 
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 255, 0), 
            2
        )
    
    def _display_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self.cap_lbl.imgtk = imgtk
        self.cap_lbl.configure(image=imgtk)


class DroneApplication:
    
    def __init__(self):
        self.is_mac = platform.system() == 'Darwin'
        
        print(f"[INFO] Platform: {platform.system()}")
        print(f"[INFO] OpenCV version: {cv2.__version__}")
        print(f"[INFO] OpenCV threads: {cv2.getNumThreads()}")
        
        self._setup_gui()
        self._setup_drone()
        self._setup_face_recognition()
        self._setup_controllers()
        self._setup_video_stream()
        
    def _setup_gui(self):
        self.root = Tk()
        self.root.title("Drone Keyboard Controller - Tkinter")
        self.root.minsize(800, 600)
        
        if self.is_mac:
            self.root.lift()
            self.root.attributes('-topmost', True)
            self.root.after_idle(self.root.attributes, '-topmost', False)
        
        self.input_frame = Frame(self.root)
        self.cap_lbl = Label(self.root)
        self.button_frame = Frame(self.root)
        
        self.demo_button = Button(
            self.button_frame, 
            text="Takeoff/Land", 
            command=self._takeoff_land_handler
        )
        
        self.face_detection_var = StringVar(self.root)
        self.face_detection_var.set("Disable")
        
    def _setup_drone(self):
        print("[INFO] Connecting to drone...")
        self.drone_controller = DroneController()
        self.drone_controller.connect()
        
        print("[INFO] Starting video stream...")
        self.drone_controller.start_stream()
        
        if self.is_mac:
            print("[INFO] Waiting for video stream to stabilize...")
            time.sleep(2.0)
        
        print(f"Battery: {self.drone_controller.get_battery()}%")
        
    def _setup_face_recognition(self):
        faces_dir = "faces"
        self.face_recognition = FaceRecognitionSystem(faces_dir)
        
        face_options = ["Disable"] + self.face_recognition.get_known_names()
        self.face_detection_menu = OptionMenu(
            self.button_frame, 
            self.face_detection_var,
            *face_options
        )
        
    def _setup_controllers(self):
        self.keyboard_controller = KeyboardController(
            self.drone_controller, 
            self.input_frame
        )
        
        self.joystick_controller = JoystickController(self.drone_controller)
        
        self.autonomous_controller = AutonomousController(
            drone_controller=self.drone_controller,
            face_recognition=self.face_recognition,
            is_mac=self.is_mac
        )
        
    def _setup_video_stream(self):
        self.video_stream = VideoStream(
            cap=self.drone_controller.get_frame_reader(),
            cap_lbl=self.cap_lbl,
            face_detection_var=self.face_detection_var,
            joystick_controller=self.joystick_controller,
            autonomous_controller=self.autonomous_controller,
            drone_controller=self.drone_controller,
            is_mac=self.is_mac
        )
        
    def _takeoff_land_handler(self):
        if not self.drone_controller.is_flying:
            print("Takeoff")
            self.drone_controller.takeoff()
        else:
            print("Landing")
            self.drone_controller.land()
    
    def run(self):
        try:
            self.keyboard_controller.bind_keys()
            
            self.input_frame.pack()
            self.input_frame.focus_set()
            self.cap_lbl.pack(anchor="center", pady=15)
            self.demo_button.pack(side='left', padx=10)
            self.face_detection_menu.pack(side='left')
            self.button_frame.pack(anchor="center", pady=10)
            
            if self.is_mac:
                self.root.after(1000, self.joystick_controller.initialize)
            else:
                self.joystick_controller.initialize()
            
            print("[INFO] Starting video stream loop...")
            self.video_stream.start()
            
            if self.is_mac:
                self.root.after(100, self._check_responsiveness)
            
            self.root.mainloop()
            
        except Exception as e:
            print(f"[ERROR] Application: {e}")
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def _check_responsiveness(self):
        if self.is_mac:
            self.root.update_idletasks()
            self.root.after(100, self._check_responsiveness)
    
    def cleanup(self):
        print("\n[INFO] Starting cleanup...")
        
        self.video_stream.stop()
        time.sleep(0.5)
        
        self.drone_controller.stop_stream()
        
        self._print_metrics()
        
        self.autonomous_controller.cleanup()
        
        print("[INFO] Cleanup complete.")
    
    def _print_metrics(self):
        print("\n=== EVALUATION METRICS ===")
        
        fps_list = self.video_stream.get_fps_list()
        if fps_list:
            avg_fps = sum(fps_list) / len(fps_list)
            print(f"Average FPS       : {avg_fps:.2f}")
        
        metrics = self.autonomous_controller.get_metrics()
        
        if metrics['total_frames'] > 0:
            tsr = (metrics['success_frames'] / metrics['total_frames']) * 100
            print(f"Tracking Success  : {tsr:.2f}%")
        
        if metrics['point_motion']:
            stability = sum(metrics['point_motion']) / len(metrics['point_motion'])
            print(f"Tracking Stability: {stability:.2f} px")
        
        print(f"Tracking Duration : {metrics['duration']:.2f} sec")
        print("===== END METRICS =====\n")


def main():
    app = DroneApplication()
    app.run()


if __name__ == "__main__":
    main()