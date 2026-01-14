# Drone Face Tracking System

A Python-based reactive autonomous drone control system that uses face recognition and optical flow tracking to follow a specific person. Built for DJI Tello drones with support for both macOS and Windows platforms.

## Features

- **Face Recognition**: Identifies and tracks specific individuals from a database of known faces
- **Autonomous Following**: Drone automatically adjusts position to maintain optimal distance and centering
- **Optical Flow Tracking**: Uses Lucas-Kanade optical flow for smooth, continuous tracking between face recognition frames
- **Multiple Control Modes**:
  - Keyboard control (WASD + Arrow keys)
  - Joystick/Gamepad support
  - Autonomous face-following mode
- **Real-time Metrics**: FPS, tracking success rate, and stability measurements
- **Webcam Testing**: Test face recognition without a drone using webcam scripts

## Project Structure

```
├── Drone_Mac.py          # Main drone controller for macOS
├── Drone_Windows.py      # Main drone controller for Windows
├── Webcam_Mac.py         # Webcam-only testing for macOS
├── Webcam_Windows.py     # Webcam-only testing for Windows
├── flight_commands.py    # Drone flight command utilities
├── faces/                # Directory for known face images
│   ├── Person1.jpg
│   └── Person2.png
└── requirements.txt      # Python dependencies
```

## Requirements

### Hardware
- DJI Tello drone (for drone scripts)
- Webcam (for webcam testing scripts)
- USB Gamepad/Joystick (optional)

### Software
- Python 3.8 - 3.10 (recommended)
- Compatible operating system: macOS or Windows

## Installation

### 1. Clone or Download the Repository

```bash
cd /path/to/SkripsiRepo
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv

# On macOS/Linux
source venv/bin/activate

# On Windows
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add Known Faces

Place images of people you want to track in the `faces/` directory:
- Supported formats: `.jpg`, `.png`, `.jpeg`
- The filename (without extension) will be used as the person's name
- Example: `John_Doe.jpg` → Person name: "John_Doe"

## Usage

### Running with Drone

**macOS:**
```bash
python Drone_Mac.py
```

**Windows:**
```bash
python Drone_Windows.py
```

### Testing with Webcam (No Drone Required)

**macOS:**
```bash
python Webcam_Mac.py
```

**Windows:**
```bash
python Webcam_Windows.py
```

## Controls

### Keyboard Controls
| Key | Action |
|-----|--------|
| `W` | Move Forward |
| `S` | Move Backward |
| `A` | Move Left |
| `D` | Move Right |
| `↑` | Move Up |
| `↓` | Move Down |
| `Q` | Yaw Left (Rotate) |
| `E` | Yaw Right (Rotate) |

### Joystick Controls (Xbox-style)
| Button/Axis | Action |
|-------------|--------|
| Left Stick Y | Throttle (Up/Down) |
| Left Stick X | Yaw (Rotate) |
| Right Stick Y | Pitch (Forward/Backward) |
| Right Stick X | Roll (Left/Right) |
| START (Button 7) | Takeoff |
| SELECT (Button 6) | Land |
| X (Button 0) | Emergency Stop |

### GUI Controls
- **Takeoff/Land Button**: Toggle drone flight state
- **Face Dropdown**: Select a person to track or "Disable" for manual control

## How It Works

1. **Face Recognition Phase**: When a target person is selected, the system uses `face_recognition` library to identify the person in the video feed.

2. **Tracking Initialization**: Once the target face is found, a tracking point is initialized at the face center.

3. **Optical Flow Tracking**: The system uses Lucas-Kanade optical flow to track the point between frames, reducing computational load.

4. **Distance Estimation**: Using the known average face width and camera focal length, the system estimates distance to the target.

5. **Movement Commands**: Based on the target's position and distance, the drone automatically adjusts:
   - Forward/Backward: Maintain optimal distance (60-85 cm)
   - Left/Right: Keep target centered horizontally
   - Up/Down: Keep target centered vertically

## Evaluation Metrics

The system tracks several performance metrics:
- **Average FPS**: Frames processed per second
- **Tracking Success Rate (TSR)**: Percentage of frames where tracking was successful
- **Tracking Stability**: Average pixel movement between frames (lower = more stable)
- **Tracking Duration**: Total time spent in autonomous tracking mode

## Troubleshooting

### Camera Not Detected
- Ensure the drone is connected to your WiFi
- For webcam scripts, check if another application is using the camera

### Face Recognition Not Working
- Ensure face images in `faces/` directory are clear and well-lit
- Try images with the face taking up most of the frame
- Supported formats: `.jpg`, `.png`, `.jpeg`

### Low FPS on macOS
- The macOS version uses reduced threading for Tkinter compatibility
- This is expected behavior for stability

### Drone Not Responding
- Check drone battery level
- Ensure you're connected to the Tello WiFi network
- Try restarting the drone and reconnecting

## Dependencies

See `requirements.txt` for full list. Key dependencies:
- `opencv-python`: Video capture and image processing
- `face_recognition`: Face detection and recognition
- `djitellopy`: DJI Tello drone control
- `mediapipe`: Additional face detection for tracking
- `pygame`: Joystick/gamepad support
- `Pillow`: Image handling for Tkinter GUI
- `numpy`: Numerical operations

## License

This project is for educational/research purposes (Skripsi/Thesis project).

