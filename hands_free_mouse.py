import streamlit as st
import av
import torch
import torchvision.transforms as transforms
from torch import nn
import numpy as np
import time
import threading
import pyautogui

class MouseController:
    def __init__(self):
        # Screen dimensions
        self.screen_w, self.screen_h = pyautogui.size()
        
        # State tracking
        self.last_click_time = 0
        self.click_cooldown = 0.5
        self.gesture_active = False
        
        # Smoothing parameters
        self.prev_x, self.prev_y = 0, 0
        self.smoothing = 5

    def detect_gestures(self, face_region):
        """
        Simple gesture detection based on face region
        """
        current_time = time.time()
        
        # Simplified gesture detection
        brightness = np.mean(face_region)
        
        # Left click on high brightness (eyebrow raise simulation)
        if brightness > 180 and current_time - self.last_click_time > self.click_cooldown:
            try:
                pyautogui.click()
                st.toast("Brightness Gesture - Left Click!")
                self.last_click_time = current_time
            except Exception as e:
                st.error(f"Click error: {e}")
        
        # Right click on low brightness (head tilt simulation)
        if brightness < 100 and current_time - self.last_click_time > self.click_cooldown:
            try:
                pyautogui.rightClick()
                st.toast("Darkness Gesture - Right Click!")
                self.last_click_time = current_time
            except Exception as e:
                st.error(f"Right-click error: {e}")

    def track_movement(self, face_center):
        """
        Track head movement for cursor control
        """
        if not self.gesture_active:
            return
        
        try:
            # Convert face position to screen coordinates
            screen_x = np.interp(face_center[0], (0, 640), (0, self.screen_w))
            screen_y = np.interp(face_center[1], (0, 480), (0, self.screen_h))
            
            # Apply smoothing
            screen_x = (self.prev_x * (self.smoothing-1) + screen_x) / self.smoothing
            screen_y = (self.prev_y * (self.smoothing-1) + screen_y) / self.smoothing
            
            # Move mouse
            pyautogui.moveTo(screen_x, screen_y)
            
            # Update previous coordinates
            self.prev_x, self.prev_y = screen_x, screen_y
        
        except Exception as e:
            st.error(f"Movement tracking error: {e}")

# Simple Face Detection Model (Placeholder)
class SimpleFaceDetector(nn.Module):
    def __init__(self):
        super().__init__()
        # Use a pre-trained model for face detection
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
    
    def detect_face(self, frame):
        """
        Basic face detection and tracking
        """
        # Convert frame to tensor
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        input_tensor = transform(frame).unsqueeze(0)
        
        # Simple placeholder for face detection
        # In a real scenario, you'd use more advanced detection
        return np.array([frame.shape[1]//2, frame.shape[0]//2])

def main():
    st.title("Hands-Free Mouse Control")
    
    # Gesture Guide
    st.markdown("""
    ### 🖱️ Gesture Controls
    - **Activate/Deactivate**: Toggle Switch
    - **Move Cursor**: Head Movement (when active)
    - **Left Click**: Bright Area Gesture
    - **Right Click**: Dark Area Gesture
    """)
    
    # System Compatibility Check
    try:
        import av
        import torch
        import pyautogui
    except ImportError as e:
        st.error(f"Required library not found: {e}")
        st.stop()
    
    # Initialize controllers
    mouse_controller = MouseController()
    face_detector = SimpleFaceDetector()
    
    # Gesture activation toggle
    gesture_toggle = st.checkbox("Activate Gesture Control", key="gesture_active")
    mouse_controller.gesture_active = gesture_toggle
    
    # Webcam input
    webcam = st.camera_input("Open Webcam", key="webcam")
    
    if webcam is not None and gesture_toggle:
        # Convert to OpenCV format
        bytes_data = webcam.getvalue()
        av_frame = av.VideoFrame.from_ndarray(
            np.frombuffer(bytes_data, np.uint8), format="bgr24"
        )
        
        # Convert to numpy array
        frame = av_frame.to_ndarray(format="bgr24")
        
        # Flip frame
        frame = np.fliplr(frame)
        
        try:
            # Detect face center
            face_center = face_detector.detect_face(frame)
            
            # Get face region for gesture detection
            x, y = face_center
            face_region = frame[max(0, y-50):min(frame.shape[0], y+50), 
                                max(0, x-50):min(frame.shape[1], x+50)]
            
            # Perform gesture detection
            mouse_controller.detect_gestures(face_region)
            
            # Track movement
            mouse_controller.track_movement(face_center)
            
            # Optional: Visualize detection point
            st.image(frame, channels="BGR")
        
        except Exception as e:
            st.error(f"Processing error: {e}")

if __name__ == "__main__":
    main()
