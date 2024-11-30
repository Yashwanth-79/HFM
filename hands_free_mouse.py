import streamlit as st

# Check and handle library imports
try:
    import cv2
except ImportError:
    st.error("OpenCV (cv2) is not installed. Please install it using: pip install opencv-python")
    st.stop()

try:
    import numpy as np
except ImportError:
    st.error("NumPy is not installed. Please install it using: pip install numpy")
    st.stop()

try:
    import mediapipe as mp
except ImportError:
    st.error("MediaPipe is not installed. Please install it using: pip install mediapipe")
    st.stop()

try:
    import pyautogui
except ImportError:
    st.error("PyAutoGUI is not installed. Please install it using: pip install pyautogui")
    st.stop()

import time

class GestureMouseController:
    def __init__(self):
        # Screen dimensions
        try:
            self.screen_w, self.screen_h = pyautogui.size()
        except Exception as e:
            st.error(f"Error getting screen size: {e}")
            self.screen_w, self.screen_h = 1920, 1080  # Default fallback
        
        # Tracking parameters
        self.prev_x, self.prev_y = 0, 0
        self.smoothing = 5  # Increased smoothing for slower movement
        
        # Gesture detection variables
        self.last_click_time = 0
        self.click_cooldown = 0.5  # 0.5 seconds between clicks
        
        # Tracking variables for specific gestures
        self.last_eyebrow_state = False
        self.last_head_position = None
        self.head_tilt_threshold = 20  # Pixel difference for head tilt
        
        # Gesture control state
        self.gesture_active = False
        
        # MediaPipe Face Mesh setup
        try:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                min_detection_confidence=0.5
            )
        except Exception as e:
            st.error(f"Error initializing MediaPipe: {e}")
            self.face_mesh = None

    def detect_click_gestures(self, landmarks, frame_shape):
        """
        Detect click gestures using facial landmarks
        """
        current_time = time.time()
        height, width, _ = frame_shape
        
        try:
            # Eyebrow raise detection using landmark positions
            left_eyebrow_points = [105, 63, 190, 52, 25]
            right_eyebrow_points = [334, 296, 416, 277, 257]
            
            # Calculate eyebrow movement
            left_eyebrow_y = np.mean([landmarks[p].y * height for p in left_eyebrow_points])
            right_eyebrow_y = np.mean([landmarks[p].y * height for p in right_eyebrow_points])
            
            # Left Click - Eyebrow Raise Detection
            if left_eyebrow_y < height * 0.3:  # Adjust threshold as needed
                if not self.last_eyebrow_state:
                    if current_time - self.last_click_time > self.click_cooldown:
                        try:
                            pyautogui.click()
                            st.toast("Eyebrow Raise - Left Click!")
                            self.last_click_time = current_time
                        except Exception as e:
                            st.error(f"Left-click error: {e}")
                self.last_eyebrow_state = True
            else:
                self.last_eyebrow_state = False
            
            # Right Click - Head Tilt Detection
            head_center_x = landmarks[168].x * width
            if self.last_head_position is not None:
                if abs(head_center_x - self.last_head_position) > self.head_tilt_threshold:
                    if current_time - self.last_click_time > self.click_cooldown:
                        try:
                            pyautogui.rightClick()
                            st.toast("Head Tilt - Right Click!")
                            self.last_click_time = current_time
                        except Exception as e:
                            st.error(f"Right-click error: {e}")
            
            # Update last head position
            self.last_head_position = head_center_x
        except Exception as e:
            st.error(f"Click gesture detection error: {e}")

    def track_head_movement(self, landmarks, frame_shape):
        """
        Track head movement for cursor control with reduced sensitivity
        """
        if not self.gesture_active:
            return
        
        try:
            height, width, _ = frame_shape
            # Use nose landmark for cursor control
            nose_x = landmarks[1].x * width
            nose_y = landmarks[1].y * height
            
            # Convert to screen coordinates with more gradual mapping
            screen_x = np.interp(nose_x, (0, width), (0, self.screen_w))
            screen_y = np.interp(nose_y, (0, height), (0, self.screen_h))
            
            # More aggressive smoothing for slower, more stable cursor movement
            screen_x = (self.prev_x * (self.smoothing-1) + screen_x) / self.smoothing
            screen_y = (self.prev_y * (self.smoothing-1) + screen_y) / self.smoothing
            
            # Move mouse
            pyautogui.moveTo(screen_x, screen_y)
            
            # Update previous coordinates
            self.prev_x, self.prev_y = screen_x, screen_y
        except Exception as e:
            st.error(f"Mouse movement error: {e}")

def main():
    st.title("Advanced Hands-Free Mouse Control")
    
    # Gesture Guide
    st.markdown("""
    ### 🖱️ Gesture Controls
    - **Activate/Deactivate**: Toggle Switch
    - **Move Cursor**: Head Movement (when active)
    - **Left Click**: Eyebrow Raise
    - **Right Click**: Head Tilt
    """)
    
    # Warning and Tips
    st.info("""
    🚨 Tips:
    - Sit in a well-lit area
    - Look directly at the camera
    - Make deliberate movements
    """)
    
    # Initialize controller
    controller = GestureMouseController()
    
    # Ensure face mesh is initialized
    if controller.face_mesh is None:
        st.error("Failed to initialize MediaPipe Face Mesh. Cannot proceed.")
        return
    
    # Gesture activation toggle
    gesture_toggle = st.checkbox("Activate Gesture Control", key="gesture_active")
    controller.gesture_active = gesture_toggle
    
    # Webcam capture
    try:
        cap = cv2.VideoCapture(0)
    except Exception as e:
        st.error(f"Error opening webcam: {e}")
        return
    
    # Validate camera
    if not cap.isOpened():
        st.error("Error: Could not open webcam. Please check your camera connection.")
        return
    
    # Streamlit image display
    frame_placeholder = st.empty()
    
    while gesture_toggle:
        # Read frame
        ret, frame = cap.read()
        
        if not ret:
            st.error("Failed to capture frame")
            break
        
        # Flip frame
        frame = cv2.flip(frame, 1)
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame with MediaPipe Face Mesh
        results = controller.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks and controller.gesture_active:
            # Get first face landmarks
            face_landmarks = results.multi_face_landmarks[0].landmark
            
            # Detect click gestures
            controller.detect_click_gestures(face_landmarks, frame.shape)
            
            # Track head movement
            controller.track_head_movement(face_landmarks, frame.shape)
            
            # Draw face landmarks (optional, for visualization)
            for landmark in face_landmarks:
                h, w, _ = frame.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (cx, cy), 1, (0, 255, 0), -1)
        
        # Display frame
        frame_placeholder.image(frame, channels="BGR")
        
        # Small delay
        time.sleep(0.05)
        
        # Check if checkbox is still checked
        gesture_toggle = st.session_state.gesture_active
    
    # Release resources
    cap.release()

if __name__ == "__main__":
    main()
