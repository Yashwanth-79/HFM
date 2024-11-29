import streamlit as st
import cv2
import numpy as np
import pyautogui
import time

class GestureMouseController:
    def __init__(self):
        # Load Haar Cascade Classifier for face detection
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        except Exception as e:
            st.error(f"Error loading Haar Cascade: {e}")
            self.face_cascade = None
        
        # Safe initialization of pyautogui
        try:
            pyautogui.FAILSAFE = False
        except Exception as e:
            st.error(f"Error initializing PyAutoGUI: {e}")
            return

        # Screen dimensions
        self.screen_w, self.screen_h = pyautogui.size()
        
        # Tracking parameters
        self.prev_x, self.prev_y = 0, 0
        self.smoothing = 5  # Increased smoothing for slower movement
        
        # Gesture detection variables
        self.last_click_time = 0
        self.click_cooldown = 0.5  # 0.5 seconds between clicks
        
        # Tracking variables for specific gestures
        self.last_eyebrow_state = False
        self.last_head_tilt = 0
        self.head_tilt_threshold = 20  # Pixel difference for head tilt
        
        # Gesture control state
        self.gesture_active = False

    def detect_click_gestures(self, face, gray_frame):
        """
        Detect click gestures using facial features
        """
        current_time = time.time()
        (x, y, w, h) = face
        
        # Region of Interest for eyebrows (upper part of face)
        eyebrow_roi = gray_frame[y:y+h//3, x:x+w]
        
        # Detect eyebrow movement (using intensity variation)
        eyebrow_intensity = np.mean(eyebrow_roi)
        
        # Left Click - Eyebrow Raise Detection
        if eyebrow_intensity > 150:  # Adjust threshold as needed
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
        face_center_x = x + w // 2
        if abs(face_center_x - self.last_head_tilt) > self.head_tilt_threshold:
            if current_time - self.last_click_time > self.click_cooldown:
                try:
                    pyautogui.rightClick()
                    st.toast("Head Tilt - Right Click!")
                    self.last_click_time = current_time
                except Exception as e:
                    st.error(f"Right-click error: {e}")
        
        # Update last head position
        self.last_head_tilt = face_center_x

    def track_head_movement(self, face, frame_shape):
        """
        Track head movement for cursor control with reduced sensitivity
        """
        if not self.gesture_active:
            return
        
        try:
            (x, y, w, h) = face
            # Use face center as cursor reference
            nose_x = x + w // 2
            nose_y = y + h // 2
            
            # Convert to screen coordinates with more gradual mapping
            screen_x = np.interp(nose_x, (0, frame_shape[1]), (0, self.screen_w))
            screen_y = np.interp(nose_y, (0, frame_shape[0]), (0, self.screen_h))
            
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
    
    # System compatibility check
    try:
        import cv2
        import numpy as np
        import pyautogui
    except ImportError as e:
        st.error(f"Required library not found: {e}")
        st.stop()
    
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
    
    # Gesture activation toggle
    gesture_toggle = st.checkbox("Activate Gesture Control", key="gesture_active")
    controller.gesture_active = gesture_toggle
    
    # Webcam capture
    cap = cv2.VideoCapture(0)
    
    # Validate camera
    if not cap.isOpened():
        st.error("Error: Could not open webcam. Please check your camera connection.")
        return
    
    # Streamlit image display
    frame_placeholder = st.empty()
    
    while True:
        # Read frame
        ret, frame = cap.read()
        
        if not ret:
            st.error("Failed to capture frame")
            break
        
        # Flip frame
        frame = cv2.flip(frame, 1)
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        try:
            faces = controller.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(30, 30)
            )
        except Exception as e:
            st.error(f"Face detection error: {e}")
            faces = []
        
        if len(faces) > 0 and controller.gesture_active:
            # Get first face
            face = faces[0]
            
            # Detect click gestures
            controller.detect_click_gestures(face, gray)
            
            # Track head movement
            controller.track_head_movement(face, frame.shape)
            
            # Draw face rectangle
            (x, y, w, h) = face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Display frame
        frame_placeholder.image(frame, channels="BGR")
        
        # Small delay
        time.sleep(0.05)
        
        # Check if user wants to exit
        if not gesture_toggle:
            break
    
    # Release resources
    cap.release()

if __name__ == "__main__":
    main()
