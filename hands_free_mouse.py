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
        
        # Initialize pyautogui
        pyautogui.FAILSAFE = False

        # Screen dimensions
        self.screen_w, self.screen_h = pyautogui.size()
        
        # Tracking parameters
        self.prev_x, self.prev_y = 0, 0
        self.smoothing = 5
        
        # Gesture detection variables
        self.last_click_time = 0
        self.click_cooldown = 0.5

        # Head tilt tracking
        self.last_head_tilt = 0
        self.head_tilt_threshold = 20

    def detect_click_gestures(self, face, gray_frame):
        (x, y, w, h) = face
        current_time = time.time()

        # Detect left-click gesture (e.g., eyebrow raise)
        eyebrow_roi = gray_frame[y:y+h//3, x:x+w]
        eyebrow_intensity = np.mean(eyebrow_roi)
        if eyebrow_intensity > 150 and current_time - self.last_click_time > self.click_cooldown:
            pyautogui.click()
            st.toast("Eyebrow Raise - Left Click!")
            self.last_click_time = current_time

    def track_head_movement(self, face, frame_shape):
        (x, y, w, h) = face
        nose_x = x + w // 2
        nose_y = y + h // 2

        screen_x = np.interp(nose_x, (0, frame_shape[1]), (0, self.screen_w))
        screen_y = np.interp(nose_y, (0, frame_shape[0]), (0, self.screen_h))

        pyautogui.moveTo(screen_x, screen_y)

def main():
    st.title("Hands-Free Mouse Control")
    st.markdown("""
    **Controls:**
    - Move cursor: Head Movement
    - Left Click: Eyebrow Raise
    """)

    controller = GestureMouseController()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Could not access webcam.")
        return

    frame_placeholder = st.empty()
    gesture_active = st.checkbox("Enable Gesture Control", value=True)

    while gesture_active:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture frame.")
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = controller.face_cascade.detectMultiScale(gray, 1.1, 5)

        if len(faces) > 0:
            face = faces[0]
            controller.detect_click_gestures(face, gray)
            controller.track_head_movement(face, frame.shape)
            (x, y, w, h) = face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        frame_placeholder.image(frame, channels="BGR")

    cap.release()

if __name__ == "__main__":
    main()
