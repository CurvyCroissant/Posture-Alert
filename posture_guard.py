import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
import pandas as pd
import os
import platform
from datetime import datetime

# MediaPipe Pose Landmarks
# Index 2 = Left Eye, Index 5 = Right Eye
LEFT_EYE = 2
RIGHT_EYE = 5
# Index 11 = Left Shoulder, Index 12 = Right Shoulder
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12

# Thresholds
TILT_THRESHOLD_Y = 0.05
DROP_THRESHOLD = 0.03      
ABSOLUTE_MIN_WIDTH = 0.1 

# Sound Settings
ALERT_COOLDOWN_SECONDS = 3.0

def play_beep():
    """Plays a system sound based on the OS."""
    system_name = platform.system()
    try:
        if system_name == "Windows":
            import winsound
            winsound.Beep(1000, 500)
        elif system_name == "Darwin":  # macOS
            os.system('say "Alert"') 
        else: # Linux / Other
            print('\a')
    except Exception:
        pass

# Initialization Phase
if 'monitoring_active' not in st.session_state:
    st.session_state.monitoring_active = False
if 'log_data' not in st.session_state:
    st.session_state.log_data = []
if 'last_alert_time' not in st.session_state:
    st.session_state.last_alert_time = 0
if 'baseline_eye_y' not in st.session_state:
    st.session_state.baseline_eye_y = None
if 'calibrated_shoulder_width' not in st.session_state:
    st.session_state.calibrated_shoulder_width = None
if 'latest_landmarks' not in st.session_state:
    st.session_state.latest_landmarks = None
if 'show_report' not in st.session_state:
    st.session_state.show_report = False

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# The Logic Core of the program

def calculate_metrics(landmarks, baseline_eye_y):
    if not landmarks:
        return False, 0, 0, 0, "No Detection", {}

    # Find the average Y of Eyes
    eye_y_avg = (landmarks[LEFT_EYE].y + landmarks[RIGHT_EYE].y) / 2
    
    # For Lateral ing to find the: (Shoulder Difference)
    delta_y_tilt = abs(landmarks[LEFT_SHOULDER].y - landmarks[RIGHT_SHOULDER].y)
    
    vertical_drop = 0.0
    is_dropping = False
    drop_text = "Not Calibrated"
    
    if baseline_eye_y is not None:
        vertical_drop = eye_y_avg - baseline_eye_y
        if vertical_drop > DROP_THRESHOLD:
            is_dropping = True
        drop_text = f"{vertical_drop:.3f}"

    is_tilting = delta_y_tilt > TILT_THRESHOLD_Y
    is_good = not (is_tilting or is_dropping)
    
    feedback_parts = []
    if baseline_eye_y is None:
        feedback_parts.append("Please Calibrate")
    elif is_dropping: 
        feedback_parts.append("Sitting Too Low")
    if is_tilting: 
        feedback_parts.append("Leaning Sideways")
    
    if not feedback_parts:
        feedback = "Good Posture"
    else:
        feedback = "ALERT: " + ", ".join(feedback_parts)
        
    metrics_display = {
        "Vertical Drop": drop_text,
        "Lateral Tilt (Delta Y)": f"{delta_y_tilt:.3f}",
        "Status": feedback
    }
        
    return is_good, vertical_drop, delta_y_tilt, feedback, metrics_display

# UI Design using Streamlit
st.set_page_config(page_title="Posture Alert", layout="wide")
st.title("Posture Alert")

if st.session_state.baseline_eye_y is None:
    st.markdown("**Status:** Waiting for Calibration")
else:
    st.markdown("**Status:** Monitoring")

col1, col2 = st.columns([0.7, 0.3])

with col2:
    st.header("Controls")
    
    if not st.session_state.monitoring_active:
        st.info("1. Start the camera to begin.")
        if st.button("Start Camera", type="primary"):
            st.session_state.monitoring_active = True
            st.session_state.log_data = []
            st.session_state.show_report = False
            st.rerun() 
        
        if st.session_state.log_data:
            st.write("---")
            if st.button("View Session Report"):
                st.session_state.show_report = True
                st.rerun()

    else:
        st.info("2. Sit up straight in your ideal posture.")
        
        if st.button("CALIBRATE Set Baseline"):
            if st.session_state.latest_landmarks:
                lms = st.session_state.latest_landmarks
                
                # Calibrate based on EYES
                current_eye_y = (lms[LEFT_EYE].y + lms[RIGHT_EYE].y) / 2
                
                # Add offset (0.05) for test case when the user is looking below
                st.session_state.baseline_eye_y = current_eye_y + 0.05
                
                # To lock the shoulder width
                current_width = abs(lms[LEFT_SHOULDER].x - lms[RIGHT_SHOULDER].x)
                st.session_state.calibrated_shoulder_width = current_width
                
                st.toast(f"Calibrated! Baseline Y: {st.session_state.baseline_eye_y:.2f}")
                st.rerun()
            else:
                st.error("No person detected yet. Please wait for camera to load.")
        
        st.divider()
        
        if st.button("Stop Camera"):
            st.session_state.monitoring_active = False
            st.rerun() 
    
    st.divider()
    
    if st.button("Exit Program"):
        st.warning("Closing application...")
        time.sleep(1)
        os._exit(0)
    
    st.divider()
    
    st.subheader("Live Metrics")
    metric_drop = st.empty()
    metric_tilt = st.empty()
    status_indicator = st.empty()
    
    if st.session_state.baseline_eye_y is None:
        st.warning("[!] Not Calibrated")
    else:
        st.success(f"[OK] Calibrated (Y: {st.session_state.baseline_eye_y:.2f})")

# Main Function
if st.session_state.monitoring_active:
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("Camera not found! Please check your connection.")
        st.session_state.monitoring_active = False
    else:
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            st_frame = col1.empty()
            
            while st.session_state.monitoring_active and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.write("Failed to capture image")
                    break
                
                # Processing the image by converting the image color using cv2 cvtColor
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = pose.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                if results.pose_landmarks:
                    lms = results.pose_landmarks.landmark
                    st.session_state.latest_landmarks = lms
                    
                    # To filter out background (test case when someone is behind the user)
                    current_width = abs(lms[LEFT_SHOULDER].x - lms[RIGHT_SHOULDER].x)
                    ignore_frame = False
                    
                    if current_width < ABSOLUTE_MIN_WIDTH:
                        ignore_frame = True
                    if st.session_state.calibrated_shoulder_width is not None:
                        if current_width < (st.session_state.calibrated_shoulder_width * 0.6):
                            ignore_frame = True
                            
                    if ignore_frame:
                        cv2.putText(image, "Ignoring Background", (10, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        st_frame.image(image, channels="BGR", use_container_width=True)
                        time.sleep(0.03)
                        continue

                    # The Metrics and for Alerts
                    is_good, drop_val, d_y, text, metrics_txt = calculate_metrics(lms, st.session_state.baseline_eye_y)
                    
                    # Only alert/log if calibrated
                    if st.session_state.baseline_eye_y is not None:
                        if is_good:
                            color = (0, 255, 0)
                            status_indicator.success(f"[OK] {text}")
                        else:
                            color = (0, 0, 255)
                            status_indicator.error(f"[!] {text}")
                            
                            current_time = time.time()
                            if current_time - st.session_state.last_alert_time > ALERT_COOLDOWN_SECONDS:
                                play_beep()
                                st.toast("Alert!")
                                st.session_state.last_alert_time = current_time
                        
                        st.session_state.log_data.append({
                            "Timestamp": datetime.now(),
                            "Vertical_Drop": drop_val,
                            "Tilt_Delta_Y": d_y,
                            "Status": text
                        })
                    else:
                        color = (255, 255, 0)
                        status_indicator.info("Please click [CALIBRATE] to start monitoring.")
                    
                    metric_drop.metric("Vertical Drop", metrics_txt["Vertical Drop"])
                    metric_tilt.metric("Lateral Tilt", metrics_txt["Lateral Tilt (Delta Y)"])
                    
                    # To draw visuals for the csv
                    h, w, c = image.shape
                    if st.session_state.baseline_eye_y is not None:
                        base_y_px = int(st.session_state.baseline_eye_y * h)
                        cv2.line(image, (0, base_y_px), (w, base_y_px), (0, 255, 0), 2)
                        cv2.putText(image, "Baseline Height", (10, base_y_px - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                    mp_drawing.draw_landmarks(
                        image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=2)
                    )

                st_frame.image(image, channels="BGR", use_container_width=True)
                time.sleep(0.03)

        cap.release()

# To generate the report after the session
if st.session_state.show_report and st.session_state.log_data:
    st.divider()
    st.header("Session Analysis Report")
    
    df = pd.DataFrame(st.session_state.log_data)
    
    total_frames = len(df)
    if total_frames > 1:
        start_time = df['Timestamp'].iloc[0]
        end_time = df['Timestamp'].iloc[-1]
        duration_seconds = (end_time - start_time).total_seconds()
        
        alert_count = df[df['Status'].str.contains("ALERT", na=False)].shape[0]
        good_count = total_frames - alert_count
        posture_score = (good_count / total_frames) * 100
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Session Duration", f"{duration_seconds:.1f} sec")
        c2.metric("Posture Score", f"{posture_score:.1f}%")
        c3.metric("Total Slouches", f"{alert_count}")
        
        st.divider()
        
        c_chart1, c_chart2 = st.columns(2)
        with c_chart1:
            st.subheader("Posture Trends")
            st.line_chart(df[["Vertical_Drop", "Tilt_Delta_Y"]])
            st.caption("Lower values are better.")
        with c_chart2:
            st.subheader("Issue Distribution")
            status_counts = df['Status'].value_counts()
            st.bar_chart(status_counts)
            st.caption("Frequency of specific posture issues.")
            
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Detailed CSV Report",
            data=csv,
            file_name="posture_session_report.csv",
            mime="text/csv"
        )
    else:
        st.info("Analysis skipped: Session was too short.")