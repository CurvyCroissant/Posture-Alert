# Posture Alert

[cite_start]**Posture Alert** is a computer vision application designed to mitigate chronic back pain and Forward Head Posture (FHP) caused by prolonged computer use[cite: 1, 17, 24]. [cite_start]Unlike expensive wearable hardware, this software utilizes a standard webcam and the MediaPipe pose estimation framework to monitor user posture in real-time[cite: 18].

## 🚀 Key Features
* [cite_start]**Real-Time Monitoring:** Live tracking of user posture with immediate audio-visual feedback[cite: 20, 50].
* [cite_start]**Headphone Compatible:** Tracks Eye Level coordinates (Landmarks 2 & 5) instead of ears, ensuring accuracy even when wearing over-ear headphones.
* [cite_start]**Background Noise Filter:** Implements a "Calibration Lock" algorithm that ignores background movement by calculating normalized shoulder width.
* [cite_start]**Privacy Focused:** No video files are saved; all processing occurs in RAM and frames are discarded immediately.
* [cite_start]**Session Analytics:** Exports time-series posture data to CSV via Pandas for post-session analysis[cite: 180].

## 🛠 Tech Stack
* [cite_start]**Language:** Python [cite: 177]
* [cite_start]**Core Library:** MediaPipe (Pose Estimation), OpenCV2 [cite: 179]
* [cite_start]**UI Framework:** Streamlit (for browser-based interface) [cite: 178]
* [cite_start]**Data Handling:** Pandas [cite: 180]

## 🧠 How It Works
[cite_start]The application uses a geometric logic pipeline rather than simple pixel counting to achieve depth invariance (users can sit at different distances).

1.  [cite_start]**Calibration:** The user sets a dynamic baseline for "Good Posture"[cite: 51].
2.  **Slouch Detection:** Calculates the vertical descent of the head using the average Y-coordinate of the eyes. [cite_start]If `vertical_drop > DROP_THRESHOLD` (dynamic offset of ~5%), an alert is triggered[cite: 156, 166].
3.  [cite_start]**Tilt Detection:** Measures shoulder height symmetry (`delta_y_tilt`) to detect lateral leaning[cite: 173].
4.  **Background Filtering:**
    ```python
    # Logic to ignore people in the background
    if current_width < (calibrated_shoulder_width * 0.6):
        ignore_frame = True
    ```
    This ensures that subjects optically smaller than the calibrated user are ignored[cite: 153].