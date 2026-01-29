# Posture Alert

**Posture Alert** is a webcam-based posture monitoring app that provides real-time feedback to help reduce slouching and lateral leaning. It runs locally using Streamlit and MediaPipe.

## 🚀 Key Features
- **Real-Time Monitoring:** Live posture tracking with audio/visual alerts.
- **Headphone Compatible:** Uses eye landmarks for head position, so over-ear headphones won’t block detection.
- **Background Filter:** Ignores smaller, distant figures using shoulder-width calibration.
- **Privacy Focused:** No video files are saved; all processing happens in memory.
- **Session Analytics:** Logs posture events and exports a CSV report.

## 🛠 Tech Stack
- **Language:** Python
- **Core Libraries:** MediaPipe Tasks (Pose Landmarker), OpenCV
- **UI Framework:** Streamlit
- **Data Handling:** Pandas

## ✅ Setup
1. Run install_requirements.bat (creates .venv and installs dependencies).
2. Run run_app.bat to launch the app.

## ▶️ Run
Open the Streamlit URL shown in the terminal.

## 🧠 How It Works
1. **Calibration:** Sets a baseline eye height for “good posture.”
2. **Slouch Detection:** Computes vertical drop from the baseline and alerts when it exceeds a threshold.
3. **Tilt Detection:** Compares shoulder heights to detect lateral leaning.
4. **Background Filtering:** Ignores background subjects based on relative shoulder width.

## 📦 Model File
On first run, the app downloads a pose model file named model.task (required by MediaPipe Tasks). Keep it in the project folder.