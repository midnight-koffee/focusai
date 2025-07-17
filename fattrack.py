import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import av
import mediapipe as mp
import numpy as np
from collections import deque
import time
import openai  # Use the standard openai package
import traceback
import os

# Set OpenAI API key from secrets
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

st.set_page_config(page_title="FocusAI - Fatigue Detection", layout="centered")
st.title("🧠 FocusAI: Real-time Fatigue Detection")
st.markdown("""
### Smart fatigue detection with AI assistant:
- **Real-time eye tracking** for fatigue metrics
- **GPT-powered assistant** with fatigue awareness
- **Personalized suggestions** based on your state
""")
st.warning("Please look straight into the camera with good lighting")

# Setup Mediapipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Constants
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
EAR_THRESHOLD = 0.21
BLINK_THRESHOLD = 0.18
FRAME_WINDOW = 30  # Smoothing window size
PERCLOS_THRESHOLD = 35.0  # 35% of frames with closed eyes indicates fatigue
BLINK_RATE_THRESHOLD = 15  # Blinks per minute

# Initialize session state for chat and fatigue data
if 'messages' not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm your fatigue-aware assistant. I can see your current fatigue status and provide personalized suggestions. How can I help you today?"}
    ]
    
if 'fatigue_data' not in st.session_state:
    st.session_state.fatigue_data = {
        'status': "Focused ✅",
        'perclos': 0.0,
        'blink_rate': 0.0,
        'avg_ear': 0.5,
        'last_update': time.time()
    }

class FatigueDetector(VideoProcessorBase):
    def __init__(self):
        self.ear_buffer = deque(maxlen=FRAME_WINDOW)
        self.blink_counter = 0
        self.eye_closed = False
        self.last_update_time = time.time()
        self.fatigue_status = "Focused"
        self.perclos = 0.0
        self.blink_rate = 0
        self.frame_count = 0
        self.closed_frame_count = 0
        self.last_blink_time = 0
        self.avg_ear = 0.5
        self.blink_log = deque(maxlen=30)
        self.blink_animation = 0

    def calculate_ear(self, landmarks, eye_indices, image_shape):
        h, w = image_shape[:2]
        points = np.array([(landmarks[i].x * w, landmarks[i].y * h) 
                          for i in eye_indices], dtype=np.float32)
        
        # Vertical distances
        dist1 = np.linalg.norm(points[1] - points[5])
        dist2 = np.linalg.norm(points[2] - points[4])
        # Horizontal distance
        dist3 = np.linalg.norm(points[0] - points[3])
        
        ear = (dist1 + dist2) / (2.0 * dist3) if dist3 != 0 else 0.0
        return ear

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(img_rgb)
        
        current_time = time.time()
        self.frame_count += 1

        if self.blink_animation > 0:
            self.blink_animation -= 1

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            # Calculate EAR for both eyes
            left_ear = self.calculate_ear(landmarks, LEFT_EYE_INDICES, img.shape)
            right_ear = self.calculate_ear(landmarks, RIGHT_EYE_INDICES, img.shape)
            self.avg_ear = (left_ear + right_ear) / 2.0
            
            # Apply smoothing
            self.ear_buffer.append(self.avg_ear)
            smoothed_ear = np.mean(self.ear_buffer) if self.ear_buffer else self.avg_ear
            
            # Blink detection
            if smoothed_ear < BLINK_THRESHOLD and not self.eye_closed:
                self.blink_counter += 1
                self.eye_closed = True
                self.last_blink_time = current_time
                self.blink_log.append(current_time)
                self.blink_animation = 10
                
            elif smoothed_ear > EAR_THRESHOLD and self.eye_closed:
                self.eye_closed = False
                
            # Track closed frames for PERCLOS
            if smoothed_ear < EAR_THRESHOLD:
                self.closed_frame_count += 1

            # Update metrics every 3 seconds
            if current_time - self.last_update_time > 3:
                # Calculate blink rate
                recent_blinks = [t for t in self.blink_log if current_time - t <= 15]
                self.blink_rate = len(recent_blinks) * 4
                
                # Calculate PERCLOS
                if self.frame_count > 0:
                    self.perclos = (self.closed_frame_count / self.frame_count) * 100
                
                # Fatigue detection
                if self.perclos > PERCLOS_THRESHOLD or self.blink_rate > BLINK_RATE_THRESHOLD:
                    self.fatigue_status = "⚠️ FATIGUED"
                else:
                    self.fatigue_status = "Focused ✅"
                
                # Update session state
                st.session_state.fatigue_data = {
                    'status': self.fatigue_status,
                    'perclos': self.perclos,
                    'blink_rate': self.blink_rate,
                    'avg_ear': self.avg_ear,
                    'last_update': current_time
                }
                
                # Reset counters
                self.closed_frame_count = 0
                self.frame_count = 0
                self.last_update_time = current_time

        # UI rendering
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        line_type = cv2.LINE_AA
        
        # Semi-transparent background
        overlay = img.copy()
        cv2.rectangle(overlay, (5, 5), (340, 170), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        
        # Display metrics
        cv2.putText(img, f"EAR: {self.avg_ear:.2f}", (15, 40), 
                    font, font_scale, (0, 255, 255), thickness, line_type)
        cv2.putText(img, f"PERCLOS: {self.perclos:.1f}%", (15, 80), 
                    font, font_scale, (255, 200, 0), thickness, line_type)
        cv2.putText(img, f"Blink Rate: {self.blink_rate:.1f}/min", (15, 120), 
                    font, font_scale, (200, 100, 255), thickness, line_type)
        
        # Status indicator
        status_color = (0, 255, 0) if "Focused" in self.fatigue_status else (0, 0, 255)
        cv2.putText(img, f"Status: {self.fatigue_status}", (15, 160), 
                    font, 0.9, status_color, thickness, line_type)
        
        # Visual blink indicator
        if self.blink_animation > 0:
            cv2.putText(img, "BLINK DETECTED!", (img.shape[1]//2 - 120, 50), 
                        font, 1.2, (0, 200, 255), thickness, line_type)
        
        # Draw eye landmarks
        if results.multi_face_landmarks:
            for idx in LEFT_EYE_INDICES + RIGHT_EYE_INDICES:
                landmark = results.multi_face_landmarks[0].landmark[idx]
                x = int(landmark.x * img.shape[1])
                y = int(landmark.y * img.shape[0])
                color = (0, 200, 255) if self.avg_ear < EAR_THRESHOLD else (0, 255, 0)
                cv2.circle(img, (x, y), 2, color, -1)
                
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Start webcam stream
webrtc_ctx = webrtc_streamer(
    key="fatigue-detector",
    video_processor_factory=FatigueDetector,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)

# AI Assistant Section
st.divider()
st.subheader("🤖 FocusAI Assistant")
st.caption("This AI assistant is fatigue-aware. It can provide personalized suggestions based on your current state.")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Fatigue-aware GPT function with model fallback
def get_fatigue_aware_response(prompt):
    """Get AI response considering current fatigue state"""
    try:
        # Prepare context with current fatigue data
        fatigue_context = (
            f"Current user fatigue status: {st.session_state.fatigue_data['status']}\n"
            f"PERCLOS (eye closure percentage): {st.session_state.fatigue_data['perclos']}%\n"
            f"Blink rate: {st.session_state.fatigue_data['blink_rate']} blinks per minute\n"
            f"User message: {prompt}"
        )
        
        # System prompt with fatigue awareness
        system_prompt = (
            "You are a helpful, empathetic assistant specialized in fatigue management. "
            "You can see the user's current fatigue metrics and should respond accordingly. "
            "When the user seems fatigued (high PERCLOS or blink rate), suggest breaks or relaxation techniques. "
            "For general questions about fatigue, provide scientific explanations. "
            "Keep responses concise (1-3 sentences) and supportive."
        )
        
        # Try different models with fallback
        models_to_try = [
            "gpt-4-turbo",  # Latest model
            "gpt-4",        # Standard GPT-4
            "gpt-3.5-turbo" # Fallback to 3.5 if others fail
        ]
        
        response = None
        last_error = None
        
        for model_name in models_to_try:
            try:
                response = openai.ChatCompletion.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": fatigue_context}
                    ],
                    temperature=0.7,
                    max_tokens=256
                )
                break  # Exit loop if successful
            except Exception as e:
                last_error = e
                continue
        
        if response:
            return response.choices[0].message.content.strip()
        else:
            st.error(f"All models failed: {str(last_error)}")
            return "I'm having trouble connecting to the assistant. Please try again later."
    
    except Exception as e:
        st.error(f"Error getting AI response: {str(e)}")
        st.text(traceback.format_exc())  # Show detailed traceback
        return "I encountered an unexpected error. Please try again."

# React to user input
if prompt := st.chat_input("Ask me about fatigue or request a break suggestion"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate AI response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Get assistant response
        assistant_response = get_fatigue_aware_response(prompt)
        
        # Simulate typing effect
        for chunk in assistant_response.split():
            full_response += chunk + " "
            time.sleep(0.05)
            message_placeholder.markdown(full_response + "▌")
        
        message_placeholder.markdown(full_response)
    
    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# Predefined suggestion buttons
st.markdown("### Quick Suggestions")
col1, col2, col3 = st.columns(3)
if col1.button("😴 I'm feeling tired"):
    if "messages" in st.session_state:
        st.session_state.messages.append({"role": "user", "content": "I'm feeling tired, what should I do?"})
    st.rerun()
    
if col2.button("💤 Break suggestion"):
    if "messages" in st.session_state:
        st.session_state.messages.append({"role": "user", "content": "Suggest a quick break activity"})
    st.rerun()
    
if col3.button("🧠 Explain fatigue science"):
    if "messages" in st.session_state:
        st.session_state.messages.append({"role": "user", "content": "Explain the science behind fatigue detection"})
    st.rerun()

# Footer
st.divider()
st.caption("FocusAI v3.2 | Real-time fatigue monitoring | For optimal results, ensure your eyes are clearly visible")