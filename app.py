import html
import threading
import time
from collections import deque

import av
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
import mediapipe as mp
from streamlit_autorefresh import st_autorefresh
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

st.set_page_config(
    page_title="GestureNet | ASL Recognition",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_PATH = "model/best_mobilenetv2_asl.keras"

# ---- Tuning knobs -----------------------------------------------------------
HOLD_SECONDS = 1.5      # how long a sign must be held to add a letter
CONF_THRESHOLD = 0.60   # minimum model confidence to count a prediction
MIN_VOTES = 5           # votes needed (out of HISTORY_SIZE) to accept a letter
HISTORY_SIZE = 8
USE_HAND_CROP = True    # crop around the hand before classifying (set False to
                        # go back to full-frame input if accuracy gets worse)
CROP_PADDING = 1.35     # how much space to keep around the hand
PREDICT_EVERY_N = 2     # classify every Nth detected hand frame
# -----------------------------------------------------------------------------

CLASS_NAMES = [
    "A", "B", "C", "D", "E", "F", "G",
    "H", "I", "J", "K", "L", "M", "N",
    "O", "P", "Q", "R", "S", "T", "U",
    "V", "W", "X", "Y", "Z",
    "del", "nothing", "space",
]
LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

if "recognized_text" not in st.session_state:
    st.session_state.recognized_text = ""

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@600;700;800&family=Manrope:wght@400;500;600;700&display=swap');

:root {
    --ink: #10242B;
    --muted: #5B6F77;
    --paper: #F2F5F7;
    --card: #FFFFFF;
    --line: #D9E1E5;
    --teal: #0E5A63;
    --teal-soft: #DDEEF0;
    --amber: #F0A32B;
}

html, body, .stApp, [class*="css"] {
    font-family: 'Manrope', system-ui, sans-serif;
}
.stApp { background: var(--paper); color: var(--ink); }
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
.block-container { padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1280px; }

h1, h2, h3, h4, p, label, span, li { color: var(--ink); }

.brand {
    display: flex; align-items: center; justify-content: space-between;
    gap: 16px; flex-wrap: wrap;
    padding-bottom: 18px; margin-bottom: 22px;
    border-bottom: 1px solid var(--line);
}
.brand h1 {
    font-family: 'Bricolage Grotesque', sans-serif;
    font-weight: 800; font-size: 2.2rem; letter-spacing: -0.02em;
    margin: 0; padding: 0; line-height: 1.1;
}
.brand p { margin: 4px 0 0 0; color: var(--muted) !important; font-size: 1rem; }
.pill {
    background: var(--teal-soft); color: var(--teal) !important;
    border-radius: 999px; padding: 8px 16px;
    font-size: 0.88rem; font-weight: 600;
}

h2, h3 {
    font-family: 'Bricolage Grotesque', sans-serif !important;
    font-weight: 700 !important; letter-spacing: -0.01em;
}

div[data-testid="stVideo"], video { border-radius: 14px; }
.cam-hint { color: var(--muted) !important; font-size: 0.92rem; margin-top: 8px; }

.card {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 22px;
}
.letter-tile {
    display: flex; align-items: center; justify-content: center;
    min-height: 190px;
    border-radius: 10px;
    background: var(--teal);
    font-family: 'Bricolage Grotesque', sans-serif;
    font-weight: 800; font-size: 120px; line-height: 1;
    color: #FFFFFF !important;
}
.letter-tile.idle {
    background: var(--teal-soft);
    color: var(--teal) !important;
    font-size: 26px; font-weight: 700; text-align: center; padding: 0 16px;
}
.letter-tile.word { font-size: 44px; }
.hold-label {
    display: flex; justify-content: space-between;
    margin: 18px 0 8px 0; font-size: 0.92rem; font-weight: 600;
}
.hold-label span:last-child { color: var(--muted) !important; font-weight: 500; }
.track { height: 10px; border-radius: 999px; background: var(--line); overflow: hidden; }
.fill { height: 100%; background: var(--amber); border-radius: 999px; }
.status-line { margin-top: 16px; font-size: 0.95rem; color: var(--muted) !important; }

.output {
    min-height: 96px; padding: 20px 22px;
    background: var(--card); border: 1px solid var(--line);
    border-radius: 14px;
    font-family: 'Bricolage Grotesque', sans-serif;
    font-size: 2rem; font-weight: 600; letter-spacing: 0.04em;
    color: var(--ink) !important;
    overflow-wrap: anywhere; white-space: pre-wrap;
}
.output.empty {
    font-family: 'Manrope', sans-serif; font-size: 1.05rem;
    font-weight: 500; letter-spacing: 0; color: var(--muted) !important;
}
.caret { display: inline-block; width: 3px; height: 1.6rem; background: var(--amber);
         margin-left: 3px; vertical-align: text-bottom; }
@media (prefers-reduced-motion: no-preference) {
    .caret { animation: blink 1.1s steps(1) infinite; }
    @keyframes blink { 50% { opacity: 0; } }
}

.stButton > button {
    border-radius: 10px; font-weight: 600; font-family: 'Manrope', sans-serif;
    border: 1px solid var(--line); background: var(--card); color: var(--ink);
    padding: 0.6rem 1rem;
}
.stButton > button:hover { border-color: var(--teal); color: var(--teal); }
.stButton > button:focus-visible { outline: 3px solid var(--amber); outline-offset: 2px; }

.tips { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin: 6px 0 22px 0; }
.tip { border-left: 3px solid var(--teal); padding: 2px 0 2px 14px; }
.tip b { display: block; margin-bottom: 2px; }
.tip span { color: var(--muted) !important; font-size: 0.92rem; }
@media (max-width: 800px) { .tips { grid-template-columns: 1fr; } }

.footer-note { color: var(--muted) !important; font-size: 0.85rem; padding-top: 14px;
               border-top: 1px solid var(--line); margin-top: 28px; }
</style>
""",
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
@st.cache_resource
def load_asl_model():
    m = tf.keras.models.load_model(MODEL_PATH)
    # Warm up once so the first real prediction isn't slow.
    m(np.zeros((1, 224, 224, 3), dtype=np.float32), training=False)
    return m


try:
    model = load_asl_model()
except Exception as error:
    st.error(
        f"The trained model could not be loaded from `{MODEL_PATH}`. "
        f"Check that the file exists. Details: {error}"
    )
    st.stop()


# ----------------------------------------------------------------------------
# Video processing
#   - recv() only draws and hands frames off, so the video stays smooth.
#   - A background thread runs the model and updates the prediction.
# ----------------------------------------------------------------------------
def hand_crop(rgb, landmarks):
    """Square crop around the hand, with padding, clamped to the frame."""
    h, w = rgb.shape[:2]
    xs = [p.x * w for p in landmarks.landmark]
    ys = [p.y * h for p in landmarks.landmark]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    side = max(max(xs) - min(xs), max(ys) - min(ys)) * CROP_PADDING
    side = int(max(side, 96))
    side = min(side, h, w)
    x1 = int(np.clip(cx - side / 2, 0, w - side))
    y1 = int(np.clip(cy - side / 2, 0, h - side))
    return (x1, y1, x1 + side, y1 + side)


class ASLVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.lock = threading.Lock()
        self.history = deque(maxlen=HISTORY_SIZE)
        self.prediction = "Show a hand sign"
        self.confidence = 0.0
        self.stable_label = ""
        self.stable_since = time.time()
        self.latest = None   # newest RGB frame waiting for the worker
        self.points = None   # hand landmarks as normalised (x, y)
        self.box = None      # crop box as normalised (x1, y1, x2, y2)
        self.running = True
        self.last_frame_time = time.time()

        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        )
        self.connections = list(mp.solutions.hands.HAND_CONNECTIONS)

        threading.Thread(target=self._worker, daemon=True).start()

    # ---- background: hand detection + classification -------------------------
    def _worker(self):
        count = 0
        while self.running:
            with self.lock:
                rgb = self.latest
                self.latest = None

            if rgb is None:
                if time.time() - self.last_frame_time > 10:
                    break  # camera stopped; end this thread
                time.sleep(0.01)
                continue

            try:
                results = self.hands.process(rgb)
            except Exception:
                continue

            if not results.multi_hand_landmarks:
                with self.lock:
                    self.points = None
                    self.box = None
                    self.history.clear()
                    self.prediction = "No hand detected"
                    self.confidence = 0.0
                    self.stable_label = ""
                continue

            landmarks = results.multi_hand_landmarks[0]
            h, w = rgb.shape[:2]
            x1, y1, x2, y2 = hand_crop(rgb, landmarks)
            with self.lock:
                self.points = [(p.x, p.y) for p in landmarks.landmark]
                self.box = (x1 / w, y1 / h, x2 / w, y2 / h)

            count += 1
            if count % PREDICT_EVERY_N:
                continue

            try:
                region = rgb[y1:y2, x1:x2] if USE_HAND_CROP else rgb
                crop = cv2.resize(region, (224, 224))
                x = tf.keras.applications.mobilenet_v2.preprocess_input(
                    crop.astype(np.float32)
                )
                probs = model(x[np.newaxis], training=False).numpy()[0]
                index = int(np.argmax(probs))
                label = CLASS_NAMES[index]
                score = float(probs[index])
            except Exception:
                continue

            with self.lock:
                self.history.append(label if score >= CONF_THRESHOLD else "Uncertain")
                self.confidence = score

                valid = [i for i in self.history if i != "Uncertain"]
                if valid:
                    counts = {}
                    for item in valid:
                        counts[item] = counts.get(item, 0) + 1
                    common = max(counts, key=counts.get)

                    if counts[common] >= MIN_VOTES:
                        if common != self.stable_label:
                            self.stable_label = common
                            self.stable_since = time.time()
                        self.prediction = common
                    else:
                        self.prediction = "Hold sign steady"
                else:
                    self.prediction = "Hold sign steady"

    # ---- per-frame: only hand the frame over and draw (very light) -----------
    def recv(self, frame):
        # Work on the ORIGINAL (un-mirrored) frame: the model was trained on
        # un-mirrored images. Only the final display is mirrored.
        image = frame.to_ndarray(format="bgr24")
        h, w = image.shape[:2]
        self.last_frame_time = time.time()

        if self.latest is None:  # worker is free: give it the newest frame
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            with self.lock:
                self.latest = rgb

        with self.lock:
            points = self.points
            box = self.box
            prediction = self.prediction

        if points:
            px = [(int(x * w), int(y * h)) for x, y in points]
            for a, b in self.connections:
                cv2.line(image, px[a], px[b], (255, 255, 255), 3)
            for p in px:
                cv2.circle(image, p, 5, (120, 220, 60), -1)
            if box:
                cv2.rectangle(
                    image,
                    (int(box[0] * w), int(box[1] * h)),
                    (int(box[2] * w), int(box[3] * h)),
                    (43, 163, 240), 2,
                )
            status, status_color = "Hand detected", (150, 200, 40)
        else:
            status, status_color = "Show your hand", (43, 163, 240)

        image = cv2.flip(image, 1)
        cv2.rectangle(image, (10, 10), (w - 10, 78), (43, 36, 16), -1)
        cv2.putText(image, status, (24, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, status_color, 2)
        cv2.putText(image, "Sign: " + prediction, (24, 64),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        return av.VideoFrame.from_ndarray(image, format="bgr24")


# ----------------------------------------------------------------------------
# Header + tips
# ----------------------------------------------------------------------------
st.markdown(
    """
<div class="brand">
  <div>
    <h1> GestureNet</h1>
    <p>Spell in sign language and watch your words appear.</p>
  </div>
  <div class="pill">ASL alphabet · static signs</div>
</div>

<div class="tips">
  <div class="tip"><b>Light your hand</b><span>Face a window or lamp so your fingers are clearly visible.</span></div>
  <div class="tip"><b>Hold each sign</b><span>Keep one letter still for about 1.5 seconds to add it.</span></div>
  <div class="tip"><b>Use one hand</b><span>Keep it centred in the frame, with a plain background if possible.</span></div>
</div>
""",
    unsafe_allow_html=True,
)

RTC_CONFIG = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}

left, right = st.columns([1.5, 1], gap="large")

with left:
    st.subheader("Camera")
    ctx = webrtc_streamer(
        key="gesturenet-camera",
        video_processor_factory=ASLVideoProcessor,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 480},
                "height": {"ideal": 360},
                "frameRate": {"ideal": 15},
            },
            "audio": False,
        },
        async_processing=True,
    )
    st.markdown(
        '<p class="cam-hint">The amber box shows the area sent to the model. '
        "This prototype reads single alphabet signs, not continuous signing.</p>",
        unsafe_allow_html=True,
    )


def render_recognition(prediction, confidence, progress, status_text):
    if len(prediction) == 1:
        tile = f'<div class="letter-tile">{html.escape(prediction)}</div>'
    elif prediction in ("space", "del", "nothing"):
        tile = f'<div class="letter-tile word">{html.escape(prediction)}</div>'
    else:
        tile = f'<div class="letter-tile idle">{html.escape(prediction)}</div>'

    pct = int(max(0.0, min(1.0, progress)) * 100)
    conf = f" · {int(confidence * 100)}% sure" if confidence > 0 else ""
    st.markdown(
        f"""
<div class="card">
  {tile}
  <div class="hold-label"><span>Hold to add</span><span>{pct}%</span></div>
  <div class="track"><div class="fill" style="width:{pct}%"></div></div>
  <div class="status-line">{html.escape(status_text + conf)}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def rerun_app():
    try:
        st.rerun(scope="app")
    except TypeError:  # older Streamlit without fragments
        st.rerun()


def live_panel():
    """Refreshes on its own, without re-running the whole page."""
    processor = ctx.video_processor
    if not processor:
        render_recognition("—", 0.0, 0.0, "Press START under the camera to begin.")
        return

    with processor.lock:
        prediction = processor.prediction
        confidence = processor.confidence
        stable_label = processor.stable_label
        stable_since = processor.stable_since

    now = time.time()
    addable = stable_label in LETTERS + ["space"]
    progress = (now - stable_since) / HOLD_SECONDS if addable else 0.0

    if prediction == "No hand detected":
        message = "Place your hand in front of the camera."
    elif prediction == "Hold sign steady":
        message = "Almost there. Keep the sign still."
    elif prediction in ("del", "nothing"):
        message = "This sign isn't added to your text."
    else:
        message = "Recognised. Keep holding to add it."

    render_recognition(prediction, confidence, progress, message)

    if addable and now - stable_since >= HOLD_SECONDS:
        last_added = st.session_state.get("last_added_label", "")
        last_added_at = st.session_state.get("last_added_at", 0.0)

        if stable_label != last_added or now - last_added_at >= 2.5:
            if stable_label == "space":
                st.session_state.recognized_text += " "
            else:
                st.session_state.recognized_text += stable_label

            st.session_state.last_added_label = stable_label
            st.session_state.last_added_at = now
            with processor.lock:
                processor.stable_since = now + 999999
            rerun_app()


with right:
    st.subheader("Current sign")
    if hasattr(st, "fragment"):
        st.fragment(run_every=0.6)(live_panel)()
    else:
        st_autorefresh(interval=600, key="gesturenet-refresh")
        live_panel()

# ----------------------------------------------------------------------------
# Recognised text
# ----------------------------------------------------------------------------
st.subheader("Your text")

text = st.session_state.recognized_text
if text:
    st.markdown(
        f'<div class="output">{html.escape(text)}<span class="caret"></span></div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="output empty">Letters you sign will appear here.</div>',
        unsafe_allow_html=True,
    )

st.write("")
col1, col2, col3, _ = st.columns([1, 1, 1, 2])

with col1:
    if st.button("Add space", use_container_width=True):
        st.session_state.recognized_text += " "
        st.rerun()

with col2:
    if st.button("Delete last", use_container_width=True):
        st.session_state.recognized_text = st.session_state.recognized_text[:-1]
        st.rerun()

with col3:
    if st.button("Clear all", use_container_width=True):
        st.session_state.recognized_text = ""
        st.rerun()

st.markdown(
    '<div class="footer-note">GestureNet · MobileNetV2, TensorFlow, MediaPipe, '
    "OpenCV and Streamlit</div>",
    unsafe_allow_html=True,
)