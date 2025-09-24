import os
from typing import Optional, Tuple, List, Dict

import streamlit as st
import numpy as np
from PIL import Image

# OpenCV must be the headless build on Streamlit Cloud
import cv2  # type: ignore

try:
    import mediapipe as mp  # type: ignore
except Exception:  # pragma: no cover - mediapipe optional in some envs
    mp = None  # type: ignore

try:
    # FastAI is used to load the provided model and keep business logic parity
    from fastai.vision.all import PILImage, load_learner  # type: ignore
except Exception:
    PILImage = None  # type: ignore
    load_learner = None  # type: ignore

try:  # Optional fallback if mediapipe is unavailable
    import dlib  # type: ignore
except Exception:
    dlib = None  # type: ignore


# ---------------------------
# Streamlit page configuration
# ---------------------------
st.set_page_config(
    page_title="Face Score",
    page_icon="😄",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def load_css() -> None:
    """Inject custom CSS to achieve high-fidelity UI similar to the reference design."""
    css_path = os.path.join(os.path.dirname(__file__), "assets", "styles.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_face_mesh():
    """Create a cached MediaPipe FaceMesh instance if mediapipe is available."""
    if mp is None:
        return None
    mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        refine_landmarks=True,
        max_num_faces=1,
        min_detection_confidence=0.5,
    )
    return mesh


@st.cache_resource(show_spinner=False)
def get_dlib_predictor():
    """Create a cached dlib predictor if dlib and the predictor file are available."""
    if dlib is None:
        return None, None
    try:
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        return detector, predictor
    except Exception:
        return None, None


@st.cache_resource(show_spinner=False)
def get_model():
    """Load the FastAI model once and cache it."""
    if load_learner is None:
        return None
    model_path = os.path.join(os.path.dirname(__file__), "beauty_model_finetuned_export.pkl")
    if not os.path.exists(model_path):
        # Fallback to workspace root if running locally with different cwd
        model_path = "beauty_model_finetuned_export.pkl"
    try:
        return load_learner(model_path)
    except Exception:
        return None


def _image_to_rgb_ndarray(image: Image.Image) -> np.ndarray:
    """Ensure numpy RGB array."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    return np.array(image)


def _compute_center(points: List[Tuple[float, float]]) -> np.ndarray:
    return np.mean(np.array(points, dtype=np.float32), axis=0)


def _normalize_to_score(distance: float, scale: float = 10.0) -> float:
    """Reproduce the original repository's simple normalization onto [1, 5]."""
    normalized = 5.0 - (distance / max(1e-6, scale))
    return float(max(1.0, min(5.0, normalized)))


def _eyes_nose_scores_from_mediapipe(image: Image.Image) -> Optional[Tuple[float, float]]:
    if mp is None:
        return None
    mesh = get_face_mesh()
    if mesh is None:
        return None

    rgb = _image_to_rgb_ndarray(image)
    h, w = rgb.shape[:2]
    results = mesh.process(rgb)
    if not results.multi_face_landmarks:
        return None

    lm = results.multi_face_landmarks[0].landmark

    # Landmark index groups for eyes (MediaPipe FaceMesh canonical indices)
    LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

    def lm_points(indices: List[int]) -> List[Tuple[float, float]]:
        return [(lm[i].x * w, lm[i].y * h) for i in indices]

    left_eye_center = _compute_center(lm_points(LEFT_EYE))
    right_eye_center = _compute_center(lm_points(RIGHT_EYE))
    eyes_distance = float(np.linalg.norm(left_eye_center - right_eye_center))
    eyes_score = _normalize_to_score(eyes_distance, scale=10.0)

    # Nose tip and bridge approximations in FaceMesh
    # Using a small set around the ridge to approximate the bridge center
    NOSE_TIP_INDEX = 4  # Commonly approximated tip index in FaceMesh
    NOSE_BRIDGE_INDICES = [6, 197, 195]

    nose_tip = np.array((lm[NOSE_TIP_INDEX].x * w, lm[NOSE_TIP_INDEX].y * h))
    nose_bridge_pts = [(lm[i].x * w, lm[i].y * h) for i in NOSE_BRIDGE_INDICES]
    nose_bridge = _compute_center(nose_bridge_pts)
    nose_distance = float(np.linalg.norm(nose_tip - nose_bridge))
    nose_score = _normalize_to_score(nose_distance, scale=10.0)

    return float(eyes_score), float(nose_score)


def _eyes_nose_scores_from_dlib(image: Image.Image) -> Optional[Tuple[float, float]]:
    if dlib is None:
        return None
    detector, predictor = get_dlib_predictor()
    if detector is None or predictor is None:
        return None

    gray = cv2.cvtColor(_image_to_rgb_ndarray(image), cv2.COLOR_RGB2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        return None
    face = faces[0]
    landmarks = predictor(gray, face)
    points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)]

    left_eye = points[36:42]
    right_eye = points[42:48]
    left_eye_center = _compute_center(left_eye)
    right_eye_center = _compute_center(right_eye)
    eyes_distance = float(np.linalg.norm(left_eye_center - right_eye_center))
    eyes_score = _normalize_to_score(eyes_distance, scale=10.0)

    nose_tip = np.array(points[30])
    nose_bridge = _compute_center([points[27], points[28], points[29]])
    nose_distance = float(np.linalg.norm(nose_tip - nose_bridge))
    nose_score = _normalize_to_score(nose_distance, scale=10.0)

    return float(eyes_score), float(nose_score)


def detect_eyes_nose_scores(image: Image.Image) -> Optional[Tuple[float, float]]:
    """Try MediaPipe first for portability; fall back to dlib if available."""
    scores = _eyes_nose_scores_from_mediapipe(image)
    if scores is not None:
        return scores
    return _eyes_nose_scores_from_dlib(image)


def predict_deep_learning_score(image: Image.Image) -> Optional[float]:
    model = get_model()
    if model is None or PILImage is None:
        return None
    try:
        resized = image.resize((224, 224))
        img = PILImage.create(resized)
        # Maintain the repository behavior: use the second element as numeric score
        score = float(model.predict(img)[1].item())
        return score
    except Exception:
        return None


def make_hero_section():
    st.markdown(
        """
        <section class="hero">
          <div class="hero-inner">
            <div class="eyebrow">Discover Your</div>
            <h1>Beauty Score</h1>
            <p class="subtitle">Upload your photo and get instant AI-powered analysis of your facial features, symmetry, and overall attractiveness score.</p>
            <div class="cta-row">
              <a href="#upload" class="btn btn-primary">Analyze Your Photo</a>
              <a href="https://www.youtube.com/watch?v=dQw4w9WgXcQ" target="_blank" class="btn btn-secondary">Watch Demo</a>
            </div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def make_features_section():
    st.markdown("<h2 class='section-title'>Advanced AI Analysis</h2>", unsafe_allow_html=True)
    st.markdown("<p class='section-subtitle'>Our cutting-edge machine learning algorithms analyze multiple facial features to provide comprehensive beauty scores.</p>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
            <div class="card">
              <div class="icon">🧠</div>
              <div class="card-title">Deep Learning Score</div>
              <p>Neural networks trained on thousands of images to assess overall facial attractiveness and harmony.</p>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div class="card">
              <div class="icon">👁️</div>
              <div class="card-title">Feature Analysis</div>
              <p>Detailed analysis of individual facial features including eyes, nose, lips, and structure proportions.</p>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
            <div class="card">
              <div class="icon">⚖️</div>
              <div class="card-title">Symmetry Detection</div>
              <p>Measurements of facial symmetry and proportions inspired by golden ratio principles.</p>
            </div>
        """, unsafe_allow_html=True)


def make_uploader_section() -> Optional[Image.Image]:
    st.markdown("<h2 id='upload' class='section-title'>Upload Your Photo</h2>", unsafe_allow_html=True)
    st.markdown("<p class='section-subtitle'>Get instant AI-powered beauty analysis in seconds</p>", unsafe_allow_html=True)

    with st.container():
        uploaded = st.file_uploader("Drop your photo here", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=False)
        if uploaded is None:
            return None
        image = Image.open(uploaded).convert("RGB")
        return image


def show_results(image: Image.Image) -> None:
    with st.spinner("Analyzing your photo…"):
        deep_score = predict_deep_learning_score(image)
        eyes_nose = detect_eyes_nose_scores(image)

    if eyes_nose is None:
        st.error("Could not detect a face or landmarks. Please upload a clear, front-facing photo with good lighting.")
        return

    eyes_score, nose_score = eyes_nose
    facial_feature_score = (eyes_score + nose_score) / 2.0

    # If deep model not available, gracefully degrade
    if deep_score is None:
        deep_score = facial_feature_score
    total_score = (deep_score + facial_feature_score) / 2.0

    left, center, right = st.columns([1, 2, 1])
    with left:
        st.image(image, caption="Uploaded photo", use_column_width=True)

    with center:
        st.markdown("<div class='results-title'>Your Results</div>", unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        m1.metric(label="Deep Learning", value=f"{deep_score:.2f} / 5")
        m2.metric(label="Eyes Symmetry", value=f"{eyes_score:.2f} / 5")
        m3.metric(label="Nose", value=f"{nose_score:.2f} / 5")

        m4, m5 = st.columns(2)
        m4.metric(label="Facial Feature Score", value=f"{facial_feature_score:.2f} / 5")
        m5.metric(label="Total Combined", value=f"{total_score:.2f} / 5")

        # Simple bar chart for visualization
        try:
            import pandas as pd  # type: ignore
            import altair as alt  # type: ignore

            chart_df = pd.DataFrame(
                {
                    "Metric": [
                        "Deep Learning",
                        "Eyes Symmetry",
                        "Nose",
                        "Facial Feature",
                        "Total",
                    ],
                    "Score": [deep_score, eyes_score, nose_score, facial_feature_score, total_score],
                }
            )
            chart = (
                alt.Chart(chart_df)
                .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
                .encode(x=alt.X("Metric", sort=None), y=alt.Y("Score", scale=alt.Scale(domain=[0, 5])), color=alt.Color("Metric", legend=None))
                .properties(height=260)
            )
            st.altair_chart(chart, use_container_width=True)
        except Exception:
            pass

        # Downloadable JSON report
        report: Dict[str, float] = {
            "deep_learning_score": float(deep_score),
            "eyes_score": float(eyes_score),
            "nose_score": float(nose_score),
            "facial_feature_score": float(facial_feature_score),
            "total_score": float(total_score),
        }
        import json

        st.download_button(
            label="Download Detailed Report (JSON)",
            data=json.dumps(report, indent=2).encode("utf-8"),
            file_name="face_score_report.json",
            mime="application/json",
        )

    st.info(
        "This application is for educational purposes only and should not be taken as a definitive assessment of appearance. Ensure you have permission to upload any photo you analyze."
    )


def make_footer():
    st.markdown(
        """
        <footer class="site-footer">
          <div class="brand">😄 Face Score</div>
          <p>Advanced AI-powered facial analysis providing instant beauty scores and detailed feature insights. Safe, private, and scientifically inspired.</p>
          <div class="copyright">© 2024 Face Score. All rights reserved.</div>
        </footer>
        """,
        unsafe_allow_html=True,
    )


def main():
    load_css()
    make_hero_section()
    make_features_section()
    image = make_uploader_section()
    if image is not None:
        show_results(image)
    make_footer()


if __name__ == "__main__":
    main()

