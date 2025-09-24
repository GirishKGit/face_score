"""FastAPI backend serving face score predictions for the React client.

The service exposes a JSON API that accepts a base64 encoded image payload and
returns an aggregated face score, confidence estimate, and supporting feature
metrics.  It adapts logic from the original Gradio demo while aligning with
common REST conventions used in open-source face score projects such as
https://github.com/aqeelanwar/Face-Attractiveness-Score.
"""
from __future__ import annotations

import base64
import binascii
import logging
from io import BytesIO
from typing import Dict, Tuple

import cv2
import dlib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastai.vision.all import PILImage, load_learner
from pydantic import BaseModel, Field
from starlette import status


LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Instantiate FastAPI application with helpful metadata for documentation tools.
app = FastAPI(
    title="Face Score API",
    description=(
        "Predict composite face scores and confidence values for uploaded images. "
        "The JSON schema mirrors widely used face scoring APIs for frictionless "
        "integration with web front-ends."
    ),
    version="1.0.0",
)

# Allow the React development server (and other trusted origins) to access the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ScoreRequest(BaseModel):
    """Payload containing the base64 encoded image."""

    image_base64: str = Field(
        ...,
        description="Base64 encoded representation of the image without any metadata prefix.",
        min_length=1,
    )


class ScoreDetails(BaseModel):
    """Detailed feature scores to surface alongside the aggregate result."""

    deep_learning_score: float = Field(..., description="Model predicted attractiveness score on a 1-5 scale.")
    facial_feature_score: float = Field(..., description="Average of eye and nose symmetry metrics (1-5 scale).")
    eyes_score: float = Field(..., description="Eye symmetry score on a 1-5 scale.")
    nose_score: float = Field(..., description="Nose proportion score on a 1-5 scale.")


class ScoreResponse(BaseModel):
    """Structured response returned to the UI."""

    score: float = Field(..., description="Normalized score on a 1-10 scale.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence expressed between 0 and 1.")
    details: ScoreDetails


class ErrorResponse(BaseModel):
    """Standard error payload."""

    error: str


# Load heavy model artefacts once at startup so subsequent requests are snappy.
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
learn = load_learner("beauty_model_finetuned_export.pkl")


@app.get("/health", response_model=Dict[str, str])
def health_check() -> Dict[str, str]:
    """Simple health endpoint used by uptime monitors."""

    return {"status": "ok"}


@app.post(
    "/api/score",
    response_model=ScoreResponse,
    responses={
        status.HTTP_400_BAD_REQUEST: {"model": ErrorResponse},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"model": ErrorResponse},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
    },
)
def generate_score(payload: ScoreRequest) -> ScoreResponse:
    """Decode the provided image and return an attractiveness score payload."""

    LOGGER.info("Received scoring request")
    image = _decode_image(payload.image_base64)

    try:
        deep_learning_score, confidence = _predict_deep_learning_score(image)
        eyes_score, nose_score = _detect_landmarks(image)
    except HTTPException:
        # Let HTTP errors bubble up so the client receives context.
        raise
    except Exception as exc:  # pragma: no cover - defensive logging for unexpected errors
        LOGGER.exception("Unexpected error while processing image")
        raise HTTPException(status_code=500, detail="Unexpected error while processing the image.") from exc

    facial_feature_score = (eyes_score + nose_score) / 2.0
    total_score = (deep_learning_score + facial_feature_score) / 2.0

    LOGGER.info("Successfully generated scores")
    return ScoreResponse(
        score=round(total_score * 2, 2),  # Convert 1-5 scale to a familiar 1-10 scale.
        confidence=round(confidence, 3),
        details=ScoreDetails(
            deep_learning_score=round(deep_learning_score, 2),
            facial_feature_score=round(facial_feature_score, 2),
            eyes_score=round(eyes_score, 2),
            nose_score=round(nose_score, 2),
        ),
    )


def _decode_image(image_base64: str) -> np.ndarray:
    """Decode the incoming base64 string into an RGB image array."""

    try:
        image_bytes = base64.b64decode(image_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        LOGGER.warning("Invalid base64 payload: %s", exc)
        raise HTTPException(status_code=400, detail="Invalid base64-encoded image data.") from exc

    try:
        image = np.array(PILImage.create(BytesIO(image_bytes)))
    except Exception as exc:  # pragma: no cover - PIL specific decoding issues
        LOGGER.warning("Unable to decode image: %s", exc)
        raise HTTPException(status_code=400, detail="Unable to decode the provided image.") from exc

    if image.size == 0:
        raise HTTPException(status_code=400, detail="Empty image supplied.")

    return image


def _detect_landmarks(image: np.ndarray) -> Tuple[float, float]:
    """Detect facial landmarks and derive eye/nose based metrics."""

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = detector(gray)

    if not faces:
        LOGGER.info("No faces detected in the provided image")
        raise HTTPException(status_code=422, detail="No face detected. Please upload a clear, front-facing photo.")

    landmarks = predictor(gray, faces[0])
    points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)])

    eyes_score = _calculate_eyes_score(points)
    nose_score = _calculate_nose_score(points)
    return eyes_score, nose_score


def _calculate_eyes_score(landmarks: np.ndarray) -> float:
    """Estimate eye symmetry using the distance between both eye centres."""

    left_eye = landmarks[36:42]
    right_eye = landmarks[42:48]

    left_centre = left_eye.mean(axis=0)
    right_centre = right_eye.mean(axis=0)

    symmetry_distance = np.linalg.norm(left_centre - right_centre)
    normalized_score = 5.0 - (symmetry_distance / 10.0)
    return float(np.clip(normalized_score, 1.0, 5.0))


def _calculate_nose_score(landmarks: np.ndarray) -> float:
    """Score nose proportions using simple landmark ratios."""

    nose_tip = landmarks[30]
    nose_bridge = landmarks[[27, 28, 29]].mean(axis=0)

    proportion_distance = np.linalg.norm(nose_tip - nose_bridge)
    normalized_score = 5.0 - (proportion_distance / 10.0)
    return float(np.clip(normalized_score, 1.0, 5.0))


def _predict_deep_learning_score(image: np.ndarray) -> Tuple[float, float]:
    """Run the fastai model and derive both the score and confidence."""

    resized = cv2.resize(image, (224, 224))
    pil_image = PILImage.create(resized)

    pred_class, pred_idx, raw_outputs = learn.predict(pil_image)

    score = _resolve_score_from_prediction(pred_class, pred_idx)
    confidence = _resolve_confidence(pred_idx, raw_outputs)

    return score, confidence


def _resolve_score_from_prediction(pred_class, pred_idx) -> float:
    """Convert a fastai prediction tuple into a float score."""

    # Fastai returns the class label as the first element. In many attractiveness
    # models (including the open-source reference repo) labels are stored as
    # strings such as "4.5". We attempt to coerce that into a numeric value.
    if isinstance(pred_class, (int, float, np.integer, np.floating)):
        return float(pred_class)

    try:
        return float(str(pred_class))
    except (TypeError, ValueError):
        LOGGER.warning("Falling back to index-based score for prediction: %s", pred_class)

    if hasattr(pred_idx, "item"):
        return float(pred_idx.item())

    return float(pred_idx)


def _resolve_confidence(pred_idx, raw_outputs) -> float:
    """Extract a confidence estimate from the prediction outputs."""

    try:
        idx = int(pred_idx) if not hasattr(pred_idx, "item") else int(pred_idx.item())
        outputs = raw_outputs.softmax(dim=0) if hasattr(raw_outputs, "softmax") else raw_outputs
        prob = float(outputs[idx])
    except Exception:  # pragma: no cover - best-effort fallback for uncommon tensor shapes
        LOGGER.warning("Unable to derive probability from raw outputs; defaulting confidence to 0.5")
        prob = 0.5

    return float(np.clip(prob, 0.0, 1.0))
