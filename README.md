---
title: Face Score
emoji: 🌍
colorFrom: yellow
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

## Face Score – Streamlit

This app analyzes a face image to produce a deep-learning-based score and feature scores (eyes symmetry and nose), then aggregates them into a total score. It preserves the core business logic from the referenced repository while providing a streamlined, modern UI.

### Local run

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Go to Streamlit Community Cloud and create a new app.
3. Set the entrypoint to `streamlit_app.py` and pick your repo/branch.
4. Add the large model files to the repo (already present: `beauty_model_finetuned_export.pkl` and `shape_predictor_68_face_landmarks.dat`).
5. After build, share the public URL.

### Notes

- Uses `mediapipe` FaceMesh for landmark-based features and falls back to `dlib` if available.
- Deep learning score uses the provided FastAI model if it loads successfully; otherwise the UI gracefully degrades to feature-only scoring.
