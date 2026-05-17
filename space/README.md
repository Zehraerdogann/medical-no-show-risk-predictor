---
title: Medical No-Show Risk Predictor
emoji: 🩺
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
---

# Medical Appointment No-Show Risk Predictor

This Gradio demo estimates the probability that a scheduled medical appointment may result in a no-show.

The model was trained on appointment, patient, and weather-related features and returns:
- No-show probability
- Risk level
- Suggested follow-up action