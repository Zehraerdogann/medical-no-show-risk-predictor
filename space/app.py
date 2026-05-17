import os
import joblib
import numpy as np
import pandas as pd
import gradio as gr


# --------------------------------------------------
# Load trained model pipeline
# --------------------------------------------------
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "models",
    "random_forest_no_show_pipeline.joblib"
)

model = joblib.load(MODEL_PATH)


# --------------------------------------------------
# Prediction function
# --------------------------------------------------
def predict_no_show(
    specialty,
    gender,
    disability,
    city,
    appointment_month,
    appointment_year,
    appointment_shift,
    age,
    under_12_years_old,
    over_60_years_old,
    patient_needs_companion,
    average_temp_day,
    average_rain_day,
    max_temp_day,
    max_rain_day,
    rainy_day_before,
    storm_day_before,
    rain_intensity,
    heat_intensity,
    appointment_day_of_week,
    appointment_hour
):
    age_missing = age is None
    weather_missing = any(
        value is None
        for value in [
            average_temp_day,
            average_rain_day,
            max_temp_day,
            max_rain_day
        ]
    )

    input_df = pd.DataFrame([{
        "specialty": specialty,
        "gender": gender,
        "disability": disability,
        "city": city,
        "appointment_month": appointment_month,
        "appointment_year": int(appointment_year),
        "appointment_shift": appointment_shift,
        "age": np.nan if age_missing else float(age),
        "under_12_years_old": int(under_12_years_old),
        "over_60_years_old": int(over_60_years_old),
        "patient_needs_companion": int(patient_needs_companion),
        "average_temp_day": np.nan if average_temp_day is None else float(average_temp_day),
        "average_rain_day": np.nan if average_rain_day is None else float(average_rain_day),
        "max_temp_day": np.nan if max_temp_day is None else float(max_temp_day),
        "max_rain_day": np.nan if max_rain_day is None else float(max_rain_day),
        "rainy_day_before": int(rainy_day_before),
        "storm_day_before": int(storm_day_before),
        "rain_intensity": rain_intensity,
        "heat_intensity": heat_intensity,
        "appointment_day_of_week": appointment_day_of_week,
        "appointment_hour": int(appointment_hour),
        "age_missing": age_missing,
        "weather_missing": weather_missing
    }])

    probability = model.predict_proba(input_df)[0, 1]
    probability_percent = probability * 100

    if probability >= 0.30:
        risk_level = "High Risk"
        recommendation = "Priority reminder recommended."
    elif probability >= 0.25:
        risk_level = "Moderate Risk"
        recommendation = "Reminder recommended."
    else:
        risk_level = "Low Risk"
        recommendation = "Standard follow-up is likely sufficient."

    return (
        f"{probability_percent:.2f}%",
        risk_level,
        recommendation
    )


# --------------------------------------------------
# Gradio interface
# --------------------------------------------------
with gr.Blocks(title="Medical Appointment No-Show Risk Predictor") as demo:
    gr.Markdown(
        """
        # Medical Appointment No-Show Risk Predictor

        This demo estimates the probability that a scheduled medical appointment may result in a no-show.
        The model is based on appointment, patient, and weather-related features.
        """
    )

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Appointment Information")

            specialty = gr.Dropdown(
                choices=[
                    "physiotherapy",
                    "psychotherapy",
                    "speech therapy",
                    "occupational therapy",
                    "pedagogo",
                    "enf",
                    "assist",
                    "sem especialidade",
                    "Missing"
                ],
                value="physiotherapy",
                label="Specialty"
            )

            appointment_month = gr.Dropdown(
                choices=[
                    "jan", "feb", "mar", "april", "may", "june",
                    "july", "aug", "sept", "oct", "nov", "dec"
                ],
                value="sept",
                label="Appointment Month"
            )

            appointment_year = gr.Number(
                value=2021,
                precision=0,
                label="Appointment Year"
            )

            appointment_shift = gr.Dropdown(
                choices=["morning", "afternoon"],
                value="afternoon",
                label="Appointment Shift"
            )

            appointment_day_of_week = gr.Dropdown(
                choices=[
                    "Monday", "Tuesday", "Wednesday", "Thursday",
                    "Friday", "Saturday", "Sunday"
                ],
                value="Thursday",
                label="Day of the Week"
            )

            appointment_hour = gr.Slider(
                minimum=7,
                maximum=18,
                value=13,
                step=1,
                label="Appointment Hour"
            )

        with gr.Column():
            gr.Markdown("### Patient Information")

            gender = gr.Dropdown(
                choices=["F", "M", "I"],
                value="M",
                label="Gender"
            )

            disability = gr.Dropdown(
                choices=["intellectual", "motor", "Missing"],
                value="Missing",
                label="Disability"
            )

            city = gr.Dropdown(
                choices=[
                    "ITAJAÍ",
                    "B. CAMBORIU",
                    "CAMBORIU",
                    "NAVEGANTES",
                    "ITAPEMA",
                    "BOMBINHAS",
                    "PENHA",
                    "PORTO BELO",
                    "BALN. PIÇARRAS",
                    "ILHOTA",
                    "LUIZ ALVES",
                    "MONTENEGRO",
                    "BLUMENAU",
                    "Missing"
                ],
                value="Missing",
                label="City"
            )

            age = gr.Number(
                value=None,
                precision=0,
                label="Age"
            )

            under_12_years_old = gr.Checkbox(
                value=False,
                label="Under 12 Years Old"
            )

            over_60_years_old = gr.Checkbox(
                value=False,
                label="Over 60 Years Old"
            )

            patient_needs_companion = gr.Checkbox(
                value=False,
                label="Patient Needs Companion"
            )

    with gr.Accordion("Weather Information", open=False):
        with gr.Row():
            average_temp_day = gr.Number(
                value=23.7,
                label="Average Temperature of the Day"
            )

            max_temp_day = gr.Number(
                value=23.7,
                label="Maximum Temperature of the Day"
            )

        with gr.Row():
            average_rain_day = gr.Number(
                value=0.2,
                label="Average Rainfall of the Day"
            )

            max_rain_day = gr.Number(
                value=0.2,
                label="Maximum Rainfall of the Day"
            )

        with gr.Row():
            rainy_day_before = gr.Checkbox(
                value=True,
                label="Rainy Day Before"
            )

            storm_day_before = gr.Checkbox(
                value=True,
                label="Storm Day Before"
            )

        with gr.Row():
            rain_intensity = gr.Dropdown(
                choices=["no_rain", "weak", "moderate", "heavy"],
                value="no_rain",
                label="Rain Intensity"
            )

            heat_intensity = gr.Dropdown(
                choices=["mild", "cold", "warm", "heavy_cold", "heavy_warm"],
                value="mild",
                label="Heat Intensity"
            )

    predict_button = gr.Button("Predict No-Show Risk", variant="primary")

    gr.Markdown("## Prediction Result")

    with gr.Row():
        risk_probability_output = gr.Textbox(label="No-Show Probability")
        risk_level_output = gr.Textbox(label="Risk Level")
        recommendation_output = gr.Textbox(label="Suggested Action")

    predict_button.click(
        fn=predict_no_show,
        inputs=[
            specialty,
            gender,
            disability,
            city,
            appointment_month,
            appointment_year,
            appointment_shift,
            age,
            under_12_years_old,
            over_60_years_old,
            patient_needs_companion,
            average_temp_day,
            average_rain_day,
            max_temp_day,
            max_rain_day,
            rainy_day_before,
            storm_day_before,
            rain_intensity,
            heat_intensity,
            appointment_day_of_week,
            appointment_hour
        ],
        outputs=[
            risk_probability_output,
            risk_level_output,
            recommendation_output
        ]
    )


if __name__ == "__main__":
    demo.launch()