import streamlit as st
import torch
import joblib

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

# ------------------------------------
# Page Configuration
# ------------------------------------
st.set_page_config(
    page_title="AI Hospital Triage Assistant",
    page_icon="🏥",
    layout="centered"
)

# ------------------------------------
# Load Saved Model
# ------------------------------------
MODEL_PATH = "models/bioclinical_model"
LABEL_PATH = "models/label_map.pkl"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

label_map = joblib.load(LABEL_PATH)

device = torch.device("cpu")
model.to(device)
model.eval()

# ------------------------------------
# Prediction Function
# ------------------------------------
MAX_LEN = 128

def predict_intent(text):

    encoding = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        predicted_class = torch.argmax(outputs.logits, dim=1).item()

        probabilities = torch.softmax(outputs.logits, dim=1)
        confidence = torch.max(probabilities).item()

    return label_map[predicted_class], confidence

# ------------------------------------
# Hospital Interface
# ------------------------------------
st.title("🏥 AI Hospital Triage Assistant")

st.markdown("""
This AI assistant classifies a patient's medical query into one of the following intents:

- 📅 Appointment
- 🚨 Emergency
- 🏥 Hospital Information
- 🧪 Lab Report
- 💊 Prescription
- 🤒 Symptom
""")

st.divider()

user_input = st.text_area(
    "Enter the patient's symptoms or medical query:",
    placeholder="Example: I have severe chest pain and difficulty breathing",
    height=150
)

# ------------------------------------
# Predict Button
# ------------------------------------
if st.button("🔍 Predict Triage Intent", use_container_width=True):

    if user_input.strip() == "":
        st.warning("Please enter the patient's symptoms or medical query.")

    else:
        intent, confidence = predict_intent(user_input)

        st.divider()

        st.subheader("Prediction Result")

        # Color-coded result
        if intent == "Emergency":
            st.error(f"🚨 Predicted Intent: {intent}")

        elif intent == "Appointment":
            st.success(f"📅 Predicted Intent: {intent}")

        elif intent == "Prescription":
            st.success(f"💊 Predicted Intent: {intent}")

        elif intent == "Lab_Report":
            st.info(f"🧪 Predicted Intent: {intent}")

        elif intent == "Hospital_Info":
            st.info(f"🏥 Predicted Intent: {intent}")

        else:
            st.success(f"🤒 Predicted Intent: {intent}")

        st.metric(
            label="Model Confidence",
            value=f"{confidence * 100:.2f}%"
        )

# ------------------------------------
# Sidebar
# ------------------------------------
with st.sidebar:

    st.header("🏥 AI Triage")

    st.write("**Model:** BioClinicalBERT")

    st.write("**Supported Intents:**")

    st.markdown("""
    - 📅 Appointment
    - 🚨 Emergency
    - 🏥 Hospital_Info
    - 🧪 Lab_Report
    - 💊 Prescription
    - 🤒 Symptom
    """)

    st.divider()

    st.caption("Developed as an NLP Major Project")


# ------------------------------------
# Footer
# ------------------------------------
st.divider()

st.caption(
    "⚠️ This AI assistant is for educational purposes and should not replace professional medical advice."
)