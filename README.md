# 🏥 AI Hospital Triage Assistant

> **An NLP-powered hospital triage system that classifies patient symptom descriptions into medical intents using BioClinicalBERT.**

![Python](https://img.shields.io/badge/Python-3.13-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.8-red)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![Streamlit](https://img.shields.io/badge/Streamlit-Web_App-ff4b4b)

---

## 📌 Project Overview

Hospitals receive thousands of patient queries every day. Manually identifying whether a patient needs emergency care, an appointment, a prescription, or hospital information is time-consuming.

This project uses **BioClinicalBERT**, a medical-domain transformer model, to automatically classify a patient's symptom description into the appropriate medical intent.

---

## 📸 Application Screenshots

### Home Page

Patients can enter their symptoms in natural language.

![Home](assets/home.png)

### Prediction Result

The model predicts the patient's medical intent along with its confidence score.

![Prediction](assets/prediction.png)

---

## 🎯 Supported Medical Intents

- 🚨 Emergency
- 🤒 Symptom
- 📅 Appointment
- 💊 Prescription
- 🧪 Lab Report
- 🏥 Hospital Information

---

## ⚙️ Technologies Used

| Category | Technology |
|----------|------------|
| Language | Python |
| NLP Model | BioClinicalBERT |
| Framework | Hugging Face Transformers |
| Deep Learning | PyTorch |
| Data Processing | Pandas, NumPy |
| Deployment | Streamlit |
| Development | Jupyter Notebook |

---

## 🔄 Project Workflow

1. Data Understanding
2. Data Annotation
3. Text Preprocessing
4. Tokenization using BioClinicalBERT
5. Fine-tuning the Transformer Model
6. Intent Prediction
7. Streamlit Deployment

---

## 🌍 Real-World Applications

- AI Hospital Reception Systems
- Telemedicine Platforms
- Medical Chatbots
- Emergency Patient Routing
- Healthcare NLP Solutions

---

## 📁 Repository Structure

```text
AI-Hospital-Triage-Assistant/
│
├── assets/
│   ├── home.png
│   └── prediction.png
│
├── models/
├── notebooks/
├── app.py
├── requirements.txt
└── README.md
```

---

## 👨‍💻 Author

**Saiprasad Nukala**

- GitHub: https://github.com/saiprasad00700
