# 🏥 AI-Powered Mental Health Diagnostic System
### Automated Clinical Assessment using Machine Learning (DASS-42)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Library](https://img.shields.io/badge/Library-Gradio%20%7C%20Scikit--learn-orange)
![Status](https://img.shields.io/badge/Status-Prototype%20%2F%20Deployed-green)

---

## Overview
This repository contains an AI-powered diagnostic system that predicts severity levels of Depression, Anxiety, and Stress using the DASS-42 questionnaire. The system accepts user responses, runs inference with a pre-trained model, and generates a professional clinical-style report card (including a verified stamp and signature images).

Key points:
- Primary language & tools: Python, Scikit-learn, Gradio, Pillow (PIL)
- Model file: `mental_health_models.pkl` (large — do not upload to GitHub)
- Author: Md. Nazmus Sakib

---

## Key Features
- Interactive UI built with Gradio for quick, user-friendly assessments
- Verified report generation (downloadable image/pdf with stamp & signature)
- Class imbalance handling via SMOTETomek during preprocessing
- Real-time assessment using 22 clinical questions derived from DASS-42
- Model interpretability support (optional integration with SHAP / LIME)

---

## Folder Structure
```text
Mental_Health_Project/
│
├── img/                        # Project assets (stamp, signature)
│   ├── verified.png
│   └── signature.png
│
├── venv/                       # Virtual environment (local only)
├── app.py                      # Main application script (Gradio UI + inference)
├── mental_health_models.pkl    # Trained model (DO NOT upload to GitHub)
├── requirements.txt            # Python dependencies
├── .gitignore                  # Files to be ignored by Git
└── README.md                   # Project documentation (this file)
```

---

## .gitignore (Recommended)
Make sure your repository ignores environment files and large model artifacts. Example entries:
```text
# Python cache files
__pycache__/
*.py[cod]
*$py.class

# Virtual Environment
venv/
env/
.env

# VS Code settings
.vscode/

# Large Model Files
*.pkl

# MacOS files
.DS_Store
```

---

## Dataset & Training (Summary)
- Dataset: DASS-42 (Depression Anxiety Stress Scales)
- Preprocessing steps:
  - Remove or impute null values and clean the dataset
  - Apply SMOTETomek to handle class imbalance
  - Feature scaling (Standardization / Z-score) and label encoding as needed
  - Optionally create polynomial interaction features to capture non-linear relationships
- Model training:
  - Multiple algorithms were evaluated (e.g., Logistic Regression, Random Forest, LightGBM)
  - Logistic Regression produced the best/balanced results during development (update reported metrics based on your final experiments)
  - Example reported accuracy: ~99.2% (please replace with your verified final results)

---

## How to Run Locally
1. Clone the repository:
```bash
git clone https://github.com/<YourUsername>/Mental-Health-Diagnostic-System.git
cd Mental-Health-Diagnostic-System
```

2. Create and activate a virtual environment, then install dependencies:
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

3. Ensure the trained model file `mental_health_models.pkl` is available locally (do not push it to GitHub). If you store it remotely (e.g., Google Drive, Hugging Face), update `app.py` to load from that path.

4. Run the application:
```bash
python app.py
```
Open the Gradio URL printed in the console (commonly `http://127.0.0.1:7860`) to access the UI.

---

## Deployment Tips
- For small public demos, consider deploying the UI to Hugging Face Spaces (Gradio) or a simple Cloud VM.
- Do not commit large binary model files to Git. Use external storage (Google Drive, AWS S3, Hugging Face Hub) and load them at runtime.
- If deploying on a shared service, secure any private datasets and credentials via environment variables.

---

## Reproducibility Notes
- Include training scripts and a `train/` folder containing:
  - `train.py` (training pipeline)
  - `config.yaml` (hyperparameters and data paths)
  - `requirements.txt` (matching environment used for training)
- Provide a dataset source link and any instructions to prepare the dataset (e.g., mapping files, label definitions).
- Save model metadata (training date, random seeds, cross-validation folds, final metrics) alongside the model artifacts.

---

## Interpretability & Clinical Traceability
To support explainability in clinical contexts:
- Integrate SHAP or LIME to produce both global and local explanations for model predictions.
- Log feature contributions used for each generated report so clinicians can review why a decision was made.
- Document any ethical considerations and limitations (bias, generalizability, required clinical oversight).

---

## Example: Where to Store Large Models
Recommended approaches:
- Upload heavy model files to a cloud storage (Google Drive, AWS S3) and provide a loader function in `app.py` to download at runtime.
- Use Hugging Face Model Hub or a private artifact repository if you want versioning and access control.

---

## Requirements (example)
Ensure `requirements.txt` contains packages used by the app:
```text
gradio
scikit-learn
pandas
numpy
imbalanced-learn
pillow
shap        # optional
lime        # optional
```
Adjust versions as needed for reproducibility.

---

## Credits & Contact
- Developed by: Md. Nazmus Sakib  
- Model artifact filename referenced in this repo: `mental_health_models.pkl` (keep it local / remote storage — do not commit)

---

## Final Notes
- Before sharing the repository publicly, double-check `.gitignore` to ensure no sensitive data, credentials, or large binaries are committed.
- Update reported metrics (accuracy, precision/recall, confusion matrices) in this README after final evaluation.
- If you want, I can help:
  - Polish `app.py` for secure remote model loading
  - Prepare training scripts and a minimal Dockerfile for reproducible deployment
  - Generate a short CONTRIBUTING.md or CHANGELOG.md

```