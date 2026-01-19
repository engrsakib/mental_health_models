# 🏥 AI-Powered Mental Health Diagnostic System
### Automated Clinical Assessment using Machine Learning (DASS-42)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Library](https://img.shields.io/badge/Library-Gradio%20%7C%20Scikit--learn-orange)
![Status](https://img.shields.io/badge/Status-Prototype%20%2F%20Deployed-green)

---

## Overview
This repository implements an AI-driven diagnostic system that predicts severity levels of Depression, Anxiety, and Stress using responses derived from the DASS-42 instrument. The system runs inference with a pre-trained model and generates a professional clinical-style report (including verified stamp and signature images).

Author: Md. Nazmus Sakib (GitHub: @engrsakib)

---

## Key Features
- Interactive Gradio UI for quick, user-friendly assessments  
- Downloadable verified report (stamp + signature) generated with Pillow (PIL)  
- Class imbalance handling using SMOTETomek during preprocessing  
- Real-time evaluation based on 22 clinical questions derived from DASS-42  
- Optional interpretability support (SHAP / LIME) for global and local explanations

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
└── README.md                   # Project documentation
```

---

## Important: Large Files & Git
Do NOT commit the trained model file (`mental_health_models.pkl`) or the virtual environment directory to the repository. Use the `.gitignore` entries below to prevent accidental commits.

Recommended `.gitignore` entries:
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
- Preprocessing:
  - Clean missing values and standardize inputs
  - Handle class imbalance using SMOTETomek
  - Feature scaling (Standardization / Z-score)
  - Label encoding as required
  - Optionally add polynomial interaction features to capture non-linear relationships
- Model development:
  - Multiple algorithms evaluated (e.g., Logistic Regression, Random Forest, LightGBM)
  - Final inference uses a trained model saved as `mental_health_models.pkl`
  - Record and report final evaluation metrics (accuracy, precision, recall, F1, confusion matrix)

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

3. Ensure the trained model file `mental_health_models.pkl` is available locally or configure `app.py` to load it from external storage (e.g., Google Drive, S3, Hugging Face).

4. Start the application:
```bash
python app.py
```
Open the Gradio URL printed in the console (commonly `http://127.0.0.1:7860`) to access the UI.

---

## Deployment Recommendations
- Host the Gradio app on Hugging Face Spaces or a cloud VM for public demos.  
- Store large binary artifacts in external storage (Google Drive, AWS S3, Hugging Face Hub) and load them at runtime.  
- Use environment variables for any credentials. Avoid committing secrets to the repository.

---

## Reproducibility & Artifacts
To facilitate reproducibility, include the following in the repository or a separate `train/` folder:
- `train.py` — training pipeline script  
- `config.yaml` or `config.json` — hyperparameters and data paths  
- `requirements.txt` — environment specification (precise versions)  
- Model metadata (training date, random seed, CV folds, final metrics)  

---

## Interpretability & Clinical Traceability
- Integrate SHAP or LIME to produce explanation reports for each prediction.  
- Log feature contributions for generated reports so clinicians can review decision rationale.  
- Document ethical considerations, potential biases, and limits of generalizability.

---

## Requirements (example)
Ensure `requirements.txt` includes:
```
gradio
scikit-learn
pandas
numpy
imbalanced-learn
pillow
shap       # optional
lime       # optional
```
Pin versions as needed for reproducibility.

---

## Credits
Developed by Md. Nazmus Sakib — GitHub: @engrsakib

---
```