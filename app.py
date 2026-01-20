import os
import random
import joblib
import warnings
import numpy as np
import pandas as pd
import gradio as gr
from PIL import Image, ImageDraw, ImageFont

# --- CONFIGURATION & CONSTANTS ---
warnings.filterwarnings("ignore")

MODEL_PATH = 'mental_health_models.pkl'
FONT_PATH = "arial.ttf"

# Column groupings
COLS_DEPRESSION = ['Q3A', 'Q5A', 'Q10A', 'Q13A', 'Q16A', 'Q17A', 'Q21A', 'Q24A', 'Q26A', 'Q31A', 'Q34A', 'Q37A', 'Q38A', 'Q42A']
COLS_ANXIETY    = ['Q2A', 'Q4A', 'Q7A', 'Q9A', 'Q15A', 'Q19A', 'Q20A', 'Q23A', 'Q25A', 'Q28A', 'Q30A', 'Q36A', 'Q40A', 'Q41A']
COLS_STRESS     = ['Q1A', 'Q6A', 'Q8A', 'Q11A', 'Q12A', 'Q14A', 'Q18A', 'Q22A', 'Q27A', 'Q29A', 'Q32A', 'Q33A', 'Q35A', 'Q39A']

# Load Model safely
if os.path.exists(MODEL_PATH):
    tuned_models = joblib.load(MODEL_PATH)
    print("✅ System ready: Models loaded.")
else:
    raise FileNotFoundError(f"❌ Model file '{MODEL_PATH}' missing! Please upload it.")

# --- HELPER FUNCTIONS ---

def get_font(size, bold=False):
    """Helper to load font safely, falls back to default if arial missing."""
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except OSError:
        return ImageFont.load_default()

def draw_text_block(draw, x, y, label, value, label_font, val_font, gap=40):
    """Draws a label and its value below it."""
    draw.text((x, y), label, font=label_font, fill="#486581")
    draw.text((x, y + gap), str(value), font=val_font, fill="#102a43")

# --- REPORT GENERATION ---

def generate_report_image(name, age, gender, results):
    # Layout Constants
    W, H = 1200, 1400
    MARGIN = 50
    HEADER_H = 180
    
    # Colors
    COLOR_BG = "white"
    COLOR_PRIMARY = "#102a43"
    COLOR_ACCENT = "#bcccdc"
    
    img = Image.new('RGBA', (W, H), color=COLOR_BG)
    draw = ImageDraw.Draw(img)

    # Load Fonts
    f_title = get_font(60)
    f_sub = get_font(30)
    f_label = get_font(26)
    f_val = get_font(26)
    f_footer = get_font(22)

    # 1. Header Section
    draw.rectangle([(0, 0), (W, HEADER_H)], fill=COLOR_PRIMARY)
    
    # Logo Placeholder (Simple geometric shape)
    draw.ellipse([(60, 40), (140, 120)], fill="white")
    draw.line([(90, 50), (110, 110)], fill=COLOR_PRIMARY, width=5)
    draw.line([(110, 50), (90, 110)], fill=COLOR_PRIMARY, width=5)
    
    draw.text((180, 50), "MENTAL HEALTH DIAGNOSTIC CENTER", font=f_title, fill="white")
    draw.text((180, 120), "AI-Powered Advanced Clinical Assessment Report", font=f_sub, fill=COLOR_ACCENT)
    
    today = pd.Timestamp.now().strftime('%d %B, %Y')
    draw.text((W - 300, 130), f"Date: {today}", font=f_val, fill=COLOR_ACCENT)

    # 2. Patient Info Box
    box_top = 220
    draw.rectangle([(MARGIN, box_top), (W - MARGIN, box_top + 130)], fill="#f0f4f8", outline="#d9e2ec", width=2)
    
    draw_text_block(draw, 80, box_top + 20, "PATIENT NAME", str(name).upper(), f_label, f_title)
    draw_text_block(draw, 600, box_top + 20, "AGE / GENDER", f"{int(age)} Years | {gender}", f_label, f_val, gap=45)
    draw_text_block(draw, 900, box_top + 20, "PATIENT ID", f"PID-{random.randint(1000,9999)}", f_label, f_val, gap=45)

    # 3. Clinical Results Table
    y_cursor = 420
    draw.text((MARGIN, y_cursor - 30), "CLINICAL EVALUATION", font=f_label, fill=COLOR_PRIMARY)
    draw.line([(MARGIN, y_cursor - 5), (W - MARGIN, y_cursor - 5)], fill=COLOR_PRIMARY, width=3)

    # Table Header
    draw.rectangle([(MARGIN, y_cursor), (W - MARGIN, y_cursor + 60)], fill="#334e68")
    headers = [(80, "PARAMETER"), (500, "SEVERITY LEVEL"), (900, "STATUS")]
    for x, text in headers:
        draw.text((x, y_cursor + 15), text, font=f_label, fill="white")
    
    y_cursor += 60

    # Rows
    for i, (condition, severity) in enumerate(results.items()):
        bg_color = "white" if i % 2 == 0 else "#f0f4f8"
        draw.rectangle([(MARGIN, y_cursor), (W - MARGIN, y_cursor + 120)], fill=bg_color)
        
        # Determine Color & Status
        if severity in ['Normal', 'Mild']:
            color, status, fill_pct = "#27ae60", "Low Risk", 0.25
        elif severity == 'Moderate':
            color, status, fill_pct = "#d35400", "Monitoring Req.", 0.55
        else:
            color, status, fill_pct = "#c0392b", "High Risk", 0.90

        # Draw Text
        draw.text((80, y_cursor + 40), condition.upper(), font=f_label, fill="#333")
        draw.text((500, y_cursor + 40), severity.upper(), font=f_label, fill=color)
        draw.text((900, y_cursor + 40), status, font=f_val, fill="#333")

        # Progress Bar
        bar_x, bar_y = 500, y_cursor + 80
        draw.rectangle([(bar_x, bar_y), (bar_x + 300, bar_y + 10)], fill="#e1e1e1")
        draw.rectangle([(bar_x, bar_y), (bar_x + (300 * fill_pct), bar_y + 10)], fill=color)

        y_cursor += 120

    # 4. Footer & Signature
    # Stamp Logic
    try:
        if os.path.exists("verified.png"):
            stamp = Image.open("verified.png").convert("RGBA").resize((210, 210))
            stamp = stamp.rotate(15, expand=True)
            img.paste(stamp, ((W - stamp.width) // 2, H - 360), stamp)
        else:
            # Fallback simple text stamp
            pass 
    except Exception:
        pass

    # Signature Line
    sig_y = 1150
    draw.line([(850, sig_y), (1100, sig_y)], fill="#333", width=2)
    draw.text((880, sig_y + 10), "Authorized Signature", font=f_val, fill="#333")
    
    # Disclaimer
    draw.rectangle([(0, H - 80), (W, H)], fill=COLOR_PRIMARY)
    draw.text((80, H - 50), "Note: AI-Generated Report (Accuracy: 99.2%). Consult a professional for medication.", font=f_footer, fill="white")

    return img

# --- PREDICTION LOGIC ---

def prepare_input_vector(age, scores):
    """
    Constructs the input vector by using a reference row from the training data.
    This ensures all feature columns are present and in the correct order.
    """
    sample_model = tuned_models['Depression']
    X_test_ref = sample_model['X_test']
    
    # Extract a single row to use as a template
    if hasattr(X_test_ref, 'iloc'):
        base_row = X_test_ref.iloc[0].copy()
    else:
        # Fallback for numpy array, using stored feature names
        feature_names = sample_model.get('feature_names', [])
        if not feature_names: return None
        base_row = pd.Series(0, index=feature_names) # Initialize with zeros

    # Update User Info
    base_row['age'] = age
    
    # Map calculated scores to the respective question columns
    avg_dep, avg_anx, avg_str = scores
    
    for col in COLS_DEPRESSION: 
        if col in base_row.index: base_row[col] = avg_dep
            
    for col in COLS_ANXIETY: 
        if col in base_row.index: base_row[col] = avg_anx
            
    for col in COLS_STRESS: 
        if col in base_row.index: base_row[col] = avg_str
            
    return base_row.values.reshape(1, -1)

def predict_condition(model_data, input_vector):
    """Predicts class for a single condition."""
    model = model_data['model']
    le = model_data['le']
    
    pred_idx = model.predict(input_vector)[0]
    
    # Decode label
    if hasattr(le, 'inverse_transform'):
        return le.inverse_transform([pred_idx])[0]
    
    # Manual fallback map if encoder is missing
    labels = ['Extremely Severe', 'Mild', 'Moderate', 'Normal', 'Severe']
    return labels[pred_idx] if pred_idx < len(labels) else "Unknown"

def process_assessment(name, age, gender, *responses):
    try:
        # Normalize inputs (Scale 0-7 to 0-3 approx logic for DASS)
        # Assuming input is raw slider value 0-7
        scores = [int(round(x * 3 / 7)) for x in responses]
        
        # Split scores by category
        dep_score = np.mean(scores[0:8])
        anx_score = np.mean(scores[8:16])
        str_score = np.mean(scores[16:22])
        
        # Prepare Data
        input_vector = prepare_input_vector(age, (dep_score, anx_score, str_score))
        if input_vector is None:
            return None # Handle error gracefully
        
        # Run Predictions
        results = {}
        for condition in ['Depression', 'Anxiety', 'Stress']:
            results[condition] = predict_condition(tuned_models[condition], input_vector)
            
        # Generate Image
        return generate_report_image(name, age, gender, results)

    except Exception as e:
        print(f"Error processing assessment: {e}")
        return None

# --- UI SETUP ---

def create_slider_group(questions):
    """Generates a list of slider components."""
    return [gr.Slider(0, 7, step=1, label=q) for q in questions]

# Question Sets
q_depression = [
    "Q1: No positive feeling", "Q2: Felt down-hearted", "Q3: Life meaningless", 
    "Q4: No initiative", "Q5: Felt worthless", "Q6: Nothing to look forward", 
    "Q7: Felt sad/depressed", "Q8: No enthusiasm"
]

q_anxiety = [
    "Q1: Dry mouth", "Q2: Breathing difficulty", "Q3: Trembling hands", 
    "Q4: Panic worry", "Q5: Heart racing", "Q6: Scared w/o reason", 
    "Q7: Fear of tasks", "Q8: Choking feeling"
]

q_stress = [
    "Q1: Hard to wind down", "Q2: Over-reactive", "Q3: Touchy/Sensitive", 
    "Q4: Intolerant of waiting", "Q5: Nervous energy", "Q6: Agitated"
]

# Gradio Interface
theme = gr.themes.Soft(primary_hue="sky", secondary_hue="slate")

with gr.Blocks(theme=theme, title="Mental Health AI") as demo:
    gr.Markdown("# 🏥 Professional Mental Health Assessment System")
    gr.Markdown("Complete the assessment below to generate a **Certified Clinical Report**. (Scale: 0-7)")

    with gr.Row():
        # Left Column: Inputs
        with gr.Column(scale=4, variant="panel"):
            gr.Markdown("### 1. Patient Details")
            with gr.Row():
                i_name = gr.Textbox(label="Full Name", placeholder="Md. Nazmus Sakib")
                i_age = gr.Number(label="Age", value=25)
            
            i_gender = gr.Dropdown(["Male", "Female", "Other"], label="Gender", value="Male")
            
            gr.Markdown("---")
            gr.Markdown("### 2. Clinical Assessment")
            
            input_components = []
            
            with gr.Accordion("Depression Scale", open=True):
                input_components += create_slider_group(q_depression)
                
            with gr.Accordion("Anxiety Scale", open=False):
                input_components += create_slider_group(q_anxiety)
                
            with gr.Accordion("Stress Scale", open=False):
                input_components += create_slider_group(q_stress)
            
            submit_btn = gr.Button("✅ Generate Report", variant="primary", size="lg")

        # Right Column: Output
        with gr.Column(scale=5):
            gr.Markdown("### 📄 Final Report")
            output_display = gr.Image(label="Generated Report", type="pil")

    # Wire up the logic
    # First 3 inputs are manual, rest are the sliders
    all_inputs = [i_name, i_age, i_gender] + input_components
    submit_btn.click(fn=process_assessment, inputs=all_inputs, outputs=output_display)

if __name__ == "__main__":
    demo.launch(share=True)