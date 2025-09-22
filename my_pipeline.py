


import os, re, pickle, pdfplumber
from PIL import Image
import pytesseract
import pandas as pd

# ========== 1. Load Models (if available) ==========
def load_report_type_model(path="models/report_type_model.pkl"):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)   # (vectorizer, model)
    return None

def load_risk_model(path="models/risk_model.pkl"):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)   # risk model
    return None

# ========== 2. Extract Text ==========
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def extract_text_from_image(img_path):
    img = Image.open(img_path)
    return pytesseract.image_to_string(img)

def extract_text(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(path)
    else:
        return extract_text_from_image(path)

# ========== 3. Parse Tests from Text ==========
def parse_tests(text):
    pattern = r"([A-Za-z ()]+)\s*[:\-]?\s*([\d.]+)\s*([A-Za-z/%µIUml]+)?"
    matches = re.findall(pattern, text)
    tests = {}
    for test, value, unit in matches:
        tests[test.strip()] = (float(value), unit.strip() if unit else "")
    return tests

# ========== 4. Report Type Classification ==========
def predict_report_type(text, model_tuple=None):
    if model_tuple:   # ML model
        vectorizer, clf = model_tuple
        X = vectorizer.transform([text])
        return clf.predict(X)[0]
    # fallback → keyword-based
    t = text.lower()
    if any(k in t for k in ["bilirubin", "alt", "ast", "albumin"]): return "LFT"
    if any(k in t for k in ["creatinine", "urea", "sodium", "potassium"]): return "RFT"
    if any(k in t for k in ["tsh", "t3", "t4"]): return "Thyroid"
    return "Unknown"

# ========== 5. Risk Rules ==========
def check_risk(test, value):
    test = test.lower()
    if "alt" in test and value > 40: return "Risk"
    if "ast" in test and value > 40: return "Risk"
    if "bilirubin" in test and value > 1.2: return "Risk"
    if "creatinine" in test and value > 1.5: return "Risk"
    if "urea" in test and value > 40: return "Risk"
    if "tsh" in test and (value < 0.5 or value > 5): return "Risk"
    return "Normal"

def tamil_suggestion(diagnosis):
    return ("உங்கள் பரிசோதனை மதிப்புகள் சாதாரணமாக உள்ளன. கவலைப்பட தேவையில்லை."
            if diagnosis == "Normal"
            else "உங்கள் பரிசோதனை மதிப்புகள் ஆபத்தான நிலையில் உள்ளன. உடனடியாக மருத்துவரை அணுகவும்.")

# ========== 6. Unified Pipeline ==========
def analyze_report(path):
    # Load models if available
    report_type_model = load_report_type_model()
    risk_model = load_risk_model()

    # Step 1: Extract text
    raw_text = extract_text(path)

    # Step 2: Predict Report Type
    report_type = predict_report_type(raw_text, report_type_model)

    # Step 3: Extract Tests
    tests = parse_tests(raw_text)

    # Step 4: Predict Risk (here using rules for simplicity)
    results = []
    overall_status = "Normal"
    risk_count = 0
    for test, (value, unit) in tests.items():
        diag = check_risk(test, value)
        if diag == "Risk": risk_count += 1
        results.append({
            "Test": test,
            "Value": value,
            "Unit": unit,
            "Diagnosis": diag,
            "Tamil Suggestion": tamil_suggestion(diag)
        })

    if risk_count > 0: overall_status = "Risk"

    # Step 5: Output structured dictionary
    return {
        "Report_Type": report_type,
        "Raw_Text": raw_text[:200] + "...",  # preview
        "Tests": results,
        "Overall_Diagnosis": overall_status,
        "Tamil_Suggestion": tamil_suggestion(overall_status)
    }

# ========== Example Usage ==========
if __name__ == "__main__":
    pdf_path = r"C:\Users\Ashok AK\Project\fresh\day 1\Sample Data\LFT_report_p.pdf"
    result = analyze_report(pdf_path)
    import json
    print(json.dumps(result, indent=2, ensure_ascii=False))

