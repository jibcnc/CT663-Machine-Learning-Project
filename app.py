from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
from pathlib import Path
import logging

BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

MODEL_PATH = BASE_DIR / "sleep_disorder_model.pkl"
COLUMNS_PATH = BASE_DIR / "model_columns.pkl"
LABELS_PATH = BASE_DIR / "label_classes.pkl"
MODEL_INFO_PATH = BASE_DIR / "model_info.pkl"

model = joblib.load(MODEL_PATH)
model_columns = list(joblib.load(COLUMNS_PATH))
label_classes = list(joblib.load(LABELS_PATH))

DEFAULT_MODEL_INFO = {
    "title": "Sleep Disorder Risk Screening",
    "model_name": "Final Trained Sleep Disorder Model",
    "dataset_size": 374,
    "target": "Sleep Disorder · 3 Classes",
    "deploy_mode": "Flask Web Application",
    "best_model": "Loaded from Final Notebook",
    "cv_accuracy": "-",
    "cv_macro_f1": "-",
    "final_accuracy": "-",
    "final_precision": "-",
    "final_recall": "-",
    "final_macro_f1": "-",
}

MODEL_INFO = DEFAULT_MODEL_INFO.copy()
if MODEL_INFO_PATH.exists():
    try:
        exported_info = joblib.load(MODEL_INFO_PATH)
        if isinstance(exported_info, dict):
            MODEL_INFO.update(exported_info)
    except Exception:
        logging.exception("Cannot load model_info.pkl, using default MODEL_INFO")

OCCUPATIONS = [
    {"value": "Accountant", "label": "นักบัญชี (Accountant)"},
    {"value": "Doctor", "label": "แพทย์ (Doctor)"},
    {"value": "Engineer", "label": "วิศวกร (Engineer)"},
    {"value": "Lawyer", "label": "นักกฎหมาย / ทนาย (Lawyer)"},
    {"value": "Manager", "label": "ผู้จัดการ / งานบริหาร (Manager)"},
    {"value": "Nurse", "label": "พยาบาล (Nurse)"},
    {"value": "Sales Representative", "label": "พนักงานขายภาคสนาม (Sales Representative)"},
    {"value": "Salesperson", "label": "พนักงานขาย (Salesperson)"},
    {"value": "Scientist", "label": "นักวิทยาศาสตร์ (Scientist)"},
    {"value": "Software Engineer", "label": "วิศวกรซอฟต์แวร์ (Software Engineer)"},
    {"value": "Teacher", "label": "ครู / อาจารย์ (Teacher)"},
]

OCCUPATION_THAI = {
    "Accountant": "นักบัญชี",
    "Doctor": "แพทย์",
    "Engineer": "วิศวกร",
    "Lawyer": "นักกฎหมาย / ทนาย",
    "Manager": "ผู้จัดการ / งานบริหาร",
    "Nurse": "พยาบาล",
    "Sales Representative": "พนักงานขายภาคสนาม",
    "Salesperson": "พนักงานขาย",
    "Scientist": "นักวิทยาศาสตร์",
    "Software Engineer": "วิศวกรซอฟต์แวร์",
    "Teacher": "ครู / อาจารย์",
}

OCCUPATION_GROUPS = [
    ("Accountant", "งานสำนักงาน / เอกสาร / การเงิน"),
    ("Doctor", "งานสายการแพทย์ / สาธารณสุข"),
    ("Nurse", "งานดูแลผู้ป่วย / งานที่เคลื่อนไหวสูง"),
    ("Engineer", "งานด้านเทคนิค / วิศวกรรม"),
    ("Software Engineer", "งาน IT / ซอฟต์แวร์ / นั่งหน้าคอม"),
    ("Scientist", "งานวิจัย / ห้องแล็บ"),
    ("Manager", "งานบริหาร / หัวหน้างาน"),
    ("Lawyer", "งานกฎหมาย / ที่ปรึกษา"),
    ("Teacher", "งานสอน / การศึกษา"),
    ("Salesperson", "งานขาย / บริการลูกค้า"),
    ("Sales Representative", "งานขายภาคสนาม / เดินทางบ่อย"),
]

BMI_OPTIONS = ["Normal", "Overweight", "Obese"]

PREDICTION_INFO = {
    "None": {
        "title": "ผลประเมินเบื้องต้น: ไม่พบความเสี่ยงเด่นชัด",
        "short_label": "ปกติ (None)",
        "description": "จากข้อมูลที่กรอก ระบบยังไม่พบรูปแบบที่เด่นชัดของภาวะผิดปกติการนอนในระดับคัดกรองเบื้องต้น",
        "clinical_note": "รักษาพฤติกรรมการนอน การออกกำลังกาย และการจัดการความเครียดอย่างสม่ำเสมอ",
        "risk_level": "ต่ำ",
        "status": "None",
    },
    "Insomnia": {
        "title": "ผลประเมินเบื้องต้น: มีแนวโน้มภาวะนอนไม่หลับ",
        "short_label": "Insomnia",
        "description": "ระบบพบรูปแบบข้อมูลที่สอดคล้องกับภาวะนอนไม่หลับ เช่น ชั่วโมงการนอนน้อย คุณภาพการนอนต่ำ หรือระดับความเครียดสูง",
        "clinical_note": "ควรปรับพฤติกรรมก่อนนอนและติดตามอาการ หากเป็นต่อเนื่องควรปรึกษาผู้เชี่ยวชาญ",
        "risk_level": "ปานกลาง",
        "status": "Insomnia",
    },
    "Sleep Apnea": {
        "title": "ผลประเมินเบื้องต้น: มีแนวโน้มภาวะหยุดหายใจขณะหลับ",
        "short_label": "Sleep Apnea",
        "description": "ระบบพบรูปแบบข้อมูลที่สอดคล้องกับความเสี่ยง Sleep Apnea เช่น BMI สูง ความดันโลหิตสูง หรืออัตราการเต้นหัวใจขณะพักสูง",
        "clinical_note": "หากมีอาการกรนดัง ง่วงกลางวันมาก หรือสะดุ้งตื่นกลางคืน ควรปรึกษาแพทย์เพื่อประเมินเพิ่มเติม",
        "risk_level": "สูง",
        "status": "Sleep Apnea",
    },
}

BAR_ORDER = ["None", "Insomnia", "Sleep Apnea"]


def normalize_bmi_category(value: str) -> str:
    value = (value or "").strip()
    if value == "Normal Weight":
        return "Normal"
    return value


def valid_occupation_values():
    return [item["value"] for item in OCCUPATIONS]


def resolve_occupation(data: dict) -> str:
    valid_occupations = valid_occupation_values()
    occupation = data.get("occupation", "").strip()

    if occupation == "__other__":
        mapped = data.get("occ_other", "").strip()
        return mapped if mapped in valid_occupations else "Accountant"

    return occupation if occupation in valid_occupations else "Accountant"


def preprocess_input(data: dict) -> pd.DataFrame:
    resolved_occupation = resolve_occupation(data)
    bmi_category = normalize_bmi_category(data["bmi_category"])

    df_input = pd.DataFrame([{
        "Gender": 1 if data["gender"] == "Male" else 0,
        "Age": int(data["age"]),
        "Sleep Duration": float(data["sleep_duration"]),
        "Quality of Sleep": int(data["quality_of_sleep"]),
        "Physical Activity Level": int(data["physical_activity"]),
        "Stress Level": int(data["stress_level"]),
        "BP_Systolic": int(data["bp_systolic"]),
        "BP_Diastolic": int(data["bp_diastolic"]),
        "Heart Rate": int(data["heart_rate"]),
        "Daily Steps": int(data["daily_steps"]),
        "Occupation": resolved_occupation,
        "BMI Category": bmi_category,
    }])

    df_input = pd.get_dummies(
        df_input,
        columns=["Occupation", "BMI Category"],
        drop_first=True
    )

    for col in model_columns:
        if col not in df_input.columns:
            df_input[col] = 0

    df_input = df_input[model_columns]
    return df_input


def build_recommendations(data: dict, prediction: str):
    recs = {"priority": [], "monitor": [], "medical": []}

    sleep_duration = float(data["sleep_duration"])
    quality = int(data["quality_of_sleep"])
    stress = int(data["stress_level"])
    activity = int(data["physical_activity"])
    steps = int(data["daily_steps"])
    heart_rate = int(data["heart_rate"])
    bp_sys = int(data["bp_systolic"])
    bp_dia = int(data["bp_diastolic"])
    bmi = normalize_bmi_category(data["bmi_category"])

    if sleep_duration < 7:
        recs["priority"].append("เพิ่มเวลานอนให้เข้าใกล้ 7–9 ชั่วโมงต่อคืน และพยายามนอนให้เป็นเวลา")
    if quality <= 5:
        recs["priority"].append("ลดแสงหน้าจอ คาเฟอีน และกิจกรรมกระตุ้นก่อนนอน เพื่อช่วยเพิ่มคุณภาพการนอน")
    if stress >= 7:
        recs["priority"].append("ระดับความเครียดค่อนข้างสูง ควรเพิ่มกิจกรรมผ่อนคลายก่อนนอน เช่น หายใจลึกหรือทำสมาธิสั้น ๆ")

    if activity < 30:
        recs["monitor"].append("กิจกรรมทางกายค่อนข้างน้อย ลองเพิ่มการเดินหรือออกกำลังกายเบา ๆ อย่างสม่ำเสมอ")
    if steps < 5000:
        recs["monitor"].append("จำนวนก้าวต่อวันค่อนข้างต่ำ ควรเพิ่มการเคลื่อนไหวระหว่างวัน")
    if heart_rate >= 85:
        recs["monitor"].append("อัตราการเต้นหัวใจขณะพักค่อนข้างสูง ควรติดตามร่วมกับความเครียดและการพักผ่อน")
    if bp_sys >= 130 or bp_dia >= 85:
        recs["monitor"].append("ความดันโลหิตค่อนข้างสูง ควรใส่ใจอาหาร การพักผ่อน และการออกกำลังกาย")
    if bmi in {"Overweight", "Obese"}:
        recs["monitor"].append("BMI อยู่ในกลุ่มเสี่ยง การควบคุมน้ำหนักอาจช่วยลดความเสี่ยงด้านการนอน")

    if prediction == "Sleep Apnea":
        recs["medical"].append("หากมีอาการกรนดัง ง่วงกลางวันมาก หรือสะดุ้งตื่นกลางคืน ควรตรวจเพิ่มเติมกับแพทย์")
    elif prediction == "Insomnia":
        recs["medical"].append("หากมีอาการนอนไม่หลับต่อเนื่องหลายสัปดาห์ ควรปรึกษาแพทย์หรือผู้เชี่ยวชาญด้านการนอน")
    elif not any(recs.values()):
        recs["priority"].append("ภาพรวมอยู่ในเกณฑ์ดี ควรรักษาพฤติกรรมการนอนและสุขภาพแบบนี้ต่อไป")

    return recs


@app.route("/")
def home():
    return render_template("home.html", model_info=MODEL_INFO)


@app.route("/assessment")
def assessment():
    return render_template(
        "assessment.html",
        model_info=MODEL_INFO,
        occupations=OCCUPATIONS,
        occupation_groups=OCCUPATION_GROUPS,
        bmi_options=BMI_OPTIONS,
    )


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html", model_info=MODEL_INFO)


@app.route("/about")
def about():
    return render_template("about.html", model_info=MODEL_INFO)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": True, "model_name": MODEL_INFO.get("model_name")})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        required_fields = [
            "gender", "age", "occupation",
            "sleep_duration", "quality_of_sleep",
            "physical_activity", "stress_level",
            "daily_steps", "heart_rate",
            "bmi_category", "bp_systolic", "bp_diastolic"
        ]

        missing = [f for f in required_fields if f not in data or str(data[f]).strip() == ""]
        if missing:
            return jsonify({"error": f"กรุณากรอกข้อมูลให้ครบ: {', '.join(missing)}"}), 400

        age = int(data["age"])
        sleep_duration = float(data["sleep_duration"])
        quality = int(data["quality_of_sleep"])
        activity = int(data["physical_activity"])
        stress = int(data["stress_level"])
        steps = int(data["daily_steps"])
        heart_rate = int(data["heart_rate"])
        bp_sys = int(data["bp_systolic"])
        bp_dia = int(data["bp_diastolic"])

        if not (1 <= age <= 120):
            return jsonify({"error": "อายุต้องอยู่ระหว่าง 1 ถึง 120 ปี"}), 400
        if not (0 <= sleep_duration <= 24):
            return jsonify({"error": "ชั่วโมงการนอนต้องอยู่ระหว่าง 0 ถึง 24"}), 400
        if not (1 <= quality <= 10):
            return jsonify({"error": "คุณภาพการนอนต้องอยู่ระหว่าง 1 ถึง 10"}), 400
        if not (0 <= activity <= 500):
            return jsonify({"error": "กิจกรรมทางกายต้องอยู่ระหว่าง 0 ถึง 500"}), 400
        if not (1 <= stress <= 10):
            return jsonify({"error": "ระดับความเครียดต้องอยู่ระหว่าง 1 ถึง 10"}), 400
        if not (0 <= steps <= 100000):
            return jsonify({"error": "จำนวนก้าวต่อวันไม่ถูกต้อง"}), 400
        if not (20 <= heart_rate <= 250):
            return jsonify({"error": "อัตราการเต้นหัวใจไม่ถูกต้อง"}), 400
        if not (50 <= bp_sys <= 300 and 20 <= bp_dia <= 200):
            return jsonify({"error": "ค่าความดันโลหิตไม่ถูกต้อง"}), 400
        if bp_sys <= bp_dia:
            return jsonify({"error": "ความดันตัวบนต้องมากกว่าความดันตัวล่าง"}), 400

        input_df = preprocess_input(data)

        pred = model.predict(input_df)[0]
        try:
            prediction = label_classes[int(pred)]
        except (ValueError, TypeError, IndexError):
            prediction = str(pred)

        probabilities = {}
        confidence = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_df)[0]
            probabilities = {
                label_classes[i]: round(float(probs[i]) * 100, 2)
                for i in range(len(label_classes))
            }
            confidence = round(max(probabilities.values()), 2)

        resolved_occ = resolve_occupation(data)
        info = PREDICTION_INFO.get(prediction, {
            "title": prediction,
            "short_label": prediction,
            "description": "",
            "clinical_note": "",
            "risk_level": "-",
            "status": prediction,
        })

        return jsonify({
            "prediction": prediction,
            "title": info["title"],
            "short_label": info["short_label"],
            "description": info["description"],
            "clinical_note": info["clinical_note"],
            "risk_level": info["risk_level"],
            "status": info["status"],
            "confidence": confidence,
            "probabilities": probabilities,
            "recommendations": build_recommendations(data, prediction),
            "resolved_occupation": resolved_occ,
            "resolved_occupation_label": f'{OCCUPATION_THAI.get(resolved_occ, resolved_occ)} ({resolved_occ})',
            "display_order": BAR_ORDER,
        })

    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "เกิดข้อผิดพลาดระหว่างประมวลผลการทำนาย"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
