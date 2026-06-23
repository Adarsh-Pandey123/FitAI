import streamlit as st
import pickle
import numpy as np
import json
import os
from datetime import datetime
from groq import Groq

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="FitAI", page_icon="⚡", layout="wide")

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; background: #080c14; }

/* ── HEADER ── */
.top-bar {
    display: flex; justify-content: space-between; align-items: center;
    padding: 1rem 0 1.5rem; border-bottom: 1px solid #1a2236; margin-bottom: 1.8rem;
}
.brand { font-family: 'Syne', sans-serif; font-size: 2.8rem; font-weight: 800; color: #fff; letter-spacing: -0.02em; }
.brand span { color: #f97316; }
.brand-sub { color: #475569; font-size: 0.72rem; font-family: 'Inter', sans-serif; font-weight: 500; letter-spacing: 0.08em; text-transform: uppercase; margin-top: 2px; }
.by-tag { color: #334155; font-size: 0.75rem; }
.by-tag b { color: #f97316; }

/* ── TABS ── */
.tab-row { display: flex; gap: 0.5rem; margin-bottom: 2rem; }
.tab-btn {
    padding: 0.5rem 1.2rem; border-radius: 8px; font-size: 0.82rem; font-weight: 600;
    cursor: pointer; border: 1px solid #1e293b; background: #0d1422; color: #64748b;
    transition: all 0.15s;
}
.tab-btn.active { background: #f97316; color: #fff; border-color: #f97316; }

/* ── CARDS ── */
.card {
    background: #0d1422; border: 1px solid #1a2236; border-radius: 14px;
    padding: 1.4rem; margin-bottom: 1rem;
}
.card-title { color: #94a3b8; font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.3rem; }

/* ── METRIC CARDS ── */
.metric-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.8rem; margin: 1rem 0; }
.m-card {
    background: #0d1422; border: 1px solid #1a2236; border-radius: 12px;
    padding: 1.1rem; text-align: center; position: relative; overflow: hidden;
}
.m-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: var(--accent, #f97316);
}
.m-val { font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 800; color: #f1f5f9; line-height: 1; }
.m-unit { color: #475569; font-size: 0.72rem; margin-top: 0.2rem; }
.m-label { color: #64748b; font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; margin-top: 0.4rem; }

/* ── BMI ── */
.bmi-bar-wrap { background: #1a2236; border-radius: 99px; height: 10px; margin: 0.6rem 0; position: relative; overflow: hidden; }
.bmi-bar-fill { height: 100%; border-radius: 99px; transition: width 0.6s ease; }
.bmi-labels { display: flex; justify-content: space-between; color: #334155; font-size: 0.65rem; margin-top: 0.3rem; }

/* ── HEALTH SCORE ── */
.score-ring {
    width: 110px; height: 110px; border-radius: 50%; margin: 0 auto 0.8rem;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    border: 6px solid;
    font-family: 'Syne', sans-serif;
}
.score-num { font-size: 1.8rem; font-weight: 800; line-height: 1; }
.score-label { font-size: 0.6rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.08em; }

/* ── FOOD LOOKUP ── */
.food-card {
    background: #0d1422; border: 1px solid #1a2236; border-radius: 10px;
    padding: 0.9rem 1.1rem; margin-bottom: 0.5rem;
    display: flex; justify-content: space-between; align-items: center;
}
.food-name { color: #e2e8f0; font-weight: 600; font-size: 0.9rem; }
.food-meta { color: #475569; font-size: 0.72rem; margin-top: 0.15rem; }
.food-cal { font-family: 'Syne', sans-serif; font-size: 1.3rem; font-weight: 800; color: #f97316; }
.food-cal-unit { color: #475569; font-size: 0.65rem; }

/* ── PROGRESS ── */
.history-row {
    background: #0d1422; border: 1px solid #1a2236; border-radius: 10px;
    padding: 1rem 1.2rem; margin-bottom: 0.5rem;
    display: flex; justify-content: space-between; align-items: center;
}
.h-date { color: #475569; font-size: 0.72rem; }
.h-goal { color: #f97316; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; }
.h-cal { font-family: 'Syne', sans-serif; font-size: 2.8rem; font-weight: 800; color: #f1f5f9; }

/* ── PLAN BOX ── */
.plan-box {
    background: #0a0f1a; border: 1px solid #1a2236; border-left: 3px solid #f97316;
    border-radius: 12px; padding: 1.5rem; color: #cbd5e1; line-height: 1.9;
    white-space: pre-wrap; font-size: 0.88rem; margin-top: 1rem;
}

/* ── LEVEL BADGE ── */
.lvl { display: inline-block; padding: 0.3rem 1rem; border-radius: 20px; font-weight: 700; font-size: 0.8rem; }
.Beginner     { background: #0f2a1a; color: #4ade80; border: 1px solid #166534; }
.Intermediate { background: #0f1e3a; color: #60a5fa; border: 1px solid #1e40af; }
.Advanced     { background: #2a0f0f; color: #f87171; border: 1px solid #991b1b; }

/* ── BUTTON ── */
.stButton > button {
    background: linear-gradient(135deg, #f97316, #ea580c) !important;
    color: #fff !important; border: none !important; border-radius: 10px !important;
    padding: 0.7rem 2rem !important; font-weight: 700 !important;
    font-size: 0.95rem !important; width: 100% !important; letter-spacing: 0.02em !important;
}
.stButton > button:hover { opacity: 0.9 !important; }

/* ── INPUTS ── */
div[data-testid="stSelectbox"] label,
div[data-testid="stNumberInput"] label,
div[data-testid="stSlider"] label,
div[data-testid="stTextInput"] label {
    color: #64748b !important; font-size: 0.78rem !important;
    font-weight: 600 !important; text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}
.stApp { background-color: #080c14 !important; }
.block-container { padding-top: 1.2rem !important; max-width: 1250px !important; }
hr { border-color: #1a2236 !important; }
</style>
""", unsafe_allow_html=True)

# ── INDIAN FOOD DATABASE ───────────────────────────────────────────────────────
INDIAN_FOODS = {
    "Dal Tadka (1 bowl)":         {"cal": 180, "protein": 10, "carbs": 28, "fat": 4,  "serving": "200ml"},
    "Paneer Bhurji (100g)":       {"cal": 265, "protein": 18, "carbs": 6,  "fat": 19, "serving": "100g"},
    "Roti (1 piece)":             {"cal": 80,  "protein": 3,  "carbs": 15, "fat": 1,  "serving": "30g"},
    "Boiled Rice (1 cup)":        {"cal": 206, "protein": 4,  "carbs": 45, "fat": 0,  "serving": "186g"},
    "Egg (boiled, 1 whole)":      {"cal": 78,  "protein": 6,  "carbs": 1,  "fat": 5,  "serving": "50g"},
    "Soya Chunks (100g dry)":     {"cal": 345, "protein": 52, "carbs": 33, "fat": 0,  "serving": "100g"},
    "Milk (full fat, 250ml)":     {"cal": 150, "protein": 8,  "carbs": 12, "fat": 8,  "serving": "250ml"},
    "Banana (1 medium)":          {"cal": 89,  "protein": 1,  "carbs": 23, "fat": 0,  "serving": "120g"},
    "Peanut Butter (1 tbsp)":     {"cal": 94,  "protein": 4,  "carbs": 3,  "fat": 8,  "serving": "16g"},
    "Chole (1 bowl)":             {"cal": 210, "protein": 11, "carbs": 35, "fat": 4,  "serving": "200g"},
    "Aloo Paratha (1 piece)":     {"cal": 210, "protein": 4,  "carbs": 32, "fat": 8,  "serving": "100g"},
    "Dahi / Curd (100g)":         {"cal": 61,  "protein": 3,  "carbs": 5,  "fat": 3,  "serving": "100g"},
    "Whey Protein (1 scoop)":     {"cal": 120, "protein": 24, "carbs": 3,  "fat": 2,  "serving": "30g"},
    "Sprouts (100g)":             {"cal": 86,  "protein": 9,  "carbs": 14, "fat": 1,  "serving": "100g"},
    "Poha (1 plate)":             {"cal": 250, "protein": 4,  "carbs": 47, "fat": 6,  "serving": "150g"},
    "Rajma (1 bowl)":             {"cal": 230, "protein": 13, "carbs": 40, "fat": 2,  "serving": "200g"},
    "Oats (50g dry)":             {"cal": 190, "protein": 6,  "carbs": 34, "fat": 3,  "serving": "50g"},
    "Chicken Breast (100g)":      {"cal": 165, "protein": 31, "carbs": 0,  "fat": 4,  "serving": "100g"},
    "Samosa (1 piece)":           {"cal": 262, "protein": 4,  "carbs": 32, "fat": 13, "serving": "100g"},
    "Idli (2 pieces)":            {"cal": 140, "protein": 4,  "carbs": 30, "fat": 1,  "serving": "120g"},
    "Upma (1 plate)":             {"cal": 220, "protein": 4,  "carbs": 38, "fat": 5,  "serving": "150g"},
    "Peanuts (handful, 30g)":     {"cal": 170, "protein": 8,  "carbs": 5,  "fat": 14, "serving": "30g"},
    "Masala Omelette (2 eggs)":   {"cal": 200, "protein": 14, "carbs": 4,  "fat": 14, "serving": "120g"},
    "Lassi (sweet, 250ml)":       {"cal": 180, "protein": 6,  "carbs": 30, "fat": 4,  "serving": "250ml"},
}

# ── PROGRESS FILE ──────────────────────────────────────────────────────────────
HISTORY_FILE = "progress_history.json"

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE) as f:
            return json.load(f)
    return []

def save_history(entry):
    history = load_history()
    history.append(entry)
    with open(HISTORY_FILE, "w") as f:
        json.dump(history[-10:], f)  # keep last 10

# ── LOAD MODELS ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    cal_model = pickle.load(open('calorie_model.pkl', 'rb'))
    fit_model = pickle.load(open('fitness_model.pkl', 'rb'))
    le_fit    = pickle.load(open('le_fit.pkl', 'rb'))
    return cal_model, fit_model, le_fit

try:
    cal_model, fit_model, le_fit = load_models()
    models_ok = True
except:
    models_ok = False

# ── GROQ ───────────────────────────────────────────────────────────────────────
def generate_plan(calories, protein, carbs, fat, fitness_level, goal, diet_type, age, weight):
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    goal_display = {"muscle_gain": "Muscle Gain", "fat_loss": "Fat Loss", "maintain": "Maintain Weight"}[goal]
    prompt = f"""You are an expert fitness coach and nutritionist specializing in Indian diets.
Create a detailed 7-day fitness and nutrition plan for:
- Age: {age} years | Weight: {weight} kg | Goal: {goal_display}
- Diet: {diet_type} | Fitness Level: {fitness_level}
- Daily Calories: {int(calories)} kcal | Protein: {int(protein)}g | Carbs: {int(carbs)}g | Fat: {int(fat)}g

Structure your response exactly like this:

## 🍽️ Sample Daily Meal Plan
(Indian food — dal, roti, rice, paneer, eggs, soya chunks, sprouts based on diet type)
List: breakfast, mid-morning snack, lunch, pre-workout, dinner, post-workout with approx calories.

## 🏋️ Weekly Workout Split
(Beginner: 3 days full body | Intermediate: PPL | Advanced: PPL + intensity)
List each day with exercises, sets, reps.

## ✅ 3 Key Tips
Specific actionable tips for their exact goal and level.

Keep it practical, Indian-context, and motivating."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1500
    )
    return response.choices[0].message.content

# ── HELPER: BMI ────────────────────────────────────────────────────────────────
def calc_bmi(weight, height_cm):
    h = height_cm / 100
    return round(weight / (h * h), 1)

def bmi_category(bmi):
    if bmi < 18.5: return "Underweight", "#60a5fa", 20
    elif bmi < 25: return "Normal",      "#4ade80", 50
    elif bmi < 30: return "Overweight",  "#fbbf24", 72
    else:          return "Obese",       "#f87171", 90

def health_score(bmi, activity_val, age):
    score = 100
    if bmi < 18.5 or bmi > 30: score -= 25
    elif bmi > 25:              score -= 10
    if activity_val <= 1.2:     score -= 20
    elif activity_val <= 1.375: score -= 10
    if age > 40:                score -= 5
    return max(10, min(100, score))

# ══════════════════════════════════════════════════════════════════════════════
# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="top-bar">
  <div>
    <div class="brand">⚡ Fit<span>AI</span></div>
    <div class="brand-sub">AI-Powered Fitness & Nutrition Coach</div>
  </div>
  <div class="by-tag">Built by <b>Adarsh Kumar Pandey</b></div>
</div>
""", unsafe_allow_html=True)

if not models_ok:
    st.error("⚠️ Models not found. Run `python generate_data.py` then `python train_model.py` first.")
    st.stop()

# ── NAV TABS ───────────────────────────────────────────────────────────────────
tabs = st.tabs(["⚡ Plan Generator", "📊 BMI & Health Score", "🍛 Food Lookup", "📈 Progress Tracker"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PLAN GENERATOR
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown("##### Enter your details")
    c1, c2, c3 = st.columns(3)
    with c1:
        age    = st.number_input("Age", 15, 60, 21, key="t1_age")
        weight = st.number_input("Weight (kg)", 40, 150, 70, key="t1_wt")
    with c2:
        height = st.number_input("Height (cm)", 140, 210, 175, key="t1_ht")
        gender = st.selectbox("Gender", ["Male", "Female"], key="t1_gen")
    with c3:
        act_map = {
            "Sedentary – no exercise":        1.2,
            "Lightly Active – 1-3 days/week": 1.375,
            "Moderately Active – 3-5 days":   1.55,
            "Very Active – 6-7 days/week":    1.725,
        }
        act_label = st.selectbox("Activity Level", list(act_map.keys()), key="t1_act")
        act_val   = act_map[act_label]

    c4, c5 = st.columns(2)
    with c4:
        goal = st.selectbox("Goal", ["muscle_gain","fat_loss","maintain"],
            format_func=lambda x: {"muscle_gain":"💪 Muscle Gain","fat_loss":"🔥 Fat Loss","maintain":"⚖️ Maintain"}[x], key="t1_goal")
    with c5:
        diet = st.selectbox("Diet Type", ["Vegetarian","Eggetarian","Non-Vegetarian"], key="t1_diet")

    st.markdown("<br>", unsafe_allow_html=True)
    gen_btn = st.button("⚡ Predict & Generate My Plan", key="gen")

    if gen_btn:
        gender_enc = 1 if gender == "Male" else 0
        goal_enc   = {"muscle_gain":1,"fat_loss":0,"maintain":2}[goal]
        inp        = np.array([[age, weight, height, gender_enc, act_val, goal_enc]])

        with st.spinner("ML model calculating targets..."):
            calories  = cal_model.predict(inp)[0]
            fit_idx   = fit_model.predict(inp)[0]
            fit_level = le_fit.inverse_transform([fit_idx])[0]
            protein   = weight * (2.0 if goal=="muscle_gain" else 1.6)
            carbs     = (calories * 0.45) / 4
            fat_g     = (calories * 0.25) / 9

        # Save to history
        save_history({
            "date": datetime.now().strftime("%d %b %Y, %I:%M %p"),
            "weight": weight, "goal": goal,
            "calories": int(calories), "protein": int(protein),
            "carbs": int(carbs), "fat": int(fat_g),
            "fitness_level": fit_level
        })

        st.markdown("#### 📊 Your ML-Predicted Targets")
        bmi_val = calc_bmi(weight, height)

        st.markdown(f"""
        <div class="metric-row">
          <div class="m-card" style="--accent:#f97316">
            <div class="m-val">{int(calories)}</div>
            <div class="m-unit">kcal / day</div>
            <div class="m-label">Calories</div>
          </div>
          <div class="m-card" style="--accent:#60a5fa">
            <div class="m-val">{int(protein)}g</div>
            <div class="m-unit">grams / day</div>
            <div class="m-label">Protein</div>
          </div>
          <div class="m-card" style="--accent:#a78bfa">
            <div class="m-val">{int(carbs)}g</div>
            <div class="m-unit">grams / day</div>
            <div class="m-label">Carbs</div>
          </div>
          <div class="m-card" style="--accent:#34d399">
            <div class="m-val">{int(fat_g)}g</div>
            <div class="m-unit">grams / day</div>
            <div class="m-label">Healthy Fats</div>
          </div>
        </div>
        <div style="text-align:center; margin:0.8rem 0 1.5rem;">
          Fitness Level: <span class="lvl {fit_level}">{fit_level}</span>
          &nbsp;·&nbsp; <span style="color:#475569; font-size:0.8rem;">BMI: <b style="color:#f1f5f9">{bmi_val}</b></span>
        </div>
        """, unsafe_allow_html=True)

        # Macro pie via CSS
        p_pct = int(protein*4/calories*100)
        c_pct = int(carbs*4/calories*100)
        f_pct = 100 - p_pct - c_pct
        st.markdown(f"""
        <div class="card" style="margin-bottom:1.5rem">
          <div class="card-title">Macro Split</div>
          <div style="display:flex; gap:1.5rem; align-items:center; margin-top:0.6rem">
            <div style="height:14px; border-radius:99px; overflow:hidden; flex:1; display:flex; gap:2px;">
              <div style="width:{p_pct}%; background:#60a5fa; border-radius:99px 0 0 99px;"></div>
              <div style="width:{c_pct}%; background:#a78bfa;"></div>
              <div style="width:{f_pct}%; background:#34d399; border-radius:0 99px 99px 0;"></div>
            </div>
            <div style="display:flex; gap:1rem; font-size:0.75rem; white-space:nowrap;">
              <span><span style="color:#60a5fa">●</span> <span style="color:#94a3b8">Protein {p_pct}%</span></span>
              <span><span style="color:#a78bfa">●</span> <span style="color:#94a3b8">Carbs {c_pct}%</span></span>
              <span><span style="color:#34d399">●</span> <span style="color:#94a3b8">Fat {f_pct}%</span></span>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### 🗓️ Your Personalized 7-Day Plan")
        with st.spinner("⚡ Building your personalized fitness plan..."):
            try:
                plan = generate_plan(calories, protein, carbs, fat_g, fit_level, goal, diet, age, weight)
                st.markdown(f'<div class="plan-box">{plan}</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"LLM Error: {e}")

        st.markdown("""
        <div style="margin-top:1.5rem; text-align:center; color:#1e293b; font-size:0.72rem;">
          ⚡ Powered by Machine Learning + LLaMA 3.3 · Built by Adarsh Kumar Pandey
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — BMI & HEALTH SCORE
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("##### Check your BMI & Health Score")
    b1, b2, b3 = st.columns(3)
    with b1: bmi_wt  = st.number_input("Weight (kg)", 40, 150, 70, key="b_wt")
    with b2: bmi_ht  = st.number_input("Height (cm)", 140, 210, 175, key="b_ht")
    with b3:
        bmi_act_map = {"Sedentary":1.2,"Lightly Active":1.375,"Moderately Active":1.55,"Very Active":1.725}
        bmi_act_lbl = st.selectbox("Activity Level", list(bmi_act_map.keys()), key="b_act")
        bmi_act_val = bmi_act_map[bmi_act_lbl]
    bmi_age = st.number_input("Age", 15, 60, 21, key="b_age")

    bmi_val  = calc_bmi(bmi_wt, bmi_ht)
    cat, col, pct = bmi_category(bmi_val)
    hscore   = health_score(bmi_val, bmi_act_val, bmi_age)

    if hscore >= 80:  hcol, hlabel = "#4ade80", "Excellent"
    elif hscore >= 60: hcol, hlabel = "#fbbf24", "Good"
    elif hscore >= 40: hcol, hlabel = "#f97316", "Fair"
    else:              hcol, hlabel = "#f87171", "Needs Work"

    left, right = st.columns(2)

    with left:
        st.markdown(f"""
        <div class="card" style="text-align:center; padding:2rem 1.5rem;">
          <div class="card-title">Body Mass Index</div>
          <div style="font-family:'Syne',sans-serif; font-size:3.5rem; font-weight:800; color:{col}; line-height:1; margin:0.5rem 0;">{bmi_val}</div>
          <div style="color:{col}; font-weight:700; font-size:1rem; margin-bottom:1rem;">{cat}</div>
          <div class="bmi-bar-wrap">
            <div class="bmi-bar-fill" style="width:{pct}%; background:{col};"></div>
          </div>
          <div class="bmi-labels">
            <span>Underweight<br>&lt;18.5</span>
            <span>Normal<br>18.5–25</span>
            <span>Overweight<br>25–30</span>
            <span>Obese<br>&gt;30</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown(f"""
        <div class="card" style="text-align:center; padding:2rem 1.5rem;">
          <div class="card-title">Overall Health Score</div>
          <div class="score-ring" style="border-color:{hcol}; margin-top:0.5rem;">
            <div class="score-num" style="color:{hcol};">{hscore}</div>
            <div class="score-label">/ 100</div>
          </div>
          <div style="color:{hcol}; font-weight:700; font-size:1rem; margin-bottom:0.8rem;">{hlabel}</div>
          <div style="color:#475569; font-size:0.78rem; line-height:1.6;">
            Based on BMI, activity level,<br>and age profile.
          </div>
        </div>
        """, unsafe_allow_html=True)

    # Advice
    advice_map = {
        "Underweight": "⚡ Focus on calorie surplus with protein-rich Indian foods — soya chunks, paneer, eggs, dal. Aim for 0.5 kg gain/week.",
        "Normal":      "✅ Great shape! Maintain with balanced meals and consistent workouts. Focus on performance goals.",
        "Overweight":  "🔥 Moderate calorie deficit (300-400 kcal) + cardio 3x/week. Swap refined carbs for millets and oats.",
        "Obese":       "💙 Start with daily 30-min walks + portion control. Even 5–10% weight loss drastically improves health."
    }
    st.markdown(f"""
    <div class="card" style="margin-top:0.5rem; border-left:3px solid {col};">
      <div class="card-title">Recommendation</div>
      <div style="color:#cbd5e1; margin-top:0.4rem; font-size:0.88rem; line-height:1.7;">{advice_map[cat]}</div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — FOOD LOOKUP
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("##### Search Indian Foods")
    search = st.text_input("Type a food name...", placeholder="e.g. paneer, dal, egg", key="food_search")

    # Filter
    results = {k: v for k, v in INDIAN_FOODS.items()
               if search.lower() in k.lower()} if search else INDIAN_FOODS

    st.markdown(f"<div style='color:#475569; font-size:0.75rem; margin-bottom:0.8rem;'>{len(results)} items</div>", unsafe_allow_html=True)

    # Food tracker
    if "tracked" not in st.session_state:
        st.session_state.tracked = []

    for name, info in results.items():
        fc1, fc2 = st.columns([5, 1])
        with fc1:
            st.markdown(f"""
            <div class="food-card">
              <div>
                <div class="food-name">{name}</div>
                <div class="food-meta">P: {info['protein']}g &nbsp;|&nbsp; C: {info['carbs']}g &nbsp;|&nbsp; F: {info['fat']}g &nbsp;·&nbsp; {info['serving']}</div>
              </div>
              <div style="text-align:right;">
                <div class="food-cal">{info['cal']}</div>
                <div class="food-cal-unit">kcal</div>
              </div>
            </div>
            """, unsafe_allow_html=True)
        with fc2:
            if st.button("+ Add", key=f"add_{name}"):
                st.session_state.tracked.append({"name": name, **info})

    # Tracker summary
    if st.session_state.tracked:
        st.markdown("---")
        st.markdown("##### 🧮 Today's Food Log")
        total_cal = sum(f["cal"] for f in st.session_state.tracked)
        total_p   = sum(f["protein"] for f in st.session_state.tracked)
        total_c   = sum(f["carbs"] for f in st.session_state.tracked)
        total_f   = sum(f["fat"] for f in st.session_state.tracked)

        for item in st.session_state.tracked:
            st.markdown(f"<div style='color:#64748b; font-size:0.82rem; padding:0.2rem 0;'>• {item['name']} — <b style='color:#f97316'>{item['cal']} kcal</b></div>", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-row" style="margin-top:0.8rem;">
          <div class="m-card" style="--accent:#f97316"><div class="m-val">{total_cal}</div><div class="m-unit">kcal</div><div class="m-label">Total</div></div>
          <div class="m-card" style="--accent:#60a5fa"><div class="m-val">{total_p}g</div><div class="m-unit">protein</div><div class="m-label">Protein</div></div>
          <div class="m-card" style="--accent:#a78bfa"><div class="m-val">{total_c}g</div><div class="m-unit">carbs</div><div class="m-label">Carbs</div></div>
          <div class="m-card" style="--accent:#34d399"><div class="m-val">{total_f}g</div><div class="m-unit">fat</div><div class="m-label">Fat</div></div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🗑️ Clear Log"):
            st.session_state.tracked = []
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — PROGRESS TRACKER
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    history = load_history()

    if not history:
        st.markdown("""
        <div style="text-align:center; padding:3rem; color:#334155;">
          <div style="font-size:2rem; margin-bottom:0.5rem;">📈</div>
          <div style="font-size:0.9rem;">No history yet — generate a plan first!</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"##### Last {len(history)} sessions saved locally")

        for entry in reversed(history):
            goal_label = {"muscle_gain":"💪 Muscle Gain","fat_loss":"🔥 Fat Loss","maintain":"⚖️ Maintain"}[entry["goal"]]
            st.markdown(f"""
            <div class="history-row">
              <div>
                <div class="h-date">{entry['date']}</div>
                <div class="h-goal">{goal_label} · {entry['fitness_level']}</div>
                <div style="color:#475569; font-size:0.72rem; margin-top:0.2rem;">{entry['weight']} kg · P:{entry['protein']}g C:{entry['carbs']}g F:{entry['fat']}g</div>
              </div>
              <div style="text-align:right;">
                <div class="h-cal">{entry['calories']}</div>
                <div style="color:#475569; font-size:0.7rem;">kcal/day</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        # Calorie trend
        if len(history) >= 2:
            st.markdown("---")
            st.markdown("##### Calorie Trend")
            cals   = [h["calories"] for h in history]
            dates  = [h["date"].split(",")[0] for h in history]
            max_c  = max(cals); min_c = min(cals)
            points = ""
            w = 600; h_svg = 120; pad = 30
            for i, c in enumerate(cals):
                x = pad + i * (w - 2*pad) / max(len(cals)-1, 1)
                y = h_svg - pad - (c - min_c) / max(max_c - min_c, 1) * (h_svg - 2*pad)
                points += f"{x},{y} "
            st.markdown(f"""
            <div class="card">
              <svg viewBox="0 0 {w} {h_svg}" style="width:100%; height:auto;">
                <polyline points="{points.strip()}" fill="none" stroke="#f97316" stroke-width="2.5" stroke-linejoin="round"/>
                {''.join(f'<circle cx="{pad + i*(w-2*pad)/max(len(cals)-1,1)}" cy="{h_svg-pad-(c-min_c)/max(max_c-min_c,1)*(h_svg-2*pad)}" r="4" fill="#f97316"/>' for i,c in enumerate(cals))}
              </svg>
              <div style="display:flex; justify-content:space-between; color:#334155; font-size:0.65rem; margin-top:0.2rem;">
                {''.join(f'<span>{d}</span>' for d in dates)}
              </div>
            </div>
            """, unsafe_allow_html=True)

        if st.button("🗑️ Clear All History"):
            if os.path.exists(HISTORY_FILE):
                os.remove(HISTORY_FILE)
            st.rerun()
