# ⚡ FitAI — AI-Powered Fitness & Nutrition Coach

> Built by **Adarsh Kumar Pandey** · B.Tech CSE (AI & ML) · JSSATEN, Noida

A production-ready fitness web app combining **Machine Learning** and **Large Language Models** to deliver personalized Indian diet and workout plans — all in a clean, dark UI.

🔗 **Live Demo:**(https://m3g4bwehqnud6ck35fgbwd.streamlit.app/)

---

## 🚀 Features

### ⚡ Plan Generator
- Predicts daily calorie target using a trained **Random Forest Regressor**
- Classifies fitness level (Beginner / Intermediate / Advanced) using **Random Forest Classifier**
- Generates a complete **7-day Indian meal + workout plan** using **LLaMA 3.3 via Groq API**
- Displays a real-time **macro split bar** (Protein / Carbs / Fat %)

### 📊 BMI & Health Score
- Visual BMI bar with category detection (Underweight / Normal / Overweight / Obese)
- Composite health score (0–100) based on BMI, activity level, and age
- Personalised diet + lifestyle recommendations

### 🍛 Indian Food Lookup
- Database of 24+ common Indian foods with full macro breakdown
- Search by name, add to daily log, track total calories + macros in real time

### 📈 Progress Tracker
- Every plan generation is auto-saved locally
- Session history with goal, weight, and macro breakdown
- SVG calorie trend graph across sessions

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| ML Models | scikit-learn (Random Forest Regressor + Classifier) |
| LLM | LLaMA 3.3 70B via Groq API |
| Frontend | Streamlit + Custom CSS |
| Data | Synthetic dataset (2000 samples, Harris-Benedict BMR formula) |
| Language | Python 3.10+ |

---

## 📁 Project Structure

```
fitai/
├── app.py                  # Main Streamlit application
├── generate_data.py        # Synthetic dataset generation
├── train_model.py          # ML model training
├── requirements.txt        # Dependencies
├── .streamlit/
│   └── secrets.toml        # API key (local only, never pushed)
├── calorie_model.pkl       # Trained regression model
├── fitness_model.pkl       # Trained classifier model
├── le_fit.pkl              # Label encoder
└── le_goal.pkl             # Label encoder
```

---

## ⚙️ Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/fitai.git
cd fitai

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate data and train models
python generate_data.py
python train_model.py

# 5. Add your Groq API key
mkdir .streamlit
echo '[secrets]\nGROQ_API_KEY = "gsk_your_key_here"' > .streamlit/secrets.toml

# 6. Run the app
streamlit run app.py
```

---

## 🧠 ML Model Details

| Model | Type | Algorithm | Metric |
|---|---|---|---|
| Calorie Predictor | Regression | Random Forest (200 trees) | MAE ~15 kcal |
| Fitness Classifier | Classification | Random Forest (200 trees) | Accuracy ~99% |

**Dataset:** 2,000 synthetic samples generated using the Harris-Benedict BMR formula with TDEE multipliers. Goal-based calorie adjustments: +300 kcal (muscle gain), −400 kcal (fat loss).

**Features used:** Age, Weight, Height, Gender, Activity Level, Goal

---


## 📄 License

MIT License — free to use and modify with attribution.

---

*Made with 🔥 by Adarsh Kumar Pandey*
