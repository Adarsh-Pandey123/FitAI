import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_absolute_error

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv('fitness_data.csv')
df.dropna(inplace=True)

# ── Encode categoricals ────────────────────────────────────────────────────────
le_goal = LabelEncoder()
df['goal_enc'] = le_goal.fit_transform(df['goal'])

le_fit = LabelEncoder()
df['fit_enc'] = le_fit.fit_transform(df['fitness_level'])

FEATURES = ['age', 'weight', 'height', 'gender', 'activity', 'goal_enc']
X = df[FEATURES]

# ── Model 1: Calorie Regression ────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, df['calories'], test_size=0.2, random_state=42)
cal_model = RandomForestRegressor(n_estimators=200, random_state=42)
cal_model.fit(X_train, y_train)
cal_preds = cal_model.predict(X_test)
print(f"Calorie Model  →  MAE: {mean_absolute_error(y_test, cal_preds):.1f} kcal")

# ── Model 2: Fitness Level Classifier ─────────────────────────────────────────
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, df['fit_enc'], test_size=0.2, random_state=42)
fit_model = RandomForestClassifier(n_estimators=200, random_state=42)
fit_model.fit(X_train2, y_train2)
fit_preds = fit_model.predict(X_test2)
print(f"Fitness Model  →  Accuracy: {accuracy_score(y_test2, fit_preds)*100:.1f}%")

# ── Save everything ────────────────────────────────────────────────────────────
pickle.dump(cal_model,  open('calorie_model.pkl', 'wb'))
pickle.dump(fit_model,  open('fitness_model.pkl', 'wb'))
pickle.dump(le_goal,    open('le_goal.pkl', 'wb'))
pickle.dump(le_fit,     open('le_fit.pkl', 'wb'))

print("\n✅ All models saved successfully!")
print("   calorie_model.pkl  |  fitness_model.pkl  |  le_goal.pkl  |  le_fit.pkl")
