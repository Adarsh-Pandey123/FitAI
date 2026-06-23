import pandas as pd
import numpy as np

np.random.seed(42)
n = 2000

activity_levels = [1.2, 1.375, 1.55, 1.725]
goals = ['muscle_gain', 'fat_loss', 'maintain']

df = pd.DataFrame({
    'age':      np.random.randint(18, 50, n),
    'weight':   np.random.randint(45, 105, n),
    'height':   np.random.randint(150, 195, n),
    'gender':   np.random.choice([0, 1], n),         # 0=Female, 1=Male
    'activity': np.random.choice(activity_levels, n),
    'goal':     np.random.choice(goals, n)
})

# Harris-Benedict BMR
df['bmr'] = np.where(
    df['gender'] == 1,
    88.36 + (13.4 * df['weight']) + (4.8 * df['height']) - (5.7 * df['age']),
    447.6 + (9.2  * df['weight']) + (3.1 * df['height']) - (4.3 * df['age'])
)

df['tdee'] = df['bmr'] * df['activity']

# Calorie target based on goal
df['calories'] = np.where(df['goal'] == 'muscle_gain', df['tdee'] + 300,
                 np.where(df['goal'] == 'fat_loss',    df['tdee'] - 400,
                                                        df['tdee']))

# Macros
df['protein_g'] = df['weight'] * np.where(df['goal'] == 'muscle_gain', 2.0, 1.6)
df['carbs_g']   = (df['calories'] * 0.45) / 4
df['fat_g']     = (df['calories'] * 0.25) / 9

# Fitness level based on activity
df['fitness_level'] = pd.cut(
    df['activity'],
    bins=[0, 1.3, 1.5, 2.0],
    labels=['Beginner', 'Intermediate', 'Advanced']
)

df.to_csv('fitness_data.csv', index=False)
print(f"Dataset created: {len(df)} rows")
print(df.head())
