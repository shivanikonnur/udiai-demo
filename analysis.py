import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression

# --------------------------------
# CREATE FIGURES FOLDER SAFELY
# --------------------------------
os.makedirs("figures", exist_ok=True)

# --------------------------------
# DATA LOADING
# --------------------------------
def load_csvs(folder_path):
    files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith(".csv")
    ]
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


df_enrol = load_csvs("DATA/api_data_aadhar_enrolment")
df_bio = load_csvs("DATA/api_data_aadhar_biometric")
df_demo = load_csvs("DATA/api_data_aadhar_demographic")

# --------------------------------
# PREPROCESSING
# --------------------------------
def preprocess(df):
    df = df.copy()

    df['date'] = pd.to_datetime(df['date'], errors='coerce', dayfirst=True)
    df = df.dropna(subset=['date', 'state'])

    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month

    numeric_cols = df.select_dtypes(include='number').columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    return df


df_enrol = preprocess(df_enrol)
df_bio = preprocess(df_bio)
df_demo = preprocess(df_demo)

# --------------------------------
# MONTHLY AGGREGATION
# --------------------------------
def aggregate_monthly(df):
    numeric_cols = df.select_dtypes(include='number').columns
    return df.groupby(['year', 'month', 'state'], as_index=False)[numeric_cols].sum()


enrol_m = aggregate_monthly(df_enrol)
bio_m = aggregate_monthly(df_bio)
demo_m = aggregate_monthly(df_demo)

# --------------------------------
# MASTER DATASET
# --------------------------------
df_master = (
    enrol_m.merge(bio_m, on=['year', 'month', 'state'], how='left')
           .merge(demo_m, on=['year', 'month', 'state'], how='left')
)

df_master.fillna(0, inplace=True)

# --------------------------------
# FEATURE ENGINEERING
# --------------------------------
df_master['total_enrolments'] = df_master[
    ['age_0_5', 'age_5_17', 'age_18_greater']
].sum(axis=1)

df_master['total_updates'] = (
    df_master.filter(like='bio_').sum(axis=1) +
    df_master.filter(like='demo_').sum(axis=1)
)

df_master['update_intensity'] = (
    df_master['total_updates'] / (df_master['total_enrolments'] + 1)
)

# --------------------------------
# AI – ANOMALY DETECTION
# --------------------------------
features = df_master[['total_enrolments', 'total_updates', 'update_intensity']]
X_scaled = StandardScaler().fit_transform(features)

iso = IsolationForest(contamination=0.03, random_state=42)
df_master['anomaly_flag'] = iso.fit_predict(X_scaled)
df_master['anomaly_flag'] = df_master['anomaly_flag'].map(
    {-1: 'Anomaly', 1: 'Normal'}
)

plt.figure(figsize=(10,5))
sns.scatterplot(
    data=df_master,
    x='total_enrolments',
    y='total_updates',
    hue='anomaly_flag'
)
plt.title("AI-based Anomaly Detection in Aadhaar Activity")
plt.savefig("figures/anomaly_detection.png")
plt.show()

# --------------------------------
# ML – PREDICTION
# --------------------------------
trend = df_master.groupby(['year','month'], as_index=False)['total_updates'].sum()
trend['time_index'] = np.arange(len(trend))

lr = LinearRegression()
lr.fit(trend[['time_index']], trend['total_updates'])
trend['predicted_updates'] = lr.predict(trend[['time_index']])

plt.figure(figsize=(10,5))
plt.plot(trend['total_updates'], label='Actual')
plt.plot(trend['predicted_updates'], '--', label='Predicted')
plt.legend()
plt.title("ML-based Prediction of Aadhaar Update Load")
plt.savefig("figures/update_prediction.png")
plt.show()

# =====================================================
# UNIVARIATE ANALYSIS
# =====================================================
# Enrolment Trend
enrol_uni = enrol_m.groupby(['year','month'], as_index=False)[
    ['age_0_5','age_5_17','age_18_greater']
].sum()
enrol_uni['total'] = enrol_uni[
    ['age_0_5','age_5_17','age_18_greater']
].sum(axis=1)

plt.figure(figsize=(10,5))
plt.plot(enrol_uni['total'])
plt.title("Univariate: Aadhaar Enrolment Trend")
plt.savefig("figures/univariate_enrolment.png")
plt.show()

# Demographic Updates
demo_uni = demo_m.groupby(['year','month'], as_index=False).sum(numeric_only=True)
demo_uni['total'] = demo_uni.drop(columns=['year','month','state'], errors='ignore').sum(axis=1)

plt.figure(figsize=(10,5))
plt.plot(demo_uni['total'], color='green')
plt.title("Univariate: Demographic Update Trend")
plt.savefig("figures/univariate_demographic.png")
plt.show()

# Biometric Updates
bio_uni = bio_m.groupby(['year','month'], as_index=False).sum(numeric_only=True)
bio_uni['total'] = bio_uni.drop(columns=['year','month','state'], errors='ignore').sum(axis=1)

plt.figure(figsize=(10,5))
plt.plot(bio_uni['total'], color='purple')
plt.title("Univariate: Biometric Update Trend")
plt.savefig("figures/univariate_biometric.png")
plt.show()

# =====================================================
# BIVARIATE ANALYSIS
# =====================================================
plt.figure(figsize=(8,6))
sns.scatterplot(data=df_master, x='total_enrolments', y='total_updates')
plt.title("Bivariate: Enrolments vs Updates")
plt.savefig("figures/bivariate_enrol_vs_updates.png")
plt.show()

state_enrol = df_master.groupby('state')['total_enrolments'].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(10,5))
state_enrol.plot(kind='bar')
plt.title("Bivariate: Top States by Enrolment")
plt.savefig("figures/bivariate_state_enrolment.png")
plt.show()

# =====================================================
# TRIVARIATE ANALYSIS
# =====================================================
heatmap_data = df_master.groupby(['state','month'])['update_intensity'].mean().unstack()

plt.figure(figsize=(12,6))
sns.heatmap(heatmap_data, cmap='Reds')
plt.title("Trivariate: State × Month × Update Intensity")
plt.savefig("figures/trivariate_heatmap.png")
plt.show()

age_time = enrol_m.groupby(['year','month'], as_index=False)[
    ['age_0_5','age_5_17','age_18_greater']
].sum()

plt.figure(figsize=(10,5))
plt.plot(age_time['age_0_5'], label='0–5')
plt.plot(age_time['age_5_17'], label='5–17')
plt.plot(age_time['age_18_greater'], label='18+')
plt.legend()
plt.title("Trivariate: Age Group × Time × Enrolment")
plt.savefig("figures/trivariate_age_time.png")
plt.show()

print("✅ Analysis complete. All graphs generated.")
