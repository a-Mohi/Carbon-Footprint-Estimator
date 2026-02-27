# ============================================================
#   CARBON FOOTPRINT ESTIMATOR
#   SDG 13: Climate Action
#   Predict CO2 Emissions based on Engine Size,
#   Cylinders, and Fuel Type
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# ── Create output folders if not exist ───────────────────────
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

# ============================================================
# STEP 1: LOAD DATA
# ============================================================
print("=" * 60)
print("  STEP 1: LOADING DATA")
print("=" * 60)

df = pd.read_csv("data/co2_emissions.csv")

print(f"Dataset Shape   : {df.shape[0]} rows x {df.shape[1]} columns")
print(f"\nColumn Names:")
for col in df.columns:
    print(f"  - {col}")

print(f"\nFirst 5 Rows:")
print(df.head())

print(f"\nMissing Values:")
print(df.isnull().sum())

# ============================================================
# STEP 2: SELECT FEATURES
# ============================================================
print("\n" + "=" * 60)
print("  STEP 2: SELECTING FEATURES")
print("=" * 60)

# We only need these 4 columns for our project
df = df[['Engine Size(L)', 'Cylinders', 'Fuel Type', 'CO2 Emissions(g/km)']]
print(f"Selected columns: {df.columns.tolist()}")
print(f"Shape after selection: {df.shape}")

# ============================================================
# STEP 3: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================
print("\n" + "=" * 60)
print("  STEP 3: EXPLORATORY DATA ANALYSIS")
print("=" * 60)

print("\nBasic Statistics:")
print(df.describe())

print("\nFuel Type Distribution:")
print(df['Fuel Type'].value_counts())

# Plot 1: CO2 Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df['CO2 Emissions(g/km)'], kde=True, color='green', bins=40)
plt.title('Distribution of CO2 Emissions (g/km)', fontsize=14)
plt.xlabel('CO2 Emissions (g/km)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig("data/plot1_co2_distribution.png", dpi=150)
plt.show()
print("Plot 1 saved: CO2 Distribution")

# Plot 2: Engine Size vs CO2
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='Engine Size(L)', y='CO2 Emissions(g/km)', alpha=0.3, color='steelblue')
plt.title('Engine Size vs CO2 Emissions', fontsize=14)
plt.xlabel('Engine Size (L)')
plt.ylabel('CO2 Emissions (g/km)')
plt.tight_layout()
plt.savefig("data/plot2_engine_vs_co2.png", dpi=150)
plt.show()
print("Plot 2 saved: Engine Size vs CO2")

# Plot 3: Fuel Type vs CO2
plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x='Fuel Type', y='CO2 Emissions(g/km)', palette='Set2')
plt.title('CO2 Emissions by Fuel Type', fontsize=14)
plt.tight_layout()
plt.savefig("data/plot3_fueltype_vs_co2.png", dpi=150)
plt.show()
print("Plot 3 saved: Fuel Type vs CO2")

# ============================================================
# STEP 4: OUTLIER DETECTION ON ENGINE SIZE (IQR METHOD)
# ============================================================
print("\n" + "=" * 60)
print("  STEP 4: OUTLIER DETECTION (IQR METHOD)")
print("=" * 60)

Q1 = df['Engine Size(L)'].quantile(0.25)
Q3 = df['Engine Size(L)'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"Q1            : {Q1}")
print(f"Q3            : {Q3}")
print(f"IQR           : {IQR}")
print(f"Lower Bound   : {lower_bound}")
print(f"Upper Bound   : {upper_bound}")

outliers = df[(df['Engine Size(L)'] < lower_bound) |
            (df['Engine Size(L)'] > upper_bound)]
print(f"Outliers Found: {len(outliers)} rows")

# Plot 4: Before vs After Outlier Removal
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].boxplot(df['Engine Size(L)'])
axes[0].set_title('Engine Size BEFORE Outlier Removal')
axes[0].set_ylabel('Engine Size (L)')

# Remove outliers
df_clean = df[(df['Engine Size(L)'] >= lower_bound) & (df['Engine Size(L)'] <= upper_bound)].copy()

axes[1].boxplot(df_clean['Engine Size(L)'])
axes[1].set_title('Engine Size AFTER Outlier Removal')
axes[1].set_ylabel('Engine Size (L)')
plt.tight_layout()
plt.savefig("data/plot4_outlier_detection.png", dpi=150)
plt.show()
print("Plot 4 saved: Outlier Detection")

print(f"\nRows before cleaning : {len(df)}")
print(f"Rows after cleaning  : {len(df_clean)}")
print(f"Outliers removed     : {len(df) - len(df_clean)}")

# ============================================================
# STEP 5: ONE HOT ENCODING ON FUEL TYPE
# ============================================================
print("\n" + "=" * 60)
print("  STEP 5: ONE HOT ENCODING ON FUEL TYPE")
print("=" * 60)

print(f"Unique Fuel Types: {df_clean['Fuel Type'].unique()}")

df_encoded = pd.get_dummies(df_clean, columns=['Fuel Type'], drop_first=False)

# Convert boolean columns to int
fuel_cols = [c for c in df_encoded.columns if c.startswith('Fuel Type_')]
df_encoded[fuel_cols] = df_encoded[fuel_cols].astype(int)

print(f"\nColumns after One Hot Encoding:")
for col in df_encoded.columns:
    print(f"  - {col}")

print(f"\nShape after OHE: {df_encoded.shape}")
df_encoded.to_csv("data/co2_preprocessed.csv", index=False)
print("Preprocessed data saved: data/co2_preprocessed.csv")

# ============================================================
# STEP 6: TRAIN / TEST SPLIT
# ============================================================
print("\n" + "=" * 60)
print("  STEP 6: TRAIN / TEST SPLIT")
print("=" * 60)

X = df_encoded.drop(columns=['CO2 Emissions(g/km)'])
y = df_encoded['CO2 Emissions(g/km)']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Features      : {X.columns.tolist()}")
print(f"Target        : CO2 Emissions(g/km)")
print(f"Training rows : {X_train.shape[0]}")
print(f"Testing rows  : {X_test.shape[0]}")

# ============================================================
# STEP 7: TRAIN LINEAR REGRESSION MODEL
# ============================================================
print("\n" + "=" * 60)
print("  STEP 7: TRAINING LINEAR REGRESSION MODEL")
print("=" * 60)

model = LinearRegression()
model.fit(X_train, y_train)
print("Model training complete!")

print("\nModel Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"  {feature:30s} : {coef:.4f}")
print(f"  {'Intercept':30s} : {model.intercept_:.4f}")

# ============================================================
# STEP 8: MODEL EVALUATION
# ============================================================
print("\n" + "=" * 60)
print("  STEP 8: MODEL EVALUATION")
print("=" * 60)

y_pred = model.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)

print(f"MAE  (Mean Absolute Error)  : {mae:.2f}")
print(f"MSE  (Mean Squared Error)   : {mse:.2f}")
print(f"RMSE (Root Mean Sq. Error)  : {rmse:.2f}")
print(f"R2   (R-Squared Score)      : {r2:.4f} ({r2*100:.2f}%)")

# Plot 5: Actual vs Predicted
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.3, color='steelblue', label='Predictions')
plt.plot([y_test.min(), y_test.max()],[y_test.min(), y_test.max()], 'r--', linewidth=2, label='Perfect Fit')
plt.xlabel('Actual CO2 Emissions (g/km)')
plt.ylabel('Predicted CO2 Emissions (g/km)')
plt.title('Actual vs Predicted CO2 Emissions', fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig("data/plot5_actual_vs_predicted.png", dpi=150)
plt.show()
print("Plot 5 saved: Actual vs Predicted")

# Plot 6: Residual Plot
residuals = y_test - y_pred
plt.figure(figsize=(8, 5))
plt.scatter(y_pred, residuals, alpha=0.3, color='coral')
plt.axhline(y=0, color='black', linestyle='--', linewidth=1.5)
plt.xlabel('Predicted CO2 Emissions (g/km)')
plt.ylabel('Residuals')
plt.title('Residual Plot', fontsize=14)
plt.tight_layout()
plt.savefig("data/plot6_residuals.png", dpi=150)
plt.show()
print("Plot 6 saved: Residuals")

# ============================================================
# STEP 9: SAVE THE MODEL
# ============================================================
print("\n" + "=" * 60)
print("  STEP 9: SAVING MODEL")
print("=" * 60)

joblib.dump(model, "models/co2_model.pkl")
joblib.dump(X.columns.tolist(), "models/feature_columns.pkl")
print("Model saved      : models/co2_model.pkl")
print("Features saved   : models/feature_columns.pkl")

# ============================================================
# STEP 10: MAKE A PREDICTION
# ============================================================
print("\n" + "=" * 60)
print("  STEP 10: SAMPLE PREDICTION")
print("=" * 60)

# Load model back
loaded_model = joblib.load("models/co2_model.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

# Sample vehicle: 2.0L engine, 4 cylinders, Regular Petrol (X)
sample = {col: 0 for col in feature_columns}
sample['Engine Size(L)'] = 2.0
sample['Cylinders']      = 4
sample['Fuel Type_X']    = 1   # Regular Petrol

sample_df = pd.DataFrame([sample])[feature_columns]
result = loaded_model.predict(sample_df)[0]

print(f"Sample Vehicle  : Engine=2.0L | Cylinders=4 | Fuel=Regular Petrol")
print(f"Predicted CO2   : {result:.2f} g/km")

if result < 150:
    category = "LOW Emission - Eco Friendly!"
elif result < 250:
    category = "MODERATE Emission - Average Vehicle"
elif result < 350:
    category = "HIGH Emission - Consider alternatives"
else:
    category = "VERY HIGH Emission - Poor for environment"

print(f"Category        : {category}")

print("\n" + "=" * 60)
print("  PROJECT COMPLETE - SDG 13: CLIMATE ACTION")
print("=" * 60)