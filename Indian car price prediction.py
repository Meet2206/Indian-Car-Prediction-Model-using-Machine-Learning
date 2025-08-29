import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv("Car Sell Dataset.csv", low_memory=False)


# Quick check
print(df.head())

# Calculate depreciation percentage
# Convert to numeric, turn invalid values into NaN
df["New Car Price"] = pd.to_numeric(df["New Car Price"], errors="coerce")
df["Depreciated Price"] = pd.to_numeric(df["Depreciated Price"], errors="coerce")

# Now calculate depreciation
df["Depreciation %"] = ((df["New Car Price"] - df["Depreciated Price"]) / df["New Car Price"]) * 100


# Select features
features = [
    "Brand", "Model Name", "Model Variant", "Car Type",
    "Transmission", "Fuel Type", "Year", "Kilometers",
    "Owner", "State", "Accidental", "Insurance Price"
]

target = "Depreciated Price"

df_encoded = df.copy()
label_encoders = {}

for col in df_encoded[features].select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    label_encoders[col] = le

X = df_encoded[features]
y = df_encoded[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Check for NaN in y_test or y_pred
print("NaNs in y_test:", np.isnan(y_test).sum())
print("NaNs in y_pred:", np.isnan(y_pred).sum())

# Drop NaNs just in case (aligns both arrays)
mask = ~np.isnan(y_test) & ~np.isnan(y_pred)
y_test_clean = y_test[mask]
y_pred_clean = y_pred[mask]

# Evaluation metrics
print("MAE:", mean_absolute_error(y_test_clean, y_pred_clean))
print("R² Score:", r2_score(y_test_clean, y_pred_clean))

# graphical representation

plt.figure(figsize=(8,6))
plt.scatter(y_test_clean, y_pred_clean, alpha=0.5)
plt.plot([y_test_clean.min(), y_test_clean.max()],
         [y_test_clean.min(), y_test_clean.max()],
         'r--', lw=2)  # ideal line
plt.xlabel("Actual Depreciated Price")
plt.ylabel("Predicted Depreciated Price")
plt.title("Actual vs Predicted Car Prices")
plt.show()

# Pie chart of top 5 car brands
# Count of cars per Brand
top_brands = brand_counts[:5]
others = brand_counts[5:].sum()
top_brands["Others"] = others

plt.pie(top_brands, labels=top_brands.index, autopct='%1.1f%%', startangle=140)
plt.title("Top 5 Car Brands + Others")
plt.show()

# Prepare data
fuel_counts = df["Fuel Type"].value_counts()
trans_counts = df["Transmission"].value_counts()
accidental_counts = df["Accidental"].value_counts()

# Create subplots
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Fuel Type Pie
axes[0].pie(fuel_counts, labels=fuel_counts.index, autopct='%1.1f%%', startangle=140)
axes[0].set_title("Cars by Fuel Type")

# Transmission Pie
axes[1].pie(trans_counts, labels=trans_counts.index, autopct='%1.1f%%', startangle=140)
axes[1].set_title("Cars by Transmission")

# Accidental Pie
axes[2].pie(accidental_counts, labels=accidental_counts.index, autopct='%1.1f%%', startangle=140)
axes[2].set_title("Accidental vs Non-Accidental")

# Display
plt.tight_layout()
plt.show()

#Pie chart of cars by Fuel Type, Transmission, and Accidental status
# Prepare data
fuel_counts = df["Fuel Type"].value_counts()
trans_counts = df["Transmission"].value_counts()
accidental_counts = df["Accidental"].value_counts()

# Create subplots
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Fuel Type Pie
axes[0].pie(fuel_counts, labels=fuel_counts.index, autopct='%1.1f%%', startangle=140)
axes[0].set_title("Cars by Fuel Type")

# Transmission Pie
axes[1].pie(trans_counts, labels=trans_counts.index, autopct='%1.1f%%', startangle=140)
axes[1].set_title("Cars by Transmission")

# Accidental Pie
axes[2].pie(accidental_counts, labels=accidental_counts.index, autopct='%1.1f%%', startangle=140)
axes[2].set_title("Accidental vs Non-Accidental")

# Display
plt.tight_layout()
plt.show()
