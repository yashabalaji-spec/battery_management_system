# =========================================================
# BATTERY FIRE MANAGEMENT SYSTEM
# =========================================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

# =========================================================
# LOAD DATASET
# =========================================================

data = pd.read_csv(r"ev_battery_charging_data.csv")

# Remove extra spaces from column names
data.columns = data.columns.str.strip()

print("\n✅ DATASET LOADED SUCCESSFULLY\n")

print("Dataset Columns:\n")
print(data.columns)

# =========================================================
# AUTO DETECT REQUIRED COLUMNS
# =========================================================

temp_col = None
voltage_col = None
current_col = None

for col in data.columns:

    lower_col = col.lower()

    if "temp" in lower_col:
        temp_col = col

    if "volt" in lower_col:
        voltage_col = col

    if "current" in lower_col:
        current_col = col

# =========================================================
# CHECK COLUMNS
# =========================================================

if temp_col is None:
    raise Exception("❌ Temperature column not found")

if voltage_col is None:
    raise Exception("❌ Voltage column not found")

if current_col is None:
    raise Exception("❌ Current column not found")

print(f"\n✅ Temperature Column : {temp_col}")
print(f"✅ Voltage Column    : {voltage_col}")
print(f"✅ Current Column    : {current_col}")

# =========================================================
# HANDLE MISSING VALUES
# =========================================================

data = data.fillna(data.mean(numeric_only=True))

# =========================================================
# FIRE RISK CLASSIFICATION
# =========================================================

def classify_fire_risk(voltage, current, temp):

    # FIRE RISK
    if temp > 90 or voltage > 130 or current > 20:
        return 3

    # CRITICAL
    elif temp > 70 or voltage > 120 or current > 15:
        return 2

    # HEATING
    elif temp > 50 or voltage > 100 or current > 10:
        return 1

    # SAFE
    else:
        return 0

# Apply fire risk classification

data["Fire_Risk_Level"] = data.apply(

    lambda row: classify_fire_risk(
        row[voltage_col],
        row[current_col],
        row[temp_col]
    ),

    axis=1
)

# =========================================================
# BATTERY HEALTH STATUS
# =========================================================

def battery_health(temp):

    if temp < 40:
        return "Healthy"

    elif temp < 60:
        return "Moderate"

    else:
        return "Degraded"

data["Battery_Health"] = data[temp_col].apply(
    battery_health
)

# =========================================================
# THERMAL STABILITY INDEX
# =========================================================

# Formula:
# TSI = (Voltage × Current × Temperature) / 1000

data["TSI"] = (
    data[voltage_col]
    * data[current_col]
    * data[temp_col]
) / 1000

print("\n⚡ Thermal Stability Index Formula Used:")
print("TSI = (Voltage × Current × Temperature) / 1000")

# =========================================================
# COOLING SYSTEM RECOMMENDATION
# =========================================================

def cooling_type(risk):

    if risk == 0:
        return "Natural Air Cooling"

    elif risk == 1:
        return "Forced Air Cooling"

    elif risk == 2:
        return "Liquid Cooling"

    else:
        return "Immersion Cooling + Emergency Shutdown"

data["Cooling_System"] = data["Fire_Risk_Level"].apply(
    cooling_type
)

# =========================================================
# DISPLAY FIRE RISK DISTRIBUTION
# =========================================================

print("\n📊 FIRE RISK LEVEL COUNTS:\n")
print(data["Fire_Risk_Level"].value_counts())

# =========================================================
# FEATURES AND TARGET
# =========================================================

X = data[[voltage_col, current_col, temp_col]]

y = data["Fire_Risk_Level"]

# =========================================================
# FEATURE SCALING
# =========================================================

scaler = StandardScaler()

X = scaler.fit_transform(X)

print("\n✅ Feature Scaling Applied Using StandardScaler")

# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,
    random_state=42
)

# =========================================================
# MACHINE LEARNING MODELS
# =========================================================

models = {

    "Logistic Regression":
        LogisticRegression(
            class_weight="balanced",
            max_iter=2000
        ),

    "Decision Tree":
        DecisionTreeClassifier(random_state=42),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced"
        ),

    "Gradient Boosting":
        GradientBoostingClassifier(),

    "Neural Network":
        MLPClassifier(
            hidden_layer_sizes=(50, 50),
            max_iter=1000
        )
}

# =========================================================
# TRAIN AND EVALUATE MODELS
# =========================================================

results = []

for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    score = accuracy_score(y_test, y_pred)

    results.append([name, score])

# =========================================================
# MODEL COMPARISON TABLE
# =========================================================

results_df = pd.DataFrame(

    results,

    columns=[
        "Model Type",
        "Prediction Accuracy"
    ]
)

print("\n📊 MODEL COMPARISON TABLE:\n")
print(results_df)

# =========================================================
# BEST MODEL
# =========================================================

best_model = results_df.loc[
    results_df["Prediction Accuracy"].idxmax()
]

print("\n🏆 BEST MODEL:")

print(
    f"{best_model['Model Type']} "
    f"with accuracy "
    f"{best_model['Prediction Accuracy']:.2f}"
)

# =========================================================
# GRAPH 1
# MODEL ACCURACY
# =========================================================

plt.figure(figsize=(9, 5))

plt.bar(
    results_df["Model Type"],
    results_df["Prediction Accuracy"]
)

plt.xlabel("Machine Learning Models")
plt.ylabel("Accuracy")
plt.title("Battery Fire Management - Model Comparison")

plt.xticks(rotation=10)

plt.grid(axis='y')

plt.tight_layout()

plt.show(block=False)
plt.pause(3)
plt.close()

# =========================================================
# GRAPH 2
# TEMPERATURE VS FIRE RISK
# =========================================================

plt.figure(figsize=(8, 5))

bins = pd.cut(data[temp_col], bins=12)

group = data.groupby(bins)["Fire_Risk_Level"].mean()

centers = [i.mid for i in group.index]

plt.plot(
    centers,
    group.values,
    marker='o',
    linewidth=3
)

plt.xlabel("Temperature")
plt.ylabel("Fire Risk Level")
plt.title("Temperature vs Battery Fire Risk")

plt.grid()

plt.tight_layout()

plt.show(block=False)
plt.pause(3)
plt.close()

# =========================================================
# GRAPH 3
# VOLTAGE VS FIRE RISK
# =========================================================

plt.figure(figsize=(8, 5))

bins = pd.cut(data[voltage_col], bins=12)

group = data.groupby(bins)["Fire_Risk_Level"].mean()

centers = [i.mid for i in group.index]

plt.plot(
    centers,
    group.values,
    marker='o',
    linewidth=3
)

plt.xlabel("Voltage")
plt.ylabel("Fire Risk Level")
plt.title("Voltage vs Battery Fire Risk")

plt.grid()

plt.tight_layout()

plt.show(block=False)
plt.pause(3)
plt.close()

# =========================================================
# GRAPH 4
# CURRENT VS FIRE RISK
# =========================================================

plt.figure(figsize=(8, 5))

bins = pd.cut(data[current_col], bins=12)

group = data.groupby(bins)["Fire_Risk_Level"].mean()

centers = [i.mid for i in group.index]

plt.plot(
    centers,
    group.values,
    marker='o',
    linewidth=3
)

plt.xlabel("Current")
plt.ylabel("Fire Risk Level")
plt.title("Current vs Battery Fire Risk")

plt.grid()

plt.tight_layout()

plt.show(block=False)
plt.pause(3)
plt.close()

# =========================================================
# GRAPH 5
# TSI VS FIRE RISK
# =========================================================

plt.figure(figsize=(8, 5))

bins = pd.cut(data["TSI"], bins=12)

group = data.groupby(bins)["Fire_Risk_Level"].mean()

centers = [i.mid for i in group.index]

plt.plot(
    centers,
    group.values,
    marker='o',
    linewidth=3
)

plt.xlabel("Thermal Stability Index")
plt.ylabel("Fire Risk Level")
plt.title("TSI vs Battery Fire Risk")

plt.grid()

plt.tight_layout()

plt.show(block=False)
plt.pause(3)
plt.close()

# =========================================================
# GRAPH 6
# COOLING SYSTEM DISTRIBUTION
# =========================================================

plt.figure(figsize=(9, 5))

cooling_counts = data["Cooling_System"].value_counts()

plt.bar(
    cooling_counts.index,
    cooling_counts.values
)

plt.xlabel("Cooling System Type")
plt.ylabel("Count")
plt.title("Cooling System Recommendation Distribution")

plt.xticks(rotation=10)

plt.grid(axis='y')

plt.tight_layout()

plt.show(block=False)
plt.pause(3)
plt.close()

# =========================================================
# USER INPUT SECTION
# =========================================================

print("\n⚡ BATTERY FIRE MANAGEMENT SYSTEM\n")

voltage = float(input("Enter Voltage (V): "))
current = float(input("Enter Current (A): "))
temperature = float(input("Enter Temperature (°C): "))

# =========================================================
# USER TSI CALCULATION
# =========================================================

tsi = (voltage * current * temperature) / 1000

print(f"\n⚡ Thermal Stability Index = {tsi:.2f}")

# =========================================================
# USER INPUT DATAFRAME
# =========================================================

user_input = pd.DataFrame(

    [[voltage, current, temperature]],

    columns=[
        voltage_col,
        current_col,
        temp_col
    ]
)

# APPLY SAME SCALING

user_input_scaled = scaler.transform(user_input)

# =========================================================
# PREDICTIONS + COOLING SYSTEM
# =========================================================

print("\n📊 FIRE RISK PREDICTIONS:\n")

for name, model in models.items():

    prediction = model.predict(user_input_scaled)[0]

    # SAFE
    if prediction == 0:

        risk = "✅ SAFE"
        cooling = "Natural Air Cooling"
        action = "Normal Battery Operation"

    # HEATING
    elif prediction == 1:

        risk = "⚠️ HEATING DETECTED"
        cooling = "Forced Air Cooling"
        action = "Activate Cooling Fan"

    # CRITICAL
    elif prediction == 2:

        risk = "🔥 CRITICAL CONDITION"
        cooling = "Liquid Cooling"
        action = "Reduce Charging Current"

    # FIRE RISK
    else:

        risk = "🚨 FIRE RISK"
        cooling = "Immersion Cooling + Emergency Shutdown"
        action = "Disconnect Battery Immediately"

    # =====================================================
    # BATTERY HEALTH
    # =====================================================

    if temperature < 40:
        health = "Healthy"

    elif temperature < 60:
        health = "Moderate"

    else:
        health = "Degraded"

    # =====================================================
    # DISPLAY RESULTS
    # =====================================================

    print(f"\n{name}")

    print(f"Risk Status      : {risk}")
    print(f"Cooling Type     : {cooling}")
    print(f"Suggested Action : {action}")
    print(f"Battery Health   : {health}")

# =========================================================
# ALERT SYSTEM
# =========================================================

if temperature > 90:

    print("\n🚨 EMERGENCY ALERT")
    print("Battery temperature exceeds safe limit")
    print("Fire suppression system activated")

elif temperature > 70:

    print("\n⚠️ WARNING ALERT")
    print("Battery entering critical thermal region")
    print("Liquid cooling activated")

elif temperature > 50:

    print("\n⚠️ HEATING ALERT")
    print("Forced air cooling activated")

# =========================================================
# FINAL MESSAGE
# =========================================================

print("\n⚡ BATTERY FIRE MANAGEMENT ANALYSIS COMPLETED")