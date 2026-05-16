# 🔋 Battery Fire Management System

## 📌 Overview

The Battery Fire Management System is a Python-based machine learning project developed to monitor EV battery conditions and predict battery fire risk levels.

The system analyzes important battery parameters such as:

- Voltage
- Current
- Temperature

Using these parameters, the project classifies battery conditions into different fire risk categories and recommends suitable cooling systems and safety actions.

---

# 🚀 Features

✅ Automatic dataset column detection  
✅ Fire risk prediction using Machine Learning  
✅ Thermal Stability Index (TSI) calculation  
✅ Battery health monitoring  
✅ Cooling system recommendation  
✅ Emergency alert system  
✅ Data visualization using graphs  
✅ Comparison of multiple ML algorithms  

---

# 🧠 Machine Learning Models Used

The following machine learning algorithms are implemented and compared:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Gradient Boosting
5. Neural Network (MLP Classifier)

---

# 📊 Fire Risk Levels

| Risk Level | Condition | Description |
|------------|------------|-------------|
| 0 | Safe | Normal battery operation |
| 1 | Heating | Battery temperature rising |
| 2 | Critical | Dangerous thermal condition |
| 3 | Fire Risk | High possibility of battery fire |

---

# ⚡ Thermal Stability Index (TSI)

The Thermal Stability Index is calculated using:

TSI = (Voltage × Current × Temperature) / 1000

Higher TSI values indicate increased thermal instability and fire risk.

---

# ❄️ Cooling System Recommendations

| Fire Risk | Cooling System |
|-----------|----------------|
| Safe | Natural Air Cooling |
| Heating | Forced Air Cooling |
| Critical | Liquid Cooling |
| Fire Risk | Immersion Cooling + Emergency Shutdown |

---

# 📈 Graphs Generated

The project generates the following graphs:

- Model Accuracy Comparison
- Temperature vs Fire Risk
- Voltage vs Fire Risk
- Current vs Fire Risk
- TSI vs Fire Risk
- Cooling System Distribution

---

# 🛠️ Technologies Used

- Python
- Pandas
- Matplotlib
- Scikit-learn

---

# 📂 Dataset

The project uses EV battery charging data stored in:

```bash
ev_battery_charging_data.csv
```

---

# ▶️ How to Run the Project

## Step 1: Install Required Libraries

```bash
pip install pandas matplotlib scikit-learn
```

## Step 2: Run the Program

```bash
python your_file_name.py
```

Replace:

```bash
your_file_name.py
```

with your actual Python file name.

---

# 📥 User Input

The user enters:

- Voltage
- Current
- Temperature

The system predicts:

- Fire Risk Level
- Battery Health
- Cooling Recommendation
- Safety Action

---

# 🚨 Alert System

The system provides different alerts based on battery temperature:

- Heating Alert
- Warning Alert
- Emergency Fire Alert

---

# 📌 Applications

- Electric Vehicles (EVs)
- Battery Management Systems (BMS)
- Fire Prevention Systems
- Smart Battery Monitoring
- Industrial Battery Safety

---

# 📷 Sample Output

Example:

```text
Risk Status      : 🔥 CRITICAL CONDITION
Cooling Type     : Liquid Cooling
Suggested Action : Reduce Charging Current
Battery Health   : Degraded
```

---

# 👨‍💻 Author

Developed by Yasha Balaji

---

# 📜 License

This project is developed for educational and research purposes.
