## 🚀 Live Demo
👉 https://fetalpredictionapp-tg4tavxkroz7cae3cedkx8.streamlit.app

# 🩺 Fetal Health Prediction App (ML + Streamlit)

## 📌 Project Overview

This project predicts fetal health using **Machine Learning on CTG (Cardiotocography) data** and provides results through a **Streamlit web application**.

It classifies fetal health into:

* **Normal**
* **Suspect**
* **Pathological**

---

## 🌿 Important Branch Information

🔹 **`model-comparison` (Recommended Branch)**
This is the **main working branch** of the project.

It includes:

* ✅ Model comparison (Logistic, Random Forest, Gradient Boosting)
* ✅ Best model selection using **Weighted F1 Score**
* ✅ Trained model saved (`.joblib`)
* ✅ Fully functional **Streamlit UI**

👉 Use this branch for:

* Running the app
* Understanding complete ML pipeline

---

🔹 **`main` branch**

* Basic version of the project
* Contains initial implementation

---

## 🧠 Problem Statement

CTG graph interpretation:

* Requires expert doctors
* Can be subjective
* Time-consuming in critical situations

---

## 💡 Solution

This system:

1. Uses **CTG numeric features**
2. Applies Machine Learning models
3. Predicts fetal health condition instantly

⚠️ *Note: This is a decision-support system, not a replacement for doctors.*

---

## 📊 Features Used

* Baseline Value
* Accelerations
* Fetal Movement
* Uterine Contractions
* Light Decelerations
* Severe Decelerations
* Prolongued Decelerations
* Short-Term Variability
* Long-Term Variability
* Histogram Mean
* Histogram Variance

---

## ⚙️ Machine Learning Models

* Logistic Regression
* Random Forest Classifier
* Gradient Boosting Classifier

### ✅ Best Model Selection

* Evaluated using **Weighted F1 Score**
* Best model saved as:

  ```
  best_fetal_health_model.joblib
  ```

---

## 🔄 Data Preprocessing

* Missing values → Median Imputation
* Class imbalance → Undersampling
* Feature scaling → Applied for Logistic Regression

---

## 📈 Evaluation Metrics

* Accuracy
* **Weighted F1 Score (Primary Metric)**

👉 Used because:

* Medical data is sensitive
* Handles class imbalance better

---

## 🚀 Streamlit Application

### Features:

* Input CTG values manually
* Instant prediction
* Clean UI

⚠️ **Note:**
In real-world systems, CTG values are automatically extracted from medical devices.
Manual input is used here for demonstration.

---

## ▶️ How to Run the App

```bash
# Clone repository
git clone <your-repo-link>

# Go to project folder
cd fetal_prediction_app

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py
```

---

## 🏥 Use Cases

* Hospital decision-support systems
* Rural healthcare assistance
* Early detection of fetal distress
* Medical education & training

---

## ⚠️ Limitations

* Depends on input data quality
* Not real-time CTG integration
* Does not replace medical professionals

---

## 🔮 Future Scope

* Real-time CTG device integration
* Deep learning models
* Mobile/web healthcare deployment
* Hospital system integration

---

## 🛠️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Streamlit

---

## 📌 Conclusion

This project demonstrates how **Machine Learning + Healthcare data** can improve fetal health monitoring by providing:

* Faster predictions
* Consistent results
* Decision support for doctors

---

## 👩‍💻 Author

**Vanshika Mahant**
B.Tech CSE (AI/ML)
