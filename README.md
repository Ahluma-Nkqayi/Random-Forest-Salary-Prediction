# 💼 Predicting Salaries Based on Years of Experience Using Random Forest Regression  

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Random%20Forest-brightgreen)
![GUI](https://img.shields.io/badge/Interface-Tkinter-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)

---

## 📘 Project Overview  
This project demonstrates how **Machine Learning** can be applied to predict employee **salaries** based on **Years of Experience** and **Position Level**.  
It uses the **Random Forest Regression** algorithm to capture complex, non-linear patterns that simple linear models cannot model effectively.  

Developed as part of **ICT Electives 362S – Machine Learning (2025)**.

---

## 🧠 Problem Statement  
HR departments often require accurate salary prediction tools for **fairness, budgeting, and planning**.  
Traditional linear regression models assume straight-line relationships, which fail to capture **position-specific jumps** or **complex interactions** between experience and salary.  

This project builds a **robust and interactive prediction tool** using Random Forest Regression.

---

## ⚙️ Model Details  
- **Algorithm:** `RandomForestRegressor` (from `sklearn`)  
- **n_estimators:** 200  
- **max_depth:** 5  
- **Validation:** 5-fold cross-validation  
- **R² Score:** 0.98  
- **Mean Squared Error (MSE):** ≈ 2.5E  
- **Dataset:** Synthetic dataset containing 10 unique position levels (e.g., Analyst → CEO)  

---

## 💻 Features  
✅ Predicts salary based on **Years of Experience** and **Position Level**  
✅ Interactive **Tkinter GUI** for user input  
✅ Displays **predicted salary** and **confidence interval**  
✅ Visualization of **actual vs predicted** salary data  
✅ Includes **feature importance analysis**  

---

## 🧩 Technologies Used  
| Category | Tools |
|-----------|-------|
| Programming Language | Python |
| Libraries | `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn` |
| GUI Framework | Tkinter |
| IDE | Visual Studio Code / Jupyter Notebook |

---

## 📊 Results  
- The Random Forest model achieved a **high accuracy (R² = 0.98)**  
- Effectively modeled **non-linear salary patterns**  
- Outperformed **Linear Regression**, which oversimplified relationships  

📈 **Key Takeaway:**  
Random Forest captures both **gradual increases** and **step-wise jumps** in salaries, making it ideal for HR salary structure prediction.

---

## 🧮 Comparison Summary  

| Model | Characteristics | Performance |
|--------|-----------------|--------------|
| **Linear Regression** | Models straight-line relationships | ❌ Struggles with complex data |
| **Random Forest Regression** | Models non-linear, step-like patterns | ✅ Excellent accuracy & interpretability |

---

## 🚀 Future Improvements  
- Integrate **real-world HR datasets**  
- Include additional features such as **education, department, or performance score**  
- Deploy model using **Flask** or **Streamlit** for web access  
- Add **export to Excel/PDF** functionality for HR reporting  

---

## 🧑🏽‍💻 How to Run  

### 1️⃣ Clone the repository  
```bash
git clone https://github.com/<your-username>/Random-Forest-Salary-Prediction.git
cd Random-Forest-Salary-Prediction
