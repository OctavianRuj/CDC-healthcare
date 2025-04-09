# 🩺 Diabetes Risk Prediction with BRFSS 2015

This project uses data from the CDC’s Behavioral Risk Factor Surveillance System (BRFSS) to build a machine learning model that predicts whether an individual is at risk of having diabetes based on lifestyle and socioeconomic factors. A shorter, questionnaire-based version of the model was also developed using feature selection, offering a practical tool for early risk assessment.

---

## 📊 Overview

- **Dataset:** [CDC BRFSS 2015](https://www.cdc.gov/brfss/)
- **Goal:** Predict binary diabetes risk (0 = non-diabetic, 1 = pre-diabetic or diabetic)
- **Tech:** Python, pandas, scikit-learn, Jupyter Notebook
- **Models:** Random Forest (full model + short form)
- **Features Used:** Health indicators, income, education, BMI, physical/mental health days, etc.

---

## 📌 Key Research Questions

1. **Is the level of education reflected in income?**  
   → Yes, individuals with higher education levels tend to have higher income.

2. **Do the same features that predict diabetes also correlate with income and education?**  
   → Yes, strong negative correlations were found, indicating health disparities related to socioeconomic status.

3. **Can a subset of risk factors accurately predict diabetes?**  
   → Yes, a model using just 7 features still achieved strong performance.

4. **Can we create a short-form BRFSS questionnaire for predicting diabetes risk?**  
   → Yes, using feature selection, we built an interactive questionnaire that keeps performance high while requiring fewer inputs.

---

## ⚙️ Model Performance

| Model           | Accuracy | AUC   | F1 Score |
|------------------|----------|--------|-----------|
| Full Model       | 0.7371   | 0.8090 | 0.7475    |
| Short Form Model | 0.7046   | 0.7682 | 0.7118    |

---

## 🧠 Features Used (Short Form)

- BMI  
- Age  
- General Health  
- Income  
- High Blood Pressure  
- Physical Health (bad days)  
- Education

---

## 🧪 Technologies

- Python (Jupyter Notebook)
- pandas & numpy
- scikit-learn (Random Forest, model evaluation)
- matplotlib & seaborn (visualizations)

---

## 🔍 Usage

You can interact with the short-form diabetes risk prediction tool by running the notebook and answering simple input prompts such as:

```python
What is your BMI? 28.5  
How many days was your physical health not good? 12  
...
