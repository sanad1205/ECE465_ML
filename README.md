# IECE 465 – Introduction to Machine Learning for Engineers  
## Final Project: Marathon Finish Time Prediction

---

## 📌 Project Overview
This project focuses on predicting net marathon finish times using machine learning techniques applied to the 2023 Boston Marathon dataset (26,598 runners). The goal is to model the relationship between runner characteristics and performance metrics to accurately estimate final race outcomes.

---

## 📊 Dataset
- **Source:** 2023 Boston Marathon  
- **Samples:** 26,598 runners  
- **Features:**
  - `age_group` (categorical)
  - `gender` (categorical)
  - `half_time_sec` (numerical)
- **Target:**
  - `finish_net_sec` (numerical)

---

## ⚙️ Methodology
The project follows a structured machine learning pipeline:

### 1. Data Preprocessing
- Removed missing and invalid entries  
- Converted categorical variables  
- Prepared and standardized features  

### 2. Model Development
- **Baseline Model:** Ratio-based prediction using average finish-to-half-time relationship  
- **Linear Regression:** Captures linear relationships between features and finish time  
- **Neural Network Regression:** Learns nonlinear patterns using a multi-layer architecture  

### 3. Model Evaluation
- Mean Absolute Error (MAE)  
- Root Mean Squared Error (RMSE)  
- R-squared (R²)  
- Mean Absolute Percentage Error (MAPE)  

---

## 📈 Results
Models are compared based on prediction accuracy and error metrics:
- Lower MAE and RMSE indicate better performance  
- Neural networks are evaluated against linear models to assess benefits of nonlinear learning  

---

## 📊 Visualizations
The project includes:
- Actual vs. Predicted plots  
- Residual analysis  
- Error distribution histograms  
- Model comparison charts  
- Feature relationship analysis  
- Performance breakdown by age group and gender  

---

## 🎯 Key Insight
Half-marathon split time is the strongest predictor of final finish time, with demographic features providing additional refinement.

---

## 🔍 Extensions
**Sports Betting Style Analysis:**
- Converts finish time predictions into over/under threshold decisions  
- Evaluates classification-style accuracy based on benchmark finish times  

---

## 🛠️ Tools & Technologies
- MATLAB (primary implementation)  
- Machine Learning Toolbox  
- Data visualization and statistical analysis tools  

---

## 👥 Team
- Sanad Sahawneh  
- Brian Allen  
- Mahib Rahman  

---

## 📅 Course Information
**Course:** IECE 465 – Introduction to Machine Learning for Engineers  
**Semester:** Spring 2026  
**Instructor:** Dr. Hany Elgala  
