# 📈 Regression Study

## 📌 Overview
This project performs a full regression analysis workflow, moving from exploratory correlation analysis to model fitting and statistical evaluation.  
The study also examines how noise affects coefficient stability and model significance.

---

## 🎯 Objectives
- Identify relationships between variables
- Inspect feature distributions and linear trends
- Fit a regression model and interpret results
- Evaluate robustness under noisy data conditions

---

## 🔍 Exploratory Data Analysis

### Correlation Heatmap
The heatmap highlights the strength and direction of linear relationships between variables.

![Correlation Heatmap](docs/task1_correlation_heatmap.png)

**Why this matters:**
- Detect multicollinearity
- Identify strong predictors
- Guide feature selection

---

### Pairplot
Pairwise feature relationships and distributions used to visually assess:
- Linearity
- Outliers
- Feature separability

![Pairplot](docs/task2_pairplot.png)

---

## 🧠 Regression Model Results

### Clean Data
Full statistical output:

See: docs/task3_regression_results.txt




**Key interpretation areas:**
- Coefficients
- p-values
- R² / Adjusted R²
- Feature significance

---

### Noisy Data
Model re-fit after adding noise:


regression_study/
│── notebooks/
│── docs/
│── README.md