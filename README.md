# Spatio-Temporal Analysis and Classification of Fatal Road Crash Characteristics Using FARS Data

A machine learning project that analyzes fatal road crashes in the United States using the **2023 Fatality Analysis Reporting System (FARS)** dataset. The project combines exploratory data analysis, feature engineering, and multiple classification algorithms to identify factors associated with crashes involving multiple fatalities.

---

## Project Overview

Road traffic crashes remain a major public safety concern. This project leverages the **2023 FARS dataset** published by the **National Highway Traffic Safety Administration (NHTSA)** to identify the geographical, temporal, roadway, and environmental characteristics associated with fatal crashes.

The primary objective is to classify crashes into **single-fatality** and **multiple-fatality** events and compare the performance of several machine learning algorithms under an imbalanced classification setting.

---

## Objectives

- Analyze spatial and temporal patterns in fatal crashes.
- Identify key factors influencing crash severity.
- Handle severe class imbalance using weighted learning.
- Compare multiple machine learning classification models.
- Evaluate models using metrics suitable for imbalanced datasets.

---

## Dataset

**Source:** Fatality Analysis Reporting System (FARS) 2023

**Provider:** National Highway Traffic Safety Administration (NHTSA)

### Dataset Statistics

- **Total Records:** ~38,000 fatal crashes
- **Predictor Variables:** 80+ crash-related attributes
- **Target Variable:** Fatality_Class

Target classes:

| Class | Description |
|--------|-------------|
| 0 | Single Fatality (FATALS = 1) |
| 1 | Multiple Fatalities (FATALS ≥ 2) |

Class distribution:

- **Single Fatality:** ~93%
- **Multiple Fatalities:** ~7%

This significant imbalance required the use of class-weighted machine learning techniques.

---

## Features Used

The project utilizes variables from multiple domains.

### Geographic

- State
- County
- City
- Latitude
- Longitude
- Rural / Urban classification

### Temporal

- Month
- Day of Week
- Hour
- Minute

### Roadway

- Route Type
- Functional System
- Road Owner
- National Highway System
- Relation to Junction
- Relation to Roadway
- Intersection Type
- Work Zone

### Environmental

- Lighting Condition
- Weather Condition
- School Bus Involvement

---

## Technologies Used

- R
- RStudio
- caret
- randomForest
- nnet
- xgboost
- lightgbm
- pROC
- ggplot2
- dplyr
- tidyr

---

## Workflow

1. Data Cleaning
2. Feature Selection
3. Feature Engineering
4. Train-Test Split
5. Class Weight Calculation
6. Model Training
7. Model Evaluation
8. Performance Comparison

---

## Machine Learning Models

The following classification models were implemented and compared:

- Logistic Regression
- Decision Tree
- Random Forest
- Neural Network
- XGBoost
- LightGBM

Each model was trained using the same preprocessing pipeline to ensure a fair comparison.

---

## Evaluation Metrics

Since the dataset is highly imbalanced, evaluation was performed using multiple metrics instead of relying solely on accuracy.

- Accuracy
- Precision
- Recall (Sensitivity)
- Specificity
- F1-Score
- ROC-AUC
- Confusion Matrix
- Balanced Accuracy

---

## Project Structure

```
Fatality-Analysis/
│
├── data/
│   ├── FARS2023.csv
│
├── scripts/
│   ├── data_cleaning.R
│   ├── exploratory_analysis.R
│   ├── logistic_regression.R
│   ├── decision_tree.R
│   ├── random_forest.R
│   ├── neural_network.R
│   ├── xgboost.R
│   ├── lightgbm.R
│
├── results/
│   ├── confusion_matrices/
│   ├── feature_importance/
│   ├── plots/
│
├── presentation/
│
├── report/
│
└── README.md
```

---

## How to Run

### Clone the repository

```bash
git clone https://github.com/yourusername/Fatality-Analysis.git
```

### Open RStudio

Open the project folder.

### Install required packages

```r
install.packages(c(
  "caret",
  "randomForest",
  "nnet",
  "xgboost",
  "lightgbm",
  "pROC",
  "ggplot2",
  "dplyr",
  "tidyr"
))
```

### Run the scripts

Execute the scripts in the following order:

1. Data Cleaning
2. Exploratory Data Analysis
3. Feature Engineering
4. Train/Test Split
5. Train Individual Models
6. Evaluate Performance

---

## Key Findings

- Fatal crashes exhibit clear spatial and temporal trends.
- Lighting conditions, weather, roadway characteristics, and location significantly influence crash severity.
- Class-weighted learning improves minority class detection.
- Ensemble models generally outperform traditional statistical models for this classification task.

---

## Future Work

- Hyperparameter optimization
- Cross-validation
- SHAP explainability
- Geographic hotspot visualization
- Deep learning models
- Real-time crash severity prediction

---

## Author

**Neeraj Suriya**

Bachelor of Engineering – Computer Science and Engineering

Sathyabama Institute of Science and Technology

---

## References

1. National Highway Traffic Safety Administration (NHTSA)
2. Fatality Analysis Reporting System (FARS) 2023
3. R Documentation
4. caret Package Documentation
5. XGBoost Documentation
6. LightGBM Documentation

---

## 📄 License

This project is intended for academic and research purposes.
