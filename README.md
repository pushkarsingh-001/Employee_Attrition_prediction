# Employee Attrition Prediction

A machine learning project to predict employee attrition using classification models. This project uses exploratory data analysis (EDA) and multiple classification algorithms to identify employees at risk of leaving the company.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Features](#features)
- [Models Used](#models-used)
- [Contributing](#contributing)

## Overview

Employee attrition is a critical challenge for organizations. This project aims to:
- Analyze employee data to identify patterns and factors contributing to attrition
- Build predictive models to identify employees likely to leave
- Provide insights into retention strategies

## Dataset

The dataset contains information about 1,470 employees with 35 features including:

### Key Features:
- **Personal Information**: Age, Gender, Marital Status
- **Work Details**: Department, Job Role, Job Level, Distance from Home
- **Employment History**: Years at Company, Years in Current Role, Total Working Years
- **Satisfaction Metrics**: Environment Satisfaction, Job Satisfaction, Job Involvement, Work-Life Balance, Relationship Satisfaction
- **Compensation**: Monthly Income, Daily Rate, Hourly Rate, Monthly Rate
- **Career Development**: Years Since Last Promotion, Stock Option Level, Training Times Last Year
- **Target Variable**: Attrition (Yes/No)

### Dataset Statistics:
- **Total Records**: 1,470 employees
- **Total Features**: 35
- **Target Distribution**: Binary classification (Attrition: Yes/No)

## Project Structure

```
Employee_Attrition_prediction/
├── README.md                                      # Project documentation
├── Employee_Attrition_Prediction.py              # Main Python script
├── Employee_Attrition_(EDA_&_Prediction).ipynb   # Jupyter notebook with full analysis
└── Employee_Attrition.csv                        # Dataset
```

## Installation

### Requirements:
- Python 3.7+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

### Setup:

```bash
# Clone the repository
git clone https://github.com/pushkarsingh-001/Employee_Attrition_prediction.git
cd Employee_Attrition_prediction

# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn

# Or install from requirements.txt (if available)
pip install -r requirements.txt
```

## Usage

### Running the Python Script:

```bash
python Employee_Attrition_Prediction.py
```

### Running the Jupyter Notebook:

```bash
jupyter notebook Employee_Attrition_\(EDA_\&_Prediction\).ipynb
```

## Methodology

### 1. Data Loading & Exploration
- Load employee attrition dataset
- Examine data shape, types, and basic statistics
- Check for missing values and data quality

### 2. Exploratory Data Analysis (EDA)
- **Attrition Distribution**: Visualize target variable balance
- **Correlation Analysis**: Identify relationships between features and attrition
- **Categorical Variables**: Identify and analyze categorical features
- **Visualization**: Create heatmaps and countplots to understand patterns

### 3. Data Preprocessing
- **Encoding**: Convert categorical variables to numerical using LabelEncoder
- **Feature Scaling**: Standardize numerical features using StandardScaler
- **Train-Test Split**: 80-20 split with stratification to maintain class balance

### 4. Model Training
Multiple classification algorithms are trained and evaluated:

**Primary Model**: Random Forest Classifier
- n_estimators: 100
- Handles non-linear relationships
- Provides feature importance scores

**Alternative Models** (in notebook):
- Decision Tree Classifier
- Support Vector Machine (SVM)

### 5. Model Evaluation
- **Accuracy Score**: Overall prediction correctness
- **Confusion Matrix**: True/False positives and negatives
- **Classification Report**: Precision, Recall, F1-Score per class
- **Feature Importance**: Identify most influential factors

## Results

The model provides:

### Performance Metrics:
- **Accuracy**: Model's overall prediction correctness
- **Precision & Recall**: Class-specific performance metrics
- **Feature Importance**: Ranking of factors affecting attrition

### Key Insights:
The analysis identifies which employee characteristics most strongly correlate with attrition, helping HR teams focus retention efforts on high-risk groups.

## Features

### Features Used in Model:
The project analyzes 34 predictor variables (excluding the target variable) including:
- Demographic factors (Age, Gender, Marital Status)
- Job-related factors (Department, Job Role, Job Level)
- Compensation factors (Monthly Income, Hourly Rate)
- Satisfaction and engagement metrics
- Career development indicators
- Work-life balance factors

### Feature Engineering:
- All categorical variables are label-encoded
- All numerical features are standardized for model training
- Stratified train-test split maintains class distribution

## Models Used

### 1. Random Forest Classifier (Primary)
```python
RandomForestClassifier(n_estimators=100, random_state=42)
```
- Ensemble method combining multiple decision trees
- Robust to outliers and non-linear relationships
- Provides feature importance scores
- Ideal for HR prediction tasks

### 2. Decision Tree Classifier (Alternative)
- Interpretable model
- Good for understanding decision logic

### 3. Support Vector Machine (Alternative)
- Effective for binary classification
- Works well with high-dimensional data

## Code Overview

### Main Script Components:

```python
# 1. Import Libraries
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# 2. Load Data
df = pd.read_csv("Employee_Attrition.csv")

# 3. EDA & Visualization
# - Data shape and head
# - Null values check
# - Correlation heatmap
# - Attrition distribution

# 4. Preprocessing
# - Encode categorical variables
# - Scale numerical features
# - Split data

# 5. Model Training & Evaluation
# - Train Random Forest model
# - Generate predictions
# - Evaluate performance
# - Display feature importance
```

## Contributing

Contributions are welcome! Please feel free to:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

## Future Enhancements

- Hyperparameter tuning and cross-validation
- Advanced feature engineering
- Ensemble methods combining multiple models
- Real-time prediction API
- Interactive dashboard for visualization
- Class imbalance handling techniques (SMOTE, class weights)
- Explainability analysis (SHAP, LIME)

## License

This project is open source and available under the MIT License.

## Contact

**Author**: Pushkar Singh  
**GitHub**: [@pushkarsingh-001](https://github.com/pushkarsingh-001)

---

## Disclaimer

This model is created for educational and analytical purposes. Predictions should be used in conjunction with human judgment and other HR processes for final decision-making.
