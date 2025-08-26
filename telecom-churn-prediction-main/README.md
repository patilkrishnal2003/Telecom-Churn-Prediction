# 📞 Telecom Customer Churn Prediction

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.1+-green.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Advanced Machine Learning Solution for Predicting Customer Churn in Telecommunications**

A comprehensive, production-ready machine learning project that predicts customer churn using advanced analytics, feature engineering, and multiple ML algorithms. This project demonstrates professional data science practices with clean code, extensive documentation, and actionable business insights.

## 🎯 Project Overview

Customer churn prediction is critical for telecom companies to maintain profitability and customer satisfaction. This project implements a complete ML pipeline to identify customers at risk of churning and provides actionable insights for retention strategies.

### Key Features

- 🔍 **Advanced EDA** with statistical analysis and interactive visualizations
- 🛠️ **Feature Engineering** with domain-specific transformations
- 🤖 **Multiple ML Models** including ensemble methods and gradient boosting
- ⚡ **Hyperparameter Optimization** for maximum performance
- 📊 **Comprehensive Evaluation** with detailed metrics and visualizations
- 🎯 **Business Insights** with actionable recommendations
- 📦 **Production-Ready Code** with modular architecture

## 📊 Results Summary

| Metric | Best Model Performance |
|--------|----------------------|
| **AUC Score** | 0.85+ |
| **Precision** | 0.80+ |
| **Recall** | 0.75+ |
| **F1-Score** | 0.77+ |

## 🗂️ Project Structure

```
ml_project/
├── 📊 data/
│   ├── churn-bigml-80.csv          # Training dataset
│   └── churn-bigml-20.csv          # Test dataset
├── 📓 notebooks/
│   ├── Telecom_Customer_Churn_Prediction_Professional.ipynb  # Main analysis
│   └── Analytical Geeks_Telecom Customer Churn Prediction.ipynb.ipynb  # Original
├── 🐍 src/
│   ├── __init__.py
│   ├── data_preprocessing.py        # Data cleaning & feature engineering
│   ├── eda_utils.py                # Exploratory data analysis utilities
│   └── model_training.py           # ML model training & evaluation
├── 🤖 models/
│   └── best_churn_model.pkl        # Saved best performing model
├── 📋 requirements.txt             # Python dependencies
├── 📖 README.md                    # Project documentation
└── 🚀 .gitignore                   # Git ignore file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook
- Git (for version control)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/telecom-churn-prediction.git
   cd telecom-churn-prediction
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

5. **Open the main notebook**
   - Navigate to `Telecom_Customer_Churn_Prediction_Professional.ipynb`
   - Run all cells to reproduce the analysis

## 📈 Dataset Information

### Source
- **Training Data**: `churn-bigml-80.csv` (2,667 customers)
- **Test Data**: `churn-bigml-20.csv` (667 customers)
- **Total Features**: 20 original features + 30+ engineered features

### Key Features
- **Customer Demographics**: State, Account length, Area code
- **Service Plans**: International plan, Voice mail plan
- **Usage Patterns**: Day/Evening/Night/International minutes and calls
- **Charges**: Detailed billing information
- **Service Interactions**: Customer service calls
- **Target Variable**: Churn (True/False)

### Data Quality
- ✅ No missing values
- ✅ No duplicate records
- ⚠️ Class imbalance (~14% churn rate)
- ✅ High data quality overall

## 🔧 Feature Engineering

Our advanced feature engineering creates meaningful predictors:

### Usage Aggregation Features
- `Total_minutes`: Combined usage across all time periods
- `Total_calls`: Total number of calls made
- `Total_charge`: Complete billing amount
- `Avg_call_duration`: Average call length

### Behavioral Indicators
- `High_day_usage`: Above 75th percentile day usage
- `High_service_calls`: 4+ customer service calls
- `Has_both_plans`: International + Voice mail plans
- `No_service_calls`: Zero service interactions

### Ratio Features
- `Day_charge_per_minute`: Cost efficiency metrics
- `Usage_ratios`: Time-of-day usage patterns
- `Intl_usage_ratio`: International usage proportion

### Geographic Features
- State-level dummy variables (50 features)
- Regional churn pattern analysis

## 🤖 Machine Learning Models

### Models Implemented
1. **Logistic Regression** (Baseline)
2. **Decision Tree**
3. **Random Forest**
4. **AdaBoost**
5. **Gradient Boosting**
6. **XGBoost**
7. **LightGBM**
8. **Support Vector Machine**
9. **Neural Network (MLP)**

### Model Selection Process
1. **Baseline Training**: All models with default parameters
2. **Cross-Validation**: 5-fold CV for robust evaluation
3. **Hyperparameter Tuning**: RandomizedSearchCV for top performers
4. **Final Evaluation**: Comprehensive metrics on test set

### Handling Class Imbalance
- **SMOTE**: Synthetic Minority Oversampling Technique
- **SMOTEENN**: Combined over/under-sampling
- **Class Weights**: Balanced class weights in algorithms

## 📊 Evaluation Metrics

### Primary Metrics
- **AUC-ROC**: Area Under the Receiver Operating Characteristic Curve
- **Precision**: True Positives / (True Positives + False Positives)
- **Recall**: True Positives / (True Positives + False Negatives)
- **F1-Score**: Harmonic mean of Precision and Recall

### Business Metrics
- **Churn Rate**: Percentage of customers who churn
- **Retention Rate**: Percentage of customers retained
- **False Positive Rate**: Incorrectly flagged loyal customers
- **Cost-Benefit Analysis**: ROI of retention campaigns

## 💼 Business Insights

### Key Churn Drivers
1. **Customer Service Calls** (Strongest predictor)
   - 4+ calls indicate high churn risk
   - Recommendation: Proactive service intervention

2. **International Plan Usage**
   - Higher churn rates among international plan users
   - Recommendation: Review pricing and features

3. **Usage Patterns**
   - Extreme usage (very high/low) correlates with churn
   - Recommendation: Usage-based retention offers

4. **Account Tenure**
   - New customers at higher risk
   - Recommendation: Enhanced onboarding program

### Actionable Recommendations

#### Immediate Actions (0-3 months)
- 🚨 **Alert System**: Flag customers with 3+ service calls
- 🎯 **Targeted Campaigns**: Focus on high-risk customers (probability > 0.7)
- 📞 **Proactive Outreach**: Contact international plan users with usage issues

#### Strategic Initiatives (3-12 months)
- 🔄 **Service Quality**: Reduce need for customer service calls
- 📊 **Personalization**: Use model insights for customized experiences
- 🗺️ **Geographic Focus**: Special attention to high-churn regions

## 🔍 Model Interpretation

### Feature Importance Analysis
The model identifies the most influential factors in churn prediction:

1. **Customer service calls** (Highest importance)
2. **Total day charge** 
3. **International plan**
4. **Voice mail plan**
5. **Total day minutes**

### SHAP Analysis (Future Enhancement)
- Individual prediction explanations
- Feature contribution analysis
- Model transparency for business stakeholders

## 🚀 Deployment Guide

### Production Deployment
1. **Model Serialization**: Save trained model using joblib
2. **API Development**: Create REST API for real-time scoring
3. **Batch Scoring**: Process customer data in batches
4. **Monitoring**: Track model performance over time

### Integration Options
- **CRM Integration**: Direct integration with customer management systems
- **Real-time Scoring**: Stream processing for immediate insights
- **Scheduled Batch**: Daily/weekly churn risk updates

## 📈 Performance Monitoring

### Model Drift Detection
- Monitor feature distributions over time
- Track prediction accuracy on new data
- Set up alerts for performance degradation

### Business Impact Tracking
- Measure retention rate improvements
- Calculate ROI of retention campaigns
- Monitor customer satisfaction scores

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Commit your changes** (`git commit -m 'Add amazing feature'`)
4. **Push to the branch** (`git push origin feature/amazing-feature`)
5. **Open a Pull Request**

### Development Guidelines
- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Include unit tests for new features
- Update documentation as needed

## 📚 Documentation

### Code Documentation
- **Docstrings**: Comprehensive function documentation
- **Type Hints**: Clear parameter and return types
- **Comments**: Explain complex logic and business rules

### Notebooks
- **Professional Formatting**: Clean, well-structured notebooks
- **Markdown Explanations**: Detailed analysis explanations
- **Visualizations**: Professional charts and graphs

## 🐛 Troubleshooting

### Common Issues

**Issue**: ImportError for custom modules
```bash
# Solution: Add src directory to Python path
import sys
sys.path.append('src')
```

**Issue**: Memory errors with large datasets
```bash
# Solution: Use chunked processing or increase memory
# Consider using Dask for larger datasets
```

**Issue**: Model training takes too long
```bash
# Solution: Use RandomizedSearchCV instead of GridSearchCV
# Reduce parameter grid size or use fewer CV folds
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset**: BigML Telecom Customer Churn Dataset
- **Libraries**: scikit-learn, pandas, numpy, matplotlib, seaborn, plotly
- **Inspiration**: Industry best practices in churn prediction
- **Community**: Open source data science community

## 📞 Contact

**Project Maintainer**: Patil Krishnal
- 📧 Email: patilkrishnal2003@gmail.com.com
- 💼 LinkedIn: https://www.linkedin.com/in/krishnal-patil-b1a5a6208/
- 🐙 GitHub: https://github.com/patilkrishnal2003

---
