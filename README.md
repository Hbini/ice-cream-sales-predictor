# 🍦 Ice Cream Sales Predictor

**Machine Learning Model for Predicting Ice Cream Sales Based on Daily Temperature**

## 📋 Project Overview

This is a comprehensive Machine Learning project developed for the **DIO Challenge**: "Training Your First Machine Learning Model to Predict Ice Cream Sales". The project implements a predictive regression model to forecast daily ice cream sales based on temperature data, helping ice cream shop owners optimize production and inventory management.

### Business Scenario
Imagine you own an ice cream shop called "Gelato Mágico" located in a coastal city. You notice that the number of ice creams sold daily has a strong correlation with ambient temperature. Without proper planning, you end up producing more ice cream than necessary, leading to waste and reduced profits.

Using Machine Learning, we solve this problem by creating a model that predicts daily sales based on temperature, allowing for efficient production planning.

## 🎯 Objectives

✅ **Train a Machine Learning Model** - Develop a regression model to predict ice cream sales based on daily temperature

✅ **Model Management with MLflow** - Register, track, and manage the model using MLflow

✅ **Real-time Predictions** - Implement the model for real-time sales predictions in a cloud environment

✅ **Structured Pipeline** - Create a reproducible ML pipeline with data preprocessing, training, and evaluation

## 🛠️ Technologies & Tools

- **Python 3.8+**
- **Scikit-learn** - Machine Learning algorithms
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computations
- **Matplotlib** - Data visualization
- **MLflow** - Model tracking and management
- **Azure Machine Learning** - Cloud ML platform
- **Git/GitHub** - Version control

## 📁 Project Structure

```
ice-cream-sales-predictor/
├── inputs/
│   └── sales_data.txt          # Sample input data
├── data/
│   └── ice_cream_sales.csv     # Dataset with temperature and sales
├── src/
│   ├── __init__.py
│   ├── data_preparation.py     # Data loading and preprocessing
│   ├── model_training.py       # Model training and evaluation
│   ├── predictions.py          # Prediction functions
│   └── mlflow_utils.py         # MLflow integration
├── notebooks/
│   └── analysis.ipynb          # Exploratory data analysis
├── models/
│   └── ice_cream_model.pkl     # Trained model
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── .gitignore                   # Git ignore file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip or conda package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Hbini/ice-cream-sales-predictor.git
   cd ice-cream-sales-predictor
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Data Description

The dataset contains daily records with the following features:

| Column | Type | Description |
|--------|------|-------------|
| Temperature (°C) | Float | Daily average temperature |
| Ice Cream Sales | Integer | Number of ice creams sold that day |
| Date | Datetime | Date of the record |

### Dataset Insights

- **Strong Positive Correlation**: Temperature has a strong positive correlation with ice cream sales (r ≈ 0.95)
- **Linear Relationship**: The relationship can be modeled with linear regression
- **Seasonal Pattern**: Higher sales in summer months, lower in winter

## 🤖 Machine Learning Model

### Algorithm: Linear Regression

**Rationale**:
- Simple yet effective for this dataset
- Interpretable results
- Fast training and prediction times
- Well-suited for linear relationships

### Model Performance

- **R² Score**: Explains percentage of variance in sales
- **Mean Absolute Error (MAE)**: Average prediction error in units
- **Root Mean Squared Error (RMSE)**: Penalizes larger errors more heavily

## 📈 Key Insights

1. **Temperature Impact**: Each 1°C increase correlates with an average increase in sales
2. **Prediction Reliability**: The model shows high accuracy for temperature ranges in the training data
3. **Inventory Planning**: Can be used to optimize daily production based on weather forecasts

## 🔧 Model Management with MLflow

### Features Implemented

- **Experiment Tracking**: Log model parameters and metrics
- **Model Versioning**: Track different model versions
- **Model Registry**: Store and manage production models
- **Artifact Storage**: Save models, plots, and evaluation results

### MLflow Commands

```bash
# Start MLflow server
mlflow ui

# Access at http://localhost:5000
```

## 📝 Usage Examples

### Training the Model

```python
from src.model_training import train_model
from src.data_preparation import load_data

# Load data
X, y = load_data('data/ice_cream_sales.csv')

# Train model
model = train_model(X, y)
print(f"Model R² Score: {model.score(X, y):.4f}")
```

### Making Predictions

```python
from src.predictions import predict_sales

# Predict for specific temperatures
temperatures = [25, 28, 32, 35]
predictions = predict_sales(model, temperatures)

for temp, sales in zip(temperatures, predictions):
    print(f"Temperature: {temp}°C -> Predicted Sales: {sales:.0f} units")
```

## 🎓 Learning Outcomes

This project covers:

- ✅ Data loading and preprocessing with Pandas
- ✅ Exploratory Data Analysis (EDA)
- ✅ Feature engineering and scaling
- ✅ Model training and hyperparameter tuning
- ✅ Model evaluation and validation
- ✅ MLflow integration for experiment tracking
- ✅ Model deployment considerations
- ✅ Reproducible ML pipeline development
- ✅ Git version control best practices

## 🔗 Deployment

### Azure Machine Learning

The model can be deployed to Azure ML for:
- Real-time API endpoints
- Batch predictions
- Scheduled inference jobs
- Performance monitoring

### Steps for Deployment

1. Register model in MLflow
2. Create inference script
3. Deploy to Azure ML endpoint
4. Create REST API for predictions

## 📚 Resources & References

- [DIO Platform](https://www.dio.me/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [MLflow Documentation](https://mlflow.org/docs/latest/)
- [Azure Machine Learning Docs](https://docs.microsoft.com/en-us/azure/machine-learning/)
- [Pandas User Guide](https://pandas.pydata.org/docs/)

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

**Hernane Bini** (Hbini)

- GitHub: [@Hbini](https://github.com/Hbini)
- Email: Contact via GitHub

## 🙏 Acknowledgments

- **DIO** - Digital Innovation One for the challenge
- **Microsoft** - Azure ML platform
- **MLflow Community** - For the excellent model tracking tool
- **Scikit-learn Community** - For the robust ML libraries

## 📞 Support

If you have any questions or issues, please:

1. Check existing GitHub Issues
2. Create a new Issue with detailed description
3. Include error messages and system information

---

**Last Updated**: February 2026

**Status**: ✅ In Development

**Version**: 1.0.0
