# RideWise Churn Prediction Dashboard

A professionally designed API dashboard for predicting rider churn using machine learning with SHAP explainability.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31-red.svg)

## 🎯 Overview

This dashboard provides:
- **Churn Prediction**: Predict whether a rider will churn using a trained RandomForestClassifier
- **SHAP Explainability**: Understand which features drive each prediction
- **Interactive Dashboard**: Modern, stakeholder-friendly UI for exploring predictions
- **REST API**: Production-ready endpoints for integration

## 📁 Project Structure

```
dashboard/
├── api/
│   ├── __init__.py
│   └── main.py              # FastAPI backend with prediction endpoints
├── app/
│   ├── __init__.py
│   └── streamlit_app.py     # Streamlit dashboard frontend
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd dashboard
pip install -r requirements.txt
```

### 2. Start the API Server

```bash
cd api
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

- API Documentation: `http://localhost:8000/docs`
- Health Check: `http://localhost:8000/health`

### 3. Start the Dashboard

In a new terminal:

```bash
cd app
streamlit run streamlit_app.py --server.port 8501
```

The dashboard will open at `http://localhost:8501`

## 📊 API Endpoints

### Health Check
```
GET /health
```
Returns API status and model loading state.

### Predict Churn
```
POST /predict
```
**Request Body:**
```json
{
  "recency": 15,
  "frequency": 20,
  "monetary": 250.0,
  "surge_exposure": 0.25,
  "loyalty_status": 1,
  "churn_prob": 0.30,
  "rider_active_days": 200,
  "rating_by_rider": 4.2
}
```

**Response:**
```json
{
  "churn_probability": 0.1234,
  "churn_classification": "Not Churn",
  "risk_level": "Low",
  "confidence": 0.8766
}
```

### SHAP Explanation
```
POST /explain
```
Returns SHAP values for understanding individual predictions.

### Global Feature Importance
```
GET /feature-importance
```
Returns model-level feature importance scores.

### Sample Riders
```
GET /sample-riders
```
Returns sample rider profiles for testing.

## 🎨 Dashboard Features

### Prediction Panel
- **Churn Probability Gauge**: Visual representation of churn risk
- **Risk Classification**: Low, Medium, High, or Critical
- **Confidence Score**: Model certainty in the prediction

### SHAP Explainability
- **Waterfall Chart**: Shows how each feature contributes to the prediction
- **Key Factors**: Highlights top factors increasing/decreasing churn risk
- **Global Importance**: Model-wide feature importance rankings

### Retention Recommendations
- Automatic suggestions based on risk level
- Actionable strategies for customer retention

## 📈 Input Features

| Feature | Description | Range |
|---------|-------------|-------|
| `recency` | Days since last ride | 0-365 |
| `frequency` | Total number of trips | 0-500 |
| `monetary` | Total spending (£) | 0-10000 |
| `surge_exposure` | % of rides during surge | 0-1 |
| `loyalty_status` | Loyalty tier (0-3) | Bronze=0, Silver=1, Gold=2, Platinum=3 |
| `churn_prob` | Historical churn score | 0-1 |
| `rider_active_days` | Days since signup | 1-1000 |
| `rating_by_rider` | Average rating given | 1-5 |

## 🎯 Risk Levels

| Level | Probability Range | Action |
|-------|------------------|--------|
| 🟢 Low | 0-25% | Maintain engagement |
| 🟡 Medium | 25-50% | Monitor & engage |
| 🟠 High | 50-75% | Proactive outreach |
| 🔴 Critical | 75-100% | Immediate action |

## 🔧 Configuration

### API Configuration
Edit `api/main.py` to modify:
- Model path
- Preprocessing parameters
- Logging settings

### Dashboard Configuration
Edit `app/streamlit_app.py` to modify:
- API URL (default: `http://localhost:8000`)
- Color scheme
- Layout settings

## 📝 Sample Usage

### Using the API with Python
```python
import requests

# Predict churn
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "recency": 30,
        "frequency": 10,
        "monetary": 150.0,
        "surge_exposure": 0.4,
        "loyalty_status": 1,
        "churn_prob": 0.5,
        "rider_active_days": 100,
        "rating_by_rider": 3.8
    }
)

print(response.json())
```

### Using cURL
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "recency": 30,
    "frequency": 10,
    "monetary": 150.0,
    "surge_exposure": 0.4,
    "loyalty_status": 1,
    "churn_prob": 0.5,
    "rider_active_days": 100,
    "rating_by_rider": 3.8
  }'
```

## 🛠️ Development

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black api/ app/
isort api/ app/
```

## 📄 License

This project is part of the RideWise Customer Analytics platform.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📧 Support

For questions or issues, please contact the RideWise Analytics team.
