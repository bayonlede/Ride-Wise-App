"""
RideWise Churn Prediction API
FastAPI backend for serving predictions and SHAP explanations
"""

import os
import logging
import pickle
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from sklearn.preprocessing import PowerTransformer, StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global variables for model and preprocessors
model = None
power_transformer = None
standard_scaler = None
shap_explainer = None
feature_names = None
training_data_sample = None


class RiderFeatures(BaseModel):
    """Input schema for rider features"""
    recency: float = Field(..., ge=0, description="Days since last ride")
    frequency: int = Field(..., ge=0, description="Total number of trips")
    monetary: float = Field(..., ge=0, description="Total spending amount")
    surge_exposure: float = Field(..., ge=0, le=1, description="Percentage of rides during surge pricing (0-1)")
    loyalty_status: int = Field(..., ge=0, le=3, description="Loyalty tier: 0=Bronze, 1=Silver, 2=Gold, 3=Platinum")
    churn_prob: float = Field(..., ge=0, le=1, description="Historical churn probability score")
    rider_active_days: int = Field(..., ge=0, description="Days since signup")
    rating_by_rider: float = Field(..., ge=1, le=5, description="Average rating given by rider")
    monthly_trips: Optional[float] = Field(None, description="Average trips per month (auto-calculated if not provided)")

    @field_validator('loyalty_status')
    @classmethod
    def validate_loyalty(cls, v):
        if v not in [0, 1, 2, 3]:
            raise ValueError('Loyalty status must be 0 (Bronze), 1 (Silver), 2 (Gold), or 3 (Platinum)')
        return v


class PredictionResponse(BaseModel):
    """Response schema for predictions"""
    churn_probability: float
    churn_classification: str
    risk_level: str
    confidence: float


class SHAPResponse(BaseModel):
    """Response schema for SHAP explanations"""
    feature_names: List[str]
    shap_values: List[float]
    base_value: float
    prediction: float
    feature_values: List[float]
    top_positive_factors: List[Dict[str, Any]]
    top_negative_factors: List[Dict[str, Any]]


class GlobalImportanceResponse(BaseModel):
    """Response schema for global feature importance"""
    feature_names: List[str]
    importance_values: List[float]
    importance_type: str


class HealthResponse(BaseModel):
    """Response schema for health check"""
    status: str
    model_loaded: bool
    version: str


def get_model_path() -> str:
    """Get the path to the model file"""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(base_dir, "model", "random_forest_model.pkl")


def get_data_path() -> str:
    """Get the path to the training data for fitting transformers"""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(base_dir, "data", "processed_data", "rfms_table.csv")


def load_and_prepare_training_data() -> pd.DataFrame:
    """Load and prepare sample data for fitting transformers"""
    try:
        data_path = get_data_path()
        df = pd.read_csv(data_path)
        
        # Calculate monthly_trips from frequency and recency
        df['monthly_trips'] = (df['frequency'] / np.maximum(df['recency'], 1)) * 30
        
        # Estimate rider_active_days (using recency as proxy, typically active_days > recency)
        df['rider_active_days'] = df['recency'] * 10  # Approximation
        
        # Create churn_prob proxy based on segmentation
        churn_map = {
            'Critical Risk Rider (likely churn)': 0.7,
            'At Risk Users': 0.5,
            'Average Risk Rider': 0.3,
            'Loyal Rider': 0.15,
            'High‑Value Loyalists': 0.1,
            'Regular Commuters': 0.2,
            'Occasional Riders': 0.35
        }
        df['churn_prob'] = df['weighted_segmentation'].map(churn_map).fillna(0.3)
        
        # Create rating_by_rider (simulated between 3.5-5)
        np.random.seed(42)
        df['rating_by_rider'] = np.random.uniform(3.5, 5, len(df))
        
        # Encode loyalty status
        loyalty_map = {'Bronze': 0, 'Silver': 1, 'Gold': 2, 'Platinum': 3}
        # Map riders_segmentation to loyalty
        def get_loyalty(seg):
            if 'Loyal' in seg or 'High' in seg:
                return np.random.choice([2, 3])
            elif 'Regular' in seg:
                return np.random.choice([1, 2])
            else:
                return np.random.choice([0, 1])
        
        df['loyalty_status'] = df['riders_segmentation'].apply(get_loyalty)
        
        # Select and rename columns
        df = df.rename(columns={'surge_exposure(%)': 'surge_exposure'})
        
        return df[['recency', 'frequency', 'monetary', 'surge_exposure', 
                   'loyalty_status', 'churn_prob', 'rider_active_days', 
                   'rating_by_rider', 'monthly_trips']]
    except Exception as e:
        logger.error(f"Error loading training data: {e}")
        raise


def fit_transformers(data: pd.DataFrame):
    """Fit the preprocessing transformers on training data"""
    global power_transformer, standard_scaler
    
    columns_to_scale = ['recency', 'frequency', 'monetary', 'surge_exposure',
                        'churn_prob', 'rider_active_days', 'rating_by_rider', 'monthly_trips']
    
    # Handle infinities and NaN
    data[columns_to_scale] = data[columns_to_scale].replace([np.inf, -np.inf], np.nan)
    data[columns_to_scale] = data[columns_to_scale].fillna(0)
    
    # Fit PowerTransformer
    power_transformer = PowerTransformer(method='yeo-johnson')
    power_transformer.fit(data[columns_to_scale])
    
    # Transform and fit StandardScaler
    transformed = power_transformer.transform(data[columns_to_scale])
    standard_scaler = StandardScaler()
    standard_scaler.fit(transformed)
    
    logger.info("Transformers fitted successfully")


def preprocess_features(features: RiderFeatures) -> np.ndarray:
    """Preprocess input features for prediction"""
    # Calculate monthly_trips if not provided
    monthly_trips = features.monthly_trips
    if monthly_trips is None:
        rider_active_days = max(features.rider_active_days, 1)
        monthly_trips = (features.frequency / rider_active_days) * 30
    
    # Create feature array in correct order
    feature_values = np.array([[
        features.recency,
        features.frequency,
        features.monetary,
        features.surge_exposure,
        features.churn_prob,
        features.rider_active_days,
        features.rating_by_rider,
        monthly_trips
    ]])
    
    # Handle infinities
    feature_values = np.nan_to_num(feature_values, nan=0, posinf=0, neginf=0)
    
    # Apply transformations
    transformed = power_transformer.transform(feature_values)
    scaled = standard_scaler.transform(transformed)
    
    # Insert loyalty_status (not scaled) at correct position
    result = np.insert(scaled, 4, features.loyalty_status, axis=1)
    
    return result


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and initialize resources on startup"""
    global model, shap_explainer, feature_names, training_data_sample
    
    try:
        # Load the trained model
        model_path = get_model_path()
        logger.info(f"Loading model from {model_path}")
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        logger.info("Model loaded successfully")
        
        # Load training data and fit transformers
        training_data_sample = load_and_prepare_training_data()
        fit_transformers(training_data_sample.copy())
        
        # Define feature names
        feature_names = [
            'recency', 'frequency', 'monetary', 'surge_exposure',
            'loyalty_status', 'churn_prob', 'rider_active_days',
            'rating_by_rider', 'monthly_trips'
        ]
        
        # Initialize SHAP explainer
        logger.info("Initializing SHAP explainer...")
        shap_explainer = shap.TreeExplainer(model)
        logger.info("SHAP explainer initialized")
        
        yield
        
    except FileNotFoundError:
        logger.error(f"Model file not found at {model_path}")
        raise
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise
    finally:
        logger.info("Shutting down API")


# Create FastAPI app
app = FastAPI(
    title="RideWise Churn Prediction API",
    description="API for predicting rider churn and providing SHAP-based explanations",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check API health status"""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        version="1.0.0"
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_churn(features: RiderFeatures):
    """
    Predict churn probability for a rider.
    
    Returns:
    - churn_probability: Probability of churn (0-1)
    - churn_classification: "Churn" or "Not Churn"
    - risk_level: "Low", "Medium", "High", or "Critical"
    - confidence: Model confidence in the prediction
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        # Preprocess features
        processed = preprocess_features(features)
        
        # Get prediction
        proba = model.predict_proba(processed)[0]
        churn_prob = float(proba[1])
        prediction = int(proba[1] > 0.5)
        
        # Determine risk level
        if churn_prob < 0.25:
            risk_level = "Low"
        elif churn_prob < 0.50:
            risk_level = "Medium"
        elif churn_prob < 0.75:
            risk_level = "High"
        else:
            risk_level = "Critical"
        
        # Calculate confidence
        confidence = max(proba[0], proba[1])
        
        logger.info(f"Prediction: churn_prob={churn_prob:.4f}, classification={'Churn' if prediction else 'Not Churn'}")
        
        return PredictionResponse(
            churn_probability=round(churn_prob, 4),
            churn_classification="Churn" if prediction else "Not Churn",
            risk_level=risk_level,
            confidence=round(confidence, 4)
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/explain", response_model=SHAPResponse, tags=["Explainability"])
async def explain_prediction(features: RiderFeatures):
    """
    Get SHAP explanation for a rider's churn prediction.
    
    Returns:
    - feature_names: List of feature names
    - shap_values: SHAP values for each feature
    - base_value: Expected value (base prediction)
    - prediction: Final prediction value
    - top_positive_factors: Features pushing toward churn
    - top_negative_factors: Features pushing away from churn
    """
    if model is None or shap_explainer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model or SHAP explainer not loaded"
        )
    
    try:
        # Preprocess features
        processed = preprocess_features(features)
        
        # Get SHAP values
        shap_values = shap_explainer.shap_values(processed)
        
        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            # Binary classification - use positive class
            sv = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
            base = shap_explainer.expected_value[1] if isinstance(shap_explainer.expected_value, (list, np.ndarray)) else shap_explainer.expected_value
        else:
            sv = shap_values[0]
            base = shap_explainer.expected_value
        
        # Get prediction
        proba = model.predict_proba(processed)[0][1]
        
        # Create feature importance ranking
        feature_importance = list(zip(feature_names, sv.tolist(), processed[0].tolist()))
        
        # Sort by absolute SHAP value
        sorted_features = sorted(feature_importance, key=lambda x: abs(x[1]), reverse=True)
        
        # Separate positive and negative factors
        positive_factors = [
            {"feature": f, "shap_value": round(s, 4), "feature_value": round(v, 4)}
            for f, s, v in sorted_features if s > 0
        ][:5]
        
        negative_factors = [
            {"feature": f, "shap_value": round(s, 4), "feature_value": round(v, 4)}
            for f, s, v in sorted_features if s < 0
        ][:5]
        
        return SHAPResponse(
            feature_names=feature_names,
            shap_values=[round(v, 4) for v in sv.tolist()],
            base_value=round(float(base), 4),
            prediction=round(float(proba), 4),
            feature_values=[round(v, 4) for v in processed[0].tolist()],
            top_positive_factors=positive_factors,
            top_negative_factors=negative_factors
        )
        
    except Exception as e:
        logger.error(f"SHAP explanation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Explanation failed: {str(e)}"
        )


@app.get("/feature-importance", response_model=GlobalImportanceResponse, tags=["Explainability"])
async def get_global_feature_importance():
    """
    Get global feature importance from the trained model.
    
    Returns the Random Forest feature importances showing
    which features have the most impact on predictions overall.
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        importances = model.feature_importances_.tolist()
        
        return GlobalImportanceResponse(
            feature_names=feature_names,
            importance_values=[round(v, 4) for v in importances],
            importance_type="Random Forest Feature Importance (Gini)"
        )
        
    except Exception as e:
        logger.error(f"Feature importance error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get feature importance: {str(e)}"
        )


@app.get("/sample-riders", tags=["Data"])
async def get_sample_riders():
    """
    Get sample rider data for demonstration purposes.
    Returns 5 sample riders with different risk profiles.
    """
    samples = [
        {
            "name": "High Risk - Inactive User",
            "description": "Long-time user showing signs of disengagement",
            "features": {
                "recency": 45,
                "frequency": 8,
                "monetary": 120.50,
                "surge_exposure": 0.15,
                "loyalty_status": 1,
                "churn_prob": 0.65,
                "rider_active_days": 300,
                "rating_by_rider": 3.8
            }
        },
        {
            "name": "Low Risk - Loyal Commuter",
            "description": "Regular daily commuter with high engagement",
            "features": {
                "recency": 2,
                "frequency": 45,
                "monetary": 580.00,
                "surge_exposure": 0.25,
                "loyalty_status": 3,
                "churn_prob": 0.12,
                "rider_active_days": 450,
                "rating_by_rider": 4.8
            }
        },
        {
            "name": "Medium Risk - Weekend Rider",
            "description": "Occasional weekend user with moderate activity",
            "features": {
                "recency": 12,
                "frequency": 18,
                "monetary": 245.00,
                "surge_exposure": 0.35,
                "loyalty_status": 2,
                "churn_prob": 0.38,
                "rider_active_days": 180,
                "rating_by_rider": 4.2
            }
        },
        {
            "name": "Critical Risk - Dissatisfied User",
            "description": "User with poor ratings and declining usage",
            "features": {
                "recency": 60,
                "frequency": 5,
                "monetary": 75.00,
                "surge_exposure": 0.55,
                "loyalty_status": 0,
                "churn_prob": 0.78,
                "rider_active_days": 200,
                "rating_by_rider": 3.2
            }
        },
        {
            "name": "New User - Potential Loyal",
            "description": "Recently joined user with promising activity",
            "features": {
                "recency": 3,
                "frequency": 12,
                "monetary": 165.00,
                "surge_exposure": 0.20,
                "loyalty_status": 0,
                "churn_prob": 0.25,
                "rider_active_days": 30,
                "rating_by_rider": 4.5
            }
        }
    ]
    
    return {"samples": samples}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
