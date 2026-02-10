"""
RideWise Churn Prediction Dashboard
Streamlit frontend for visualizing predictions and SHAP explanations
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuration
API_URL = "http://localhost:8000"

# Page configuration
st.set_page_config(
    page_title="RideWise | Churn Prediction Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #10B981;
        --danger-color: #EF4444;
        --warning-color: #F59E0B;
        --info-color: #3B82F6;
        --dark-bg: #0F172A;
        --card-bg: #1E293B;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom header styling */
    .main-header {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        padding: 2rem;
        border-radius: 1rem;
        margin-bottom: 2rem;
        color: white;
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 1.1rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1E293B 0%, #334155 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        border: 1px solid #475569;
        text-align: center;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        color: #94A3B8;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Risk badges */
    .risk-low { color: #10B981; }
    .risk-medium { color: #F59E0B; }
    .risk-high { color: #F97316; }
    .risk-critical { color: #EF4444; }
    
    /* Card styling */
    .stMetric {
        background: #1E293B;
        padding: 1rem;
        border-radius: 0.75rem;
        border: 1px solid #334155;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: #1E293B;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 0.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(16, 185, 129, 0.3);
    }
    
    /* Info boxes */
    .info-box {
        background: #1E293B;
        border-left: 4px solid #10B981;
        padding: 1rem;
        border-radius: 0 0.5rem 0.5rem 0;
        margin: 1rem 0;
    }
    
    /* Factor cards */
    .factor-positive {
        background: rgba(239, 68, 68, 0.1);
        border-left: 4px solid #EF4444;
        padding: 0.75rem 1rem;
        border-radius: 0 0.5rem 0.5rem 0;
        margin: 0.5rem 0;
    }
    
    .factor-negative {
        background: rgba(16, 185, 129, 0.1);
        border-left: 4px solid #10B981;
        padding: 0.75rem 1rem;
        border-radius: 0 0.5rem 0.5rem 0;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


def check_api_health():
    """Check if the API is running"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def get_prediction(features: dict):
    """Get churn prediction from API"""
    try:
        response = requests.post(f"{API_URL}/predict", json=features, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return None


def get_explanation(features: dict):
    """Get SHAP explanation from API"""
    try:
        response = requests.post(f"{API_URL}/explain", json=features, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return None


def get_feature_importance():
    """Get global feature importance from API"""
    try:
        response = requests.get(f"{API_URL}/feature-importance", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return None


def get_sample_riders():
    """Get sample rider data from API"""
    try:
        response = requests.get(f"{API_URL}/sample-riders", timeout=10)
        response.raise_for_status()
        return response.json()["samples"]
    except requests.exceptions.RequestException as e:
        return None


def create_gauge_chart(probability: float, title: str = "Churn Probability"):
    """Create a gauge chart for churn probability"""
    # Determine color based on probability
    if probability < 0.25:
        color = "#10B981"  # Green
    elif probability < 0.50:
        color = "#F59E0B"  # Yellow
    elif probability < 0.75:
        color = "#F97316"  # Orange
    else:
        color = "#EF4444"  # Red
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 20, 'color': '#E2E8F0'}},
        number={'suffix': "%", 'font': {'size': 40, 'color': '#E2E8F0'}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': '#475569', 'tickwidth': 2},
            'bar': {'color': color, 'thickness': 0.75},
            'bgcolor': '#1E293B',
            'borderwidth': 2,
            'bordercolor': '#475569',
            'steps': [
                {'range': [0, 25], 'color': 'rgba(16, 185, 129, 0.2)'},
                {'range': [25, 50], 'color': 'rgba(245, 158, 11, 0.2)'},
                {'range': [50, 75], 'color': 'rgba(249, 115, 22, 0.2)'},
                {'range': [75, 100], 'color': 'rgba(239, 68, 68, 0.2)'}
            ],
            'threshold': {
                'line': {'color': '#E2E8F0', 'width': 4},
                'thickness': 0.75,
                'value': probability * 100
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=300,
        margin=dict(l=30, r=30, t=50, b=30)
    )
    
    return fig


def create_shap_waterfall(shap_data: dict):
    """Create a waterfall chart for SHAP values"""
    feature_names = shap_data['feature_names']
    shap_values = shap_data['shap_values']
    base_value = shap_data['base_value']
    
    # Sort by absolute value
    sorted_indices = np.argsort(np.abs(shap_values))[::-1]
    sorted_names = [feature_names[i] for i in sorted_indices]
    sorted_values = [shap_values[i] for i in sorted_indices]
    
    # Create colors based on positive/negative
    colors = ['#EF4444' if v > 0 else '#10B981' for v in sorted_values]
    
    fig = go.Figure(go.Bar(
        x=sorted_values,
        y=sorted_names,
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(color='rgba(0,0,0,0)', width=0)
        ),
        text=[f'{v:+.3f}' for v in sorted_values],
        textposition='outside',
        textfont=dict(color='#E2E8F0')
    ))
    
    fig.update_layout(
        title=dict(
            text='SHAP Feature Contributions',
            font=dict(size=18, color='#E2E8F0'),
            x=0.5
        ),
        xaxis=dict(
            title='SHAP Value (Impact on Churn Prediction)',
            titlefont=dict(color='#94A3B8'),
            tickfont=dict(color='#94A3B8'),
            gridcolor='#334155',
            zerolinecolor='#475569'
        ),
        yaxis=dict(
            tickfont=dict(color='#E2E8F0'),
            categoryorder='total ascending'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=400,
        margin=dict(l=150, r=100, t=50, b=50),
        showlegend=False
    )
    
    return fig


def create_feature_importance_chart(importance_data: dict):
    """Create a bar chart for global feature importance"""
    df = pd.DataFrame({
        'Feature': importance_data['feature_names'],
        'Importance': importance_data['importance_values']
    }).sort_values('Importance', ascending=True)
    
    fig = go.Figure(go.Bar(
        x=df['Importance'],
        y=df['Feature'],
        orientation='h',
        marker=dict(
            color=df['Importance'],
            colorscale=[[0, '#10B981'], [0.5, '#3B82F6'], [1, '#8B5CF6']],
            line=dict(color='rgba(0,0,0,0)', width=0)
        ),
        text=[f'{v:.3f}' for v in df['Importance']],
        textposition='outside',
        textfont=dict(color='#E2E8F0')
    ))
    
    fig.update_layout(
        title=dict(
            text='Global Feature Importance',
            font=dict(size=18, color='#E2E8F0'),
            x=0.5
        ),
        xaxis=dict(
            title='Importance Score',
            titlefont=dict(color='#94A3B8'),
            tickfont=dict(color='#94A3B8'),
            gridcolor='#334155'
        ),
        yaxis=dict(
            tickfont=dict(color='#E2E8F0')
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=400,
        margin=dict(l=150, r=100, t=50, b=50)
    )
    
    return fig


def render_prediction_result(prediction: dict):
    """Render the prediction result with styling"""
    risk_colors = {
        'Low': ('#10B981', '🟢'),
        'Medium': ('#F59E0B', '🟡'),
        'High': ('#F97316', '🟠'),
        'Critical': ('#EF4444', '🔴')
    }
    
    color, emoji = risk_colors.get(prediction['risk_level'], ('#64748B', '⚪'))
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1E293B 0%, #334155 100%); 
                    padding: 1.5rem; border-radius: 1rem; text-align: center;
                    border: 1px solid #475569;">
            <div style="color: #94A3B8; font-size: 0.85rem; text-transform: uppercase; 
                        letter-spacing: 0.1em; margin-bottom: 0.5rem;">Classification</div>
            <div style="font-size: 1.8rem; font-weight: 700; color: {color};">
                {prediction['churn_classification']}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1E293B 0%, #334155 100%); 
                    padding: 1.5rem; border-radius: 1rem; text-align: center;
                    border: 1px solid #475569;">
            <div style="color: #94A3B8; font-size: 0.85rem; text-transform: uppercase; 
                        letter-spacing: 0.1em; margin-bottom: 0.5rem;">Risk Level</div>
            <div style="font-size: 1.8rem; font-weight: 700; color: {color};">
                {emoji} {prediction['risk_level']}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1E293B 0%, #334155 100%); 
                    padding: 1.5rem; border-radius: 1rem; text-align: center;
                    border: 1px solid #475569;">
            <div style="color: #94A3B8; font-size: 0.85rem; text-transform: uppercase; 
                        letter-spacing: 0.1em; margin-bottom: 0.5rem;">Confidence</div>
            <div style="font-size: 1.8rem; font-weight: 700; color: #3B82F6;">
                {prediction['confidence']*100:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_key_factors(shap_data: dict):
    """Render the key factors driving the prediction"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔴 Factors Increasing Churn Risk")
        if shap_data['top_positive_factors']:
            for factor in shap_data['top_positive_factors']:
                st.markdown(f"""
                <div class="factor-positive">
                    <strong>{factor['feature'].replace('_', ' ').title()}</strong><br/>
                    <span style="color: #94A3B8;">Impact: </span>
                    <span style="color: #EF4444;">+{factor['shap_value']:.4f}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No factors significantly increasing churn risk")
    
    with col2:
        st.markdown("### 🟢 Factors Decreasing Churn Risk")
        if shap_data['top_negative_factors']:
            for factor in shap_data['top_negative_factors']:
                st.markdown(f"""
                <div class="factor-negative">
                    <strong>{factor['feature'].replace('_', ' ').title()}</strong><br/>
                    <span style="color: #94A3B8;">Impact: </span>
                    <span style="color: #10B981;">{factor['shap_value']:.4f}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No factors significantly decreasing churn risk")


def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🚗 RideWise Churn Prediction</h1>
        <p>AI-powered customer analytics for proactive retention strategies</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check API health
    api_healthy = check_api_health()
    
    if not api_healthy:
        st.error("""
        ⚠️ **API Connection Failed**
        
        The prediction API is not running. Please start the API server:
        ```bash
        cd dashboard/api
        uvicorn main:app --reload --host 0.0.0.0 --port 8000
        ```
        """)
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎯 Rider Input")
        st.markdown("---")
        
        # Sample data selector
        samples = get_sample_riders()
        if samples:
            sample_names = ["Custom Input"] + [s['name'] for s in samples]
            selected_sample = st.selectbox(
                "Load Sample Rider",
                sample_names,
                help="Select a pre-defined rider profile or enter custom values"
            )
            
            if selected_sample != "Custom Input":
                sample_data = next(s for s in samples if s['name'] == selected_sample)
                st.info(f"📝 {sample_data['description']}")
                default_values = sample_data['features']
            else:
                default_values = {
                    'recency': 15,
                    'frequency': 20,
                    'monetary': 250.0,
                    'surge_exposure': 0.25,
                    'loyalty_status': 1,
                    'churn_prob': 0.30,
                    'rider_active_days': 200,
                    'rating_by_rider': 4.2
                }
        else:
            default_values = {
                'recency': 15,
                'frequency': 20,
                'monetary': 250.0,
                'surge_exposure': 0.25,
                'loyalty_status': 1,
                'churn_prob': 0.30,
                'rider_active_days': 200,
                'rating_by_rider': 4.2
            }
        
        st.markdown("### 📊 Rider Metrics")
        
        recency = st.number_input(
            "Recency (days since last ride)",
            min_value=0,
            max_value=365,
            value=int(default_values['recency']),
            help="Number of days since the rider's last trip"
        )
        
        frequency = st.number_input(
            "Frequency (total trips)",
            min_value=0,
            max_value=500,
            value=int(default_values['frequency']),
            help="Total number of trips taken by the rider"
        )
        
        monetary = st.number_input(
            "Monetary (total spending £)",
            min_value=0.0,
            max_value=10000.0,
            value=float(default_values['monetary']),
            step=10.0,
            help="Total amount spent by the rider"
        )
        
        surge_exposure = st.slider(
            "Surge Exposure (%)",
            min_value=0.0,
            max_value=1.0,
            value=float(default_values['surge_exposure']),
            step=0.05,
            help="Percentage of rides during surge pricing"
        )
        
        loyalty_options = {
            "Bronze (0)": 0,
            "Silver (1)": 1,
            "Gold (2)": 2,
            "Platinum (3)": 3
        }
        loyalty_default = [k for k, v in loyalty_options.items() if v == default_values['loyalty_status']][0]
        loyalty_status = st.selectbox(
            "Loyalty Status",
            options=list(loyalty_options.keys()),
            index=list(loyalty_options.keys()).index(loyalty_default),
            help="Rider's loyalty tier"
        )
        
        churn_prob = st.slider(
            "Historical Churn Score",
            min_value=0.0,
            max_value=1.0,
            value=float(default_values['churn_prob']),
            step=0.05,
            help="Historical probability score for churn risk"
        )
        
        rider_active_days = st.number_input(
            "Active Days (since signup)",
            min_value=1,
            max_value=1000,
            value=int(default_values['rider_active_days']),
            help="Days since the rider signed up"
        )
        
        rating_by_rider = st.slider(
            "Rating by Rider",
            min_value=1.0,
            max_value=5.0,
            value=float(default_values['rating_by_rider']),
            step=0.1,
            help="Average rating given by the rider to drivers"
        )
        
        st.markdown("---")
        predict_button = st.button("🔮 Predict Churn", use_container_width=True)
    
    # Main content area
    if predict_button:
        # Prepare features
        features = {
            "recency": recency,
            "frequency": frequency,
            "monetary": monetary,
            "surge_exposure": surge_exposure,
            "loyalty_status": loyalty_options[loyalty_status],
            "churn_prob": churn_prob,
            "rider_active_days": rider_active_days,
            "rating_by_rider": rating_by_rider
        }
        
        with st.spinner("🔄 Analyzing rider data..."):
            # Get prediction
            prediction = get_prediction(features)
            
            if prediction:
                st.markdown("## 📈 Prediction Results")
                
                # Display prediction cards
                render_prediction_result(prediction)
                
                st.markdown("---")
                
                # Gauge chart
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.plotly_chart(
                        create_gauge_chart(prediction['churn_probability']),
                        use_container_width=True
                    )
                
                # Get SHAP explanation
                with col2:
                    explanation = get_explanation(features)
                    if explanation:
                        st.plotly_chart(
                            create_shap_waterfall(explanation),
                            use_container_width=True
                        )
                
                st.markdown("---")
                
                # Key factors
                if explanation:
                    st.markdown("## 🔍 Key Factors Driving This Prediction")
                    render_key_factors(explanation)
                
                st.markdown("---")
                
                # Global feature importance
                st.markdown("## 🌐 Global Model Insights")
                importance = get_feature_importance()
                if importance:
                    st.plotly_chart(
                        create_feature_importance_chart(importance),
                        use_container_width=True
                    )
                    
                    st.markdown("""
                    <div class="info-box">
                        <strong>📊 Understanding Feature Importance</strong><br/>
                        This chart shows which features have the most influence on predictions across all riders.
                        Higher values indicate stronger predictive power for determining churn risk.
                    </div>
                    """, unsafe_allow_html=True)
                
                # Recommendations
                st.markdown("---")
                st.markdown("## 💡 Retention Recommendations")
                
                if prediction['risk_level'] in ['High', 'Critical']:
                    st.error("""
                    **⚠️ Immediate Action Required**
                    
                    This rider shows high churn risk. Consider:
                    - 🎁 Send personalized discount or promotion
                    - 📞 Proactive customer service outreach
                    - 🎯 Targeted re-engagement campaign
                    - ⭐ Loyalty program upgrade offer
                    """)
                elif prediction['risk_level'] == 'Medium':
                    st.warning("""
                    **⚡ Monitor & Engage**
                    
                    This rider shows moderate churn risk. Consider:
                    - 📧 Regular engagement communications
                    - 🎉 Occasional promotional offers
                    - 📊 Monitor activity trends closely
                    """)
                else:
                    st.success("""
                    **✅ Healthy Engagement**
                    
                    This rider shows low churn risk. Maintain engagement:
                    - 🙏 Thank you messages for loyalty
                    - 🏆 Loyalty rewards and recognition
                    - 📣 Referral program incentives
                    """)
    
    else:
        # Default view when no prediction is made
        st.markdown("""
        ## 👋 Welcome to the RideWise Churn Prediction Dashboard
        
        This dashboard uses machine learning to predict whether a rider is likely to churn,
        helping you take proactive retention actions.
        
        ### 🚀 How to Use
        1. **Enter rider data** in the sidebar (or select a sample profile)
        2. Click **"Predict Churn"** to analyze
        3. Review the **prediction results** and **SHAP explanations**
        4. Take action based on **retention recommendations**
        
        ### 📊 Features Analyzed
        - **Recency**: Days since last ride
        - **Frequency**: Total number of trips  
        - **Monetary**: Total spending
        - **Surge Exposure**: Rides during surge pricing
        - **Loyalty Status**: Bronze, Silver, Gold, or Platinum
        - **Historical Churn Score**: Previous risk assessment
        - **Active Days**: Time since signup
        - **Rating by Rider**: Average rating given
        """)
        
        # Show global feature importance by default
        st.markdown("---")
        st.markdown("## 🌐 Global Feature Importance")
        
        importance = get_feature_importance()
        if importance:
            st.plotly_chart(
                create_feature_importance_chart(importance),
                use_container_width=True
            )


if __name__ == "__main__":
    main()
