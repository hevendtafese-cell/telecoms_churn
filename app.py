import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
from tensorflow import keras

import plotly.graph_objects as go

st.set_page_config(
    page_title="Churn Prediction AI",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .high-risk {
        background-color: #ffebee;
        color: #c62828;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #c62828;
    }
    .low-risk {
        background-color: #e8f5e9;
        color: #2e7d32;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2e7d32;
    }
    [data-testid="stSidebar"] {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_metrics(mtime):
    
    """Always load the latest metrics JSON"""
    try:
        with open('models/metrics.json', 'r') as f:
        return json.load(f)
      
    except Exception as e:
        st.error(f"Error loading metrics: {e}")
        return None
def load_model():
    """Load trained model"""

    try:
        return keras.models.load_model('models/churn_model_baseline.keras')
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def load_preprocessing():
    """Load preprocessing objects"""
    try:
        with open('models/preprocessing.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading preprocessing: {e}")
        return None

def format_currency(value):
    """Format as GBP currency"""
    return f"£{value:,.0f}"

def format_percentage(value):
    """Format as percentage"""
    return f"{value*100:.2f}%"

def create_gauge(probability):
    """Create risk gauge"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Churn Risk %", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': '#e8f5e9'},
                {'range': [30, 50], 'color': '#fff9c4'},
                {'range': [50, 70], 'color': '#ffccbc'},
                {'range': [70, 100], 'color': '#ffebee'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=60, b=10))
    return fig

def display_model_metrics(metrics, show_header=True):
    """Display model metrics consistently"""
    if show_header:
        st.markdown(f"### {metrics['model_name']}")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", format_percentage(metrics['accuracy']))
    col2.metric("Recall", format_percentage(metrics['recall']))
    col3.metric("Precision", format_percentage(metrics['precision']))
    col4.metric("F1-Score", f"{metrics['f1_score']:.3f}")
    
    st.markdown("#### Business Impact")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Customers Saved", f"{metrics['churners_caught']}/{metrics['total_churners']}")
    col2.metric("Revenue Saved", format_currency(metrics['revenue_saved']))
    col3.metric("Campaign Cost", format_currency(metrics['campaign_cost']))
    col4.metric("Net Benefit", format_currency(metrics['net_benefit']))
    
    col1, col2, col3 = st.columns(3)
    col1.metric("ROI", f"{metrics['roi']:.0f}%")
    col2.metric("Offers Sent", metrics['offers_sent'])
    col3.metric("Revenue per £1", f"£{metrics['revenue_per_dollar_spent']:.2f}")
    
    with st.expander("📋 Calculation Details"):
        st.markdown(f"""
        **Revenue Calculation:**
        - Customers saved: {metrics['churners_caught']}
        - Customer Lifetime Value: {format_currency(metrics['clv'])}
        - Revenue saved: {metrics['churners_caught']} × {format_currency(metrics['clv'])} = {format_currency(metrics['revenue_saved'])}
        
        **Campaign Cost:**
        - Offers sent: {metrics['offers_sent']} (to all predicted churners)
        - Cost per offer: {format_currency(metrics['retention_cost'])}
        - Total cost: {metrics['offers_sent']} × {format_currency(metrics['retention_cost'])} = {format_currency(metrics['campaign_cost'])}
        
        **Net Benefit:**
        - {format_currency(metrics['revenue_saved'])} - {format_currency(metrics['campaign_cost'])} = {format_currency(metrics['net_benefit'])}
        
        **ROI:**
        - ({format_currency(metrics['net_benefit'])} / {format_currency(metrics['campaign_cost'])}) × 100 = {metrics['roi']:.0f}%
        """)
    
    with st.expander("🔢 Confusion Matrix"):
        cm_data = {
            'Predicted Stay': [metrics['true_negatives'], metrics['false_negatives']],
            'Predicted Churn': [metrics['false_positives'], metrics['true_positives']]
        }
        cm_df = pd.DataFrame(cm_data, index=['Actual Stay', 'Actual Churn'])
        st.dataframe(cm_df, use_container_width=True)
        
        st.markdown(f"""
        - **True Negatives:** {metrics['true_negatives']} (correctly predicted stay)
        - **False Positives:** {metrics['false_positives']} (false alarms)
        - **False Negatives:** {metrics['false_negatives']} (missed churners )
        - **True Positives:** {metrics['true_positives']} (correctly caught churners )
        """)

def main():
    st.markdown('<h1 class="main-header">🤖 Churn Prediction AI</h1>', unsafe_allow_html=True)
    
  metrics_data = load_metrics(os.path.getmtime('models/metrics.json'))
    model = load_model() 
    preprocessing = load_preprocessing()
    
    
    tab1, tab2, tab3 = st.tabs(["🎯 Predict", "📊 Performance", "ℹ️ About"])
    
    with tab1:
        st.header("Customer Risk Assessment")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Demographics")
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.slider("Age", 18, 100, 45)
            married = st.selectbox("Married", ["Yes", "No"])
            dependents = st.number_input("Dependents", 0, 10, 0)
            
        with col2:
            st.subheader("Services")
            tenure = st.slider("Tenure (months)", 0, 72, 24)
            contract = st.selectbox("Contract", ["Month-to-Month", "One Year", "Two Year"])
            monthly_charge = st.number_input("Monthly Charge ($)", 0.0, 150.0, 65.0)
            internet_service = st.selectbox("Internet", ["Yes", "No"])
            
        with col3:
            st.subheader("Location")
            latitude = st.number_input("Latitude", 32.0, 42.0, 34.05, format="%.2f")
            longitude = st.number_input("Longitude", -125.0, -114.0, -118.25, format="%.2f")
            population = st.number_input("Population", 0, 110000, 50000)
        
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        
        multiple_lines = "Yes" if phone_service == "Yes" else "No phone service"
        internet_type = "Fiber Optic" if internet_service == "Yes" else "None"
        online_security = "Yes" if internet_service == "Yes" else "No internet service"
        online_backup = "Yes" if internet_service == "Yes" else "No internet service"
        device_protection = "Yes" if internet_service == "Yes" else "No internet service"
        tech_support = "Yes" if internet_service == "Yes" else "No internet service"
        streaming_tv = "Yes" if internet_service == "Yes" else "No internet service"
        streaming_movies = "Yes" if internet_service == "Yes" else "No internet service"
        streaming_music = "Yes" if internet_service == "Yes" else "No internet service"
        unlimited_data = "Yes" if internet_service == "Yes" else "No internet service"
        paperless_billing = "Yes"
        payment_method = "Bank Withdrawal"
        referrals = 2
        offer = "None"
        
        avg_monthly_gb = 25 if internet_service == "Yes" else 0
        avg_long_distance = 10.5 if phone_service == "Yes" else 0.0
        total_charges = monthly_charge * tenure
        total_refunds = 0.0
        total_extra_data = 5.0 if internet_service == "Yes" else 0.0
        total_long_distance = avg_long_distance * tenure
        
        if st.button("Predict Churn Risk", type="primary", use_container_width=True):
            input_data = [
                gender, age, married, dependents, latitude, longitude,
                referrals, tenure, offer, phone_service, avg_long_distance,
                multiple_lines, internet_service, internet_type, avg_monthly_gb,
                online_security, online_backup, device_protection, tech_support,
                streaming_tv, streaming_movies, streaming_music, unlimited_data,
                contract, paperless_billing, payment_method, monthly_charge,
                total_charges, total_refunds, total_extra_data, total_long_distance,
                population
            ]
            
            feature_names = preprocessing['feature_names']
            df = pd.DataFrame([input_data], columns=feature_names)
            
            label_encoders = preprocessing['label_encoders']
            for col, encoder in label_encoders.items():
                if col in df.columns:
                    try:
                        df[col] = encoder.transform(df[col])
                    except:
                        df[col] = encoder.transform([encoder.classes_[0]])[0]
            
            scaler = preprocessing['scaler']
            scaled = scaler.transform(df)
            proba = model.predict(scaled, verbose=0)[0][0]
            
            st.markdown("---")
            st.subheader("📊 Prediction Results")
            
            cola, colb = st.columns([1, 2])
            
            with cola:
                fig = create_gauge(proba)
                st.plotly_chart(fig, use_container_width=True)
            
            with colb:
                if metrics_data and 'model_3' in metrics_data:
                    clv = metrics_data['model_3']['clv']
                    ret_cost = metrics_data['model_3']['retention_cost']
                else:
                    clv = 1554
                    ret_cost = 70
                
                if proba >= 0.7:
                    st.markdown(f"""
                    <div class="high-risk">
                        <h3>⚠️ CRITICAL RISK: {proba*100:.1f}%</h3>
                        <p><strong>Action:</strong> Contact within 24 hours</p>
                        <p><strong>Offer:</strong> {format_currency(ret_cost)} retention package</p>
                        <p><strong>Loss if churned:</strong> {format_currency(clv)}</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif proba >= 0.5:
                    st.markdown(f"""
                    <div class="high-risk">
                        <h3>⚠️ HIGH RISK: {proba*100:.1f}%</h3>
                        <p><strong>Action:</strong> Proactive engagement</p>
                        <p><strong>Offer:</strong> 10-15% loyalty discount</p>
                        <p><strong>Loss if churned:</strong> {format_currency(clv)}</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif proba >= 0.3:
                    st.warning(f"⚡ **MODERATE RISK: {proba*100:.1f}%** - Monitor quarterly")
                else:
                    st.markdown(f"""
                    <div class="low-risk">
                        <h3>✅ LOW RISK: {proba*100:.1f}%</h3>
                        <p>Customer likely to stay</p>
                        <p><strong>CLV:</strong> {format_currency(monthly_charge * 24)}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                confidence = max(proba, 1 - proba)
                st.metric("Model Confidence", f"{confidence*100:.1f}%")
    
    with tab2:
        st.header("Model Performance Analysis")
        
        if metrics_data:
            st.markdown("### 📊 Model Comparison")
            
            comparison_data = []
            for key in ['model_1', 'model_2', 'model_3']:
                if key in metrics_data:
                    m = metrics_data[key]
                    comparison_data.append({
                        'Model': m['model_name'],
                        'Accuracy': format_percentage(m['accuracy']),
                        'Recall': format_percentage(m['recall']),
                        'Precision': format_percentage(m['precision']),
                        'Net Benefit': format_currency(m['net_benefit']),
                        'ROI': f"{m['roi']:.0f}%"
                    })
            
            if comparison_data:
                st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)
            
            st.markdown("---")
            
            for key in ['model_1', 'model_2', 'model_3']:
                if key in metrics_data:
                    m = metrics_data[key]
                    is_winner = (m['model_name'] == metrics_data.get('selected_model'))
                    
                    with st.expander(
                        f"{'🏆 ' if is_winner else ''}{m['model_name']}", 
                        expanded=is_winner
                    ):
                        display_model_metrics(m, show_header=False)
        else:
            st.error("No metrics available. Run notebook first.")
    
    with tab3:
        st.header("About This Application")
        
        if metrics_data and 'model_3' in metrics_data:
            m = metrics_data['model_3']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("###  Project Information")
                st.info("""
                **Course:** CSI_7_DEL Deep Learning  
                **Institution:** London South Bank University  
                """)
                
                st.markdown("###  Dataset")
                st.metric("Test Set Size", f"{m['total_customers']:,} customers")
                st.metric("Churn Rate", format_percentage(m['churn_rate']))
                st.metric("Features", "32")
                
                st.markdown("###  Selected Model")
                st.success(f"**{metrics_data.get('selected_model', 'Unknown')}**")
                st.metric("Test Accuracy", format_percentage(m['accuracy']))
                st.metric("Test Recall", format_percentage(m['recall']))
            
            with col2:
                st.markdown("###  Business Impact")
                st.metric("Customers Saved", f"{m['churners_caught']}/{m['total_churners']}")
                st.metric("Revenue Preserved", format_currency(m['revenue_saved']))
                st.metric("Campaign Cost", format_currency(m['campaign_cost']))
                st.metric("Net Benefit", format_currency(m['net_benefit']))
                st.metric("ROI", f"{m['roi']:.0f}%")
                
                st.markdown("### Assumptions")
                st.caption(f"CLV: {format_currency(m['clv'])} ({format_currency(m['avg_monthly_charge'])}/mo × {m['customer_lifetime_months']} months)")
                st.caption(f"Retention Cost: {format_currency(m['retention_cost'])} per offer")
        else:
            st.warning("Metrics not loaded")

if __name__ == "__main__":
    main()
