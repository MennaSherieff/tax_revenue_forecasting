import pathlib
import pickle
import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# ---------- CONFIG ---------- #
st.set_page_config(
    page_title="Risk Assessment Dashboard",
    page_icon="⚠️",
    layout="wide",
)


# ---------- UTILS ---------- #
@st.cache_resource
def load_models_and_data():
    """Load trained models, scaler, and historical data."""
    models_dir = pathlib.Path("models")
    
    # Load ensemble models
    models = {}
    model_files = {
        'XGB': 'xgb.pkl',
        #'LightGBM': 'lightgbm.pkl',
        #'RandomForest': 'randomforest.pkl',
        'Ridge': 'ridge.pkl',
        'Lasso': 'lasso.pkl'
    }
    
    for name, filename in model_files.items():
        path = models_dir / filename
        if path.exists():
            with open(path, 'rb') as f:
                models[name] = pickle.load(f)
    
    # Load scaler
    scaler_path = models_dir / 'scaler.pkl'
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # Load historical data
    df_eng = pd.read_csv('.\\data\\df_eng.csv', parse_dates=['date'])
    
    # Load feature info
    import json
    with open(models_dir / 'feature_info.json', 'r') as f:
        feature_info = json.load(f)
    
    # Load ensemble config
    with open(models_dir / 'ensemble_config.json', 'r') as f:
        ensemble_config = json.load(f)
    
    return models, scaler, df_eng, feature_info, ensemble_config


def calculate_risk_metrics(df_eng):
    """Calculate various risk metrics from historical data."""
    metrics = {}
    
    # Revenue volatility (annualized)
    metrics['volatility'] = df_eng['tax_revenue'].pct_change().std() * np.sqrt(4) * 100
    
    # Historical drawdown
    cummax = df_eng['tax_revenue'].cummax()
    drawdown = (df_eng['tax_revenue'] - cummax) / cummax * 100
    metrics['max_drawdown'] = drawdown.min()
    
    # Growth statistics
    metrics['avg_growth'] = df_eng['tax_revenue'].pct_change(4).mean() * 100
    metrics['growth_std'] = df_eng['tax_revenue'].pct_change(4).std() * 100
    
    # Tax-to-GDP ratio
    if 'gdp' in df_eng.columns:
        metrics['tax_gdp_ratio'] = (df_eng['tax_revenue'] / df_eng['gdp']).mean()
        metrics['tax_gdp_recent'] = (df_eng['tax_revenue'].iloc[-4:].sum() / 
                                     df_eng['gdp'].iloc[-4:].sum())
    
    # Largest historical drop
    metrics['largest_drop'] = df_eng['tax_revenue'].pct_change(4).min() * 100
    
    return metrics


def predict_with_uncertainty(models, scaler, features_dict, weights):
    """Make prediction with uncertainty from ensemble."""
    # Create feature array matching training features
    feature_names = list(features_dict.keys())
    X = np.array([[features_dict[f] for f in feature_names]])
    X_scaled = scaler.transform(X)
    
    predictions = []
    for name, model in models.items():
        # Extract model from pipeline if needed
        if hasattr(model, 'named_steps'):
            pred = model.named_steps['model'].predict(X_scaled)[0]
        else:
            pred = model.predict(X_scaled)[0]
        predictions.append(pred)
    
    # Ensemble prediction (weighted average)
    ensemble_pred = np.average(predictions, weights=weights[:len(predictions)])
    
    # Uncertainty (std across models)
    uncertainty = np.std(predictions)
    
    return ensemble_pred, uncertainty, predictions


def stress_test_scenarios(models, scaler, base_features, weights):
    """Run stress test scenarios."""
    scenarios = {
        'Base Case': {},
        'GDP Shock (-10%)': {'gdp': -0.10, 'gdp_lag_1': -0.10},
        'Export Decline (-15%)': {'exports': -0.15, 'exports_lag_1': -0.15},
        'Import Surge (+20%)': {'imports': 0.20, 'imports_lag_1': 0.20},
        'Recession Scenario': {'gdp': -0.08, 'exports': -0.12, 'cpi': -0.02},
        'Inflation Shock': {'cpi': 0.15, 'cpi_lag_1': 0.15},
    }
    
    results = {}
    base_pred, _, _ = predict_with_uncertainty(models, scaler, base_features, weights)
    
    for scenario_name, changes in scenarios.items():
        # Apply changes to base features
        modified_features = base_features.copy()
        for feature, pct_change in changes.items():
            if feature in modified_features:
                modified_features[feature] *= (1 + pct_change)
        
        pred, uncertainty, _ = predict_with_uncertainty(models, scaler, modified_features, weights)
        
        results[scenario_name] = {
            'prediction': pred,
            'change_from_base': ((pred / base_pred) - 1) * 100,
            'uncertainty': uncertainty
        }
    
    return results


# ---------- LOAD DATA ---------- #
try:
    models, scaler, df_eng, feature_info, ensemble_config = load_models_and_data()
    risk_metrics = calculate_risk_metrics(df_eng)
    ensemble_weights = np.array(ensemble_config['ensemble_weights'])
except Exception as e:
    st.error(f"Error loading models or data: {e}")
    st.info("Make sure 'saved_models/' directory and 'df_eng.csv' exist in the root folder.")
    st.stop()


# ---------- SIDEBAR ---------- #
with st.sidebar:
    st.title("⚙️ Settings")
    
    st.markdown("### Risk Thresholds")
    volatility_threshold = st.slider(
        "Volatility Alert (%)",
        min_value=5.0,
        max_value=20.0,
        value=12.0,
        step=0.5,
        help="Alert if annualized volatility exceeds this threshold"
    )
    
    growth_threshold = st.slider(
        "Low Growth Alert (%)",
        min_value=0.0,
        max_value=5.0,
        value=2.0,
        step=0.5,
        help="Alert if 5-year average growth falls below this"
    )
    
    st.markdown("### Analysis Period")
    lookback_years = st.selectbox(
        "Historical Data",
        options=[3, 5, 10, 20, "All"],
        index=2,
        help="Years of historical data to analyze"
    )


# ---------- MAIN UI ---------- #
st.title("⚠️ Risk Assessment Dashboard")
st.markdown("Comprehensive risk analysis for federal tax revenue forecasting")

# ---------- KEY RISK INDICATORS ---------- #
st.header("🎯 Key Risk Indicators")

col1, col2, col3, col4 = st.columns(4)

with col1:
    vol_status = "🔴" if risk_metrics['volatility'] > volatility_threshold else "🟢"
    st.metric(
        "Revenue Volatility",
        f"{risk_metrics['volatility']:.1f}%",
        help="Annualized standard deviation of quarterly revenue changes"
    )
    st.caption(f"{vol_status} {'HIGH RISK' if risk_metrics['volatility'] > volatility_threshold else 'ACCEPTABLE'}")

with col2:
    growth_status = "🔴" if risk_metrics['avg_growth'] < growth_threshold else "🟢"
    st.metric(
        "Avg YoY Growth",
        f"{risk_metrics['avg_growth']:.1f}%",
        help="Average year-over-year growth rate"
    )
    st.caption(f"{growth_status} {'LOW GROWTH' if risk_metrics['avg_growth'] < growth_threshold else 'HEALTHY'}")

with col3:
    st.metric(
        "Max Drawdown",
        f"{risk_metrics['max_drawdown']:.1f}%",
        help="Largest peak-to-trough decline in history"
    )
    st.caption("📉 Historical worst case")

with col4:
    tax_gdp_status = "🟡" if risk_metrics['tax_gdp_recent'] < risk_metrics['tax_gdp_ratio'] * 0.95 else "🟢"
    st.metric(
        "Tax/GDP Ratio",
        f"{risk_metrics['tax_gdp_recent']:.3f}",
        delta=f"{((risk_metrics['tax_gdp_recent']/risk_metrics['tax_gdp_ratio'])-1)*100:.1f}%",
        help="Recent tax-to-GDP ratio vs historical average"
    )
    st.caption(f"{tax_gdp_status} Tax efficiency")


# ---------- VOLATILITY ANALYSIS ---------- #
st.header("📊 Volatility Analysis")

col1, col2 = st.columns(2)

with col1:
    # Rolling volatility chart
    rolling_vol = df_eng['tax_revenue'].pct_change().rolling(window=8).std() * np.sqrt(4) * 100
    
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(
        x=df_eng['date'],
        y=rolling_vol,
        mode='lines',
        name='Rolling Volatility',
        line=dict(color='#FF6B6B', width=2)
    ))
    fig_vol.add_hline(
        y=volatility_threshold,
        line_dash="dash",
        line_color="red",
        annotation_text="Alert Threshold"
    )
    fig_vol.update_layout(
        title="Rolling 8-Quarter Volatility",
        xaxis_title="Date",
        yaxis_title="Annualized Volatility (%)",
        height=350
    )
    st.plotly_chart(fig_vol, use_container_width=True)

with col2:
    # Drawdown chart
    cummax = df_eng['tax_revenue'].cummax()
    drawdown = (df_eng['tax_revenue'] - cummax) / cummax * 100
    
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=df_eng['date'],
        y=drawdown,
        fill='tozeroy',
        mode='lines',
        name='Drawdown',
        line=dict(color='#E74C3C', width=1),
        fillcolor='rgba(231, 76, 60, 0.3)'
    ))
    fig_dd.update_layout(
        title="Revenue Drawdown from Peak",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        height=350
    )
    st.plotly_chart(fig_dd, use_container_width=True)


# ---------- STRESS TESTING ---------- #
st.header("🧪 Stress Testing")

st.markdown("**Scenario Analysis:** How tax revenue responds to economic shocks")

# Get latest features from df_eng for stress testing
latest_features = {}
for feature in feature_info['features']:
    if feature in df_eng.columns:
        latest_features[feature] = float(df_eng[feature].iloc[-1])

# Run stress tests
stress_results = stress_test_scenarios(models, scaler, latest_features, ensemble_weights)

# Display stress test results
stress_df = pd.DataFrame(stress_results).T
stress_df = stress_df.reset_index().rename(columns={'index': 'Scenario'})

col1, col2 = st.columns([3, 2])

with col1:
    # Stress test table
    st.dataframe(
        stress_df.style.format({
            'prediction': '${:,.2f}B',
            'change_from_base': '{:+.2f}%',
            'uncertainty': '${:,.2f}B'
        }).background_gradient(subset=['change_from_base'], cmap='RdYlGn', vmin=-15, vmax=5),
        use_container_width=True,
        height=280
    )

with col2:
    # Stress test visualization
    fig_stress = go.Figure()
    
    colors = ['#2ECC71' if x >= 0 else '#E74C3C' for x in stress_df['change_from_base']]
    
    fig_stress.add_trace(go.Bar(
        y=stress_df['Scenario'],
        x=stress_df['change_from_base'],
        orientation='h',
        marker=dict(color=colors),
        text=stress_df['change_from_base'].apply(lambda x: f'{x:+.1f}%'),
        textposition='outside'
    ))
    
    fig_stress.update_layout(
        title="Impact vs Base Case",
        xaxis_title="Change in Revenue (%)",
        height=280,
        showlegend=False
    )
    st.plotly_chart(fig_stress, use_container_width=True)


# ---------- GROWTH RISK ANALYSIS ---------- #
st.header("📈 Growth Risk Analysis")

col1, col2 = st.columns(2)

with col1:
    # YoY growth distribution
    yoy_growth = df_eng['tax_revenue'].pct_change(4) * 100
    
    fig_growth_dist = go.Figure()
    fig_growth_dist.add_trace(go.Histogram(
        x=yoy_growth.dropna(),
        nbinsx=30,
        name='YoY Growth',
        marker=dict(color='#3498DB', line=dict(color='white', width=1))
    ))
    fig_growth_dist.add_vline(
        x=growth_threshold,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Alert: {growth_threshold}%"
    )
    fig_growth_dist.add_vline(
        x=yoy_growth.mean(),
        line_dash="dot",
        line_color="green",
        annotation_text=f"Mean: {yoy_growth.mean():.1f}%"
    )
    fig_growth_dist.update_layout(
        title="Distribution of YoY Growth Rates",
        xaxis_title="Growth Rate (%)",
        yaxis_title="Frequency",
        height=350
    )
    st.plotly_chart(fig_growth_dist, use_container_width=True)

with col2:
    # Growth trend with moving average
    rolling_growth = yoy_growth.rolling(window=4).mean()
    
    fig_growth_trend = go.Figure()
    fig_growth_trend.add_trace(go.Scatter(
        x=df_eng['date'],
        y=yoy_growth,
        mode='lines',
        name='YoY Growth',
        line=dict(color='lightblue', width=1),
        opacity=0.5
    ))
    fig_growth_trend.add_trace(go.Scatter(
        x=df_eng['date'],
        y=rolling_growth,
        mode='lines',
        name='4Q Moving Average',
        line=dict(color='#3498DB', width=3)
    ))
    fig_growth_trend.add_hline(
        y=growth_threshold,
        line_dash="dash",
        line_color="red",
        annotation_text="Alert Threshold"
    )
    fig_growth_trend.add_hline(y=0, line_dash="dot", line_color="gray")
    fig_growth_trend.update_layout(
        title="YoY Growth Trend",
        xaxis_title="Date",
        yaxis_title="Growth Rate (%)",
        height=350
    )
    st.plotly_chart(fig_growth_trend, use_container_width=True)


# ---------- RISK ALERTS ---------- #
st.header("🚨 Active Risk Alerts")

alerts = []

# Check volatility
if risk_metrics['volatility'] > volatility_threshold:
    alerts.append({
        'severity': '🔴 HIGH',
        'category': 'Volatility',
        'message': f"Revenue volatility ({risk_metrics['volatility']:.1f}%) exceeds threshold ({volatility_threshold}%)",
        'recommendation': "Consider hedging strategies or building larger revenue reserves"
    })

# Check growth
recent_growth = df_eng['tax_revenue'].pct_change(4).iloc[-20:].mean() * 100
if recent_growth < growth_threshold:
    alerts.append({
        'severity': '🔴 HIGH',
        'category': 'Growth',
        'message': f"5-year average growth ({recent_growth:.1f}%) below threshold ({growth_threshold}%)",
        'recommendation': "Review economic policies and growth stimulation measures"
    })

# Check tax efficiency
if risk_metrics['tax_gdp_recent'] < risk_metrics['tax_gdp_ratio'] * 0.95:
    alerts.append({
        'severity': '🟡 MEDIUM',
        'category': 'Tax Efficiency',
        'message': f"Tax-to-GDP ratio declining (current: {risk_metrics['tax_gdp_recent']:.3f}, avg: {risk_metrics['tax_gdp_ratio']:.3f})",
        'recommendation': "Review tax collection efficiency and compliance measures"
    })

# Check recent trend
if df_eng['tax_revenue'].pct_change().iloc[-1] < -0.05:
    alerts.append({
        'severity': '🟡 MEDIUM',
        'category': 'Recent Trend',
        'message': "Revenue declined more than 5% last quarter",
        'recommendation': "Monitor closely for emerging downward trend"
    })

if alerts:
    for alert in alerts:
        with st.expander(f"{alert['severity']} - {alert['category']}", expanded=True):
            st.warning(alert['message'])
            st.info(f"**Recommendation:** {alert['recommendation']}")
else:
    st.success("✅ No active risk alerts. All metrics within acceptable ranges.")


# ---------- RISK SUMMARY ---------- #
st.header("📋 Risk Summary & Recommendations")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Current Risk Profile")
    
    risk_score = 0
    if risk_metrics['volatility'] > volatility_threshold:
        risk_score += 3
    if recent_growth < growth_threshold:
        risk_score += 3
    if risk_metrics['tax_gdp_recent'] < risk_metrics['tax_gdp_ratio'] * 0.95:
        risk_score += 2
    if df_eng['tax_revenue'].pct_change().iloc[-1] < -0.05:
        risk_score += 2
    
    if risk_score >= 7:
        risk_level = "🔴 HIGH RISK"
    elif risk_score >= 4:
        risk_level = "🟡 MEDIUM RISK"
    else:
        risk_level = "🟢 LOW RISK"
    
    st.markdown(f"**Overall Risk Level:** {risk_level}")
    st.progress(min(risk_score / 10, 1.0))
    
    st.markdown(f"""
    - Volatility Risk: {'High' if risk_metrics['volatility'] > volatility_threshold else 'Low'}
    - Growth Risk: {'High' if recent_growth < growth_threshold else 'Low'}
    - Efficiency Risk: {'Medium' if risk_metrics['tax_gdp_recent'] < risk_metrics['tax_gdp_ratio'] * 0.95 else 'Low'}
    """)

with col2:
    st.markdown("### Key Recommendations")
    
    recommendations = []
    
    if risk_metrics['volatility'] > 15:
        recommendations.append("1. Diversify revenue sources to reduce volatility")
    
    if recent_growth < 2:
        recommendations.append("2. Focus on economic growth initiatives")
    
    if risk_metrics['tax_gdp_recent'] < risk_metrics['tax_gdp_ratio']:
        recommendations.append("3. Review tax collection efficiency")
    
    recommendations.append("4. Monitor leading economic indicators closely")
    recommendations.append("5. Maintain adequate fiscal reserves for downturns")
    
    for rec in recommendations:
        st.markdown(rec)


# ---------- FOOTER ---------- #
st.markdown("---")
st.caption(f"Last updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Data through: {df_eng['date'].max().strftime('%Y-%m-%d')}")