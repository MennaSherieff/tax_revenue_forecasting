import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="مستكشف التنبؤات", page_icon="📈")

st.title("📈 مستكشف التنبؤات")
st.markdown("### تحليل معمق للتنبؤات والسيناريوهات")

# Generate mock forecast data
@st.cache_data
def generate_forecast_data():
    years = list(range(2010, 2030))
    historical = [120, 135, 148, 162, 180, 195, 210, 230, 250, 265, 240, 260, 290, 320, 355]
    forecast = [380, 410, 442, 475, 512]
    
    all_revenue = historical + forecast
    
    return {
        'years': years,
        'revenue': all_revenue,
        'forecast_start': 2025
    }

data = generate_forecast_data()

# Main forecast chart
st.subheader("التنبؤ طويل المدى للإيرادات")

fig = go.Figure()

# Historical
fig.add_trace(go.Scatter(
    x=data['years'][:data['forecast_start']-2010],
    y=data['revenue'][:data['forecast_start']-2010],
    mode='lines+markers',
    name='تاريخي',
    line=dict(color='#1e3a5f', width=3),
    marker=dict(size=8)
))

# Forecast
fig.add_trace(go.Scatter(
    x=data['years'][data['forecast_start']-2010:],
    y=data['revenue'][data['forecast_start']-2010:],
    mode='lines+markers',
    name='تنبؤ',
    line=dict(color='#ffc107', width=3, dash='dash'),
    marker=dict(size=8)
))

fig.update_layout(
    xaxis_title="السنة",
    yaxis_title="الإيرادات (مليار جنيه)",
    height=500,
    template='plotly_white',
    hovermode='x unified'
)

st.plotly_chart(fig, use_container_width=True)

# Sensitivity Analysis
st.markdown("---")
st.subheader("تحليل حساسية العوامل")

st.markdown("عدل العوامل أدناه لرؤية تأثيرها على توقعات 2025")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**النمو الاقتصادي**")
    gdp_growth = st.slider("", -2.0, 8.0, 4.5, 0.1, label_visibility="collapsed")
    
    st.markdown("**معدل التضخم**")
    inflation = st.slider("", 0.0, 20.0, 8.0, 0.5, label_visibility="collapsed")

with col2:
    st.markdown("**فعالية التطبيق**")
    enforcement = st.slider("", 50, 100, 80, 5, label_visibility="collapsed")
    
    st.markdown("**معدل الضريبة**")
    tax_rate = st.slider("", -5.0, 5.0, 0.0, 0.5, label_visibility="collapsed")

with col3:
    st.markdown("**الامتثال الضريبي**")
    compliance = st.slider("", 0, 30, 10, 1, label_visibility="collapsed")

# Calculate impact
base_2025 = 380
impact = (gdp_growth * 2.5) + (enforcement * 0.8) + (compliance * 1.2) + (tax_rate * 15) - (inflation * 0.5)
adjusted = base_2025 + impact

col_metric1, col_metric2 = st.columns(2)

with col_metric1:
    st.metric("التوقع الأساسي 2025", f"{base_2025}B EGP")
    
with col_metric2:
    st.metric("التوقع المعدل 2025", f"{adjusted:.1f}B EGP", f"{impact:+.1f}B EGP")

# Scenario Comparison
st.markdown("---")
st.subheader("مقارنة السيناريوهات")

scenarios = pd.DataFrame({
    'السيناريو': ['الأساسي', 'متفائل', 'متشائم', 'التعديل الحالي'],
    '2025': [380, 410, 350, adjusted],
    '2026': [410, 450, 370, adjusted * 1.08],
    '2027': [442, 495, 390, adjusted * 1.16],
})

scenario_fig = go.Figure()

colors = ['#1e3a5f', '#28a745', '#dc3545', '#ffc107']

for i, scenario in enumerate(scenarios['السيناريو']):
    scenario_data = scenarios[scenarios['السيناريو'] == scenario]
    scenario_fig.add_trace(go.Scatter(
        x=[2025, 2026, 2027],
        y=[scenario_data['2025'].values[0], scenario_data['2026'].values[0], scenario_data['2027'].values[0]],
        mode='lines+markers',
        name=scenario,
        line=dict(color=colors[i], width=3),
        marker=dict(size=8)
    ))

scenario_fig.update_layout(
    xaxis_title="السنة",
    yaxis_title="الإيرادات (مليار جنيه)",
    height=400,
    template='plotly_white'
)

st.plotly_chart(scenario_fig, use_container_width=True)

# Export options
st.markdown("---")
if st.button("📥 تصدير بيانات التوقعات"):
    scenarios_csv = scenarios.to_csv(index=False)
    st.download_button(
        label="تحميل كملف CSV",
        data=scenarios_csv,
        file_name="scenarios_forecast.csv",
        mime="text/csv"
    )