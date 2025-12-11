import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="شفافية النموذج والمعلومات",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS with modern design
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;600;700&display=swap');
    
    html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"], .block-container {
        direction: rtl !important;
        text-align: right !important;
        font-family: 'Cairo', system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    
    .model-card {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
    }
    
    .metric-highlight {
        background: linear-gradient(135deg, #0ea5e9 0%, #3b82f6 100%);
        color: white;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    
    .variable-card {
        background: white;
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        border-left: 4px solid;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.03);
    }
    
    .section-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 1.5rem;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid #f1f5f9;
    }
    
    .perf-metric {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #e2e8f0;
    }
    
    .perf-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1e293b;
        line-height: 1;
    }
    
    .perf-label {
        font-size: 0.9rem;
        color: #64748b;
        margin-top: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

def main():
    
    # Create tabs for different sections
    tab1, tab2= st.tabs([
        "📊 نظرة عامة على النموذج",
        "⚙️ المتغيرات والميزات", 
    ])
    
    with tab1:
        st.markdown('<div class="section-title">نظرة عامة على النموذج</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            <div class="model-card">
                <h3 style='color: #1e40af; margin-top: 0;'>نموذج توقع الإيرادات الضريبية - LASSO</h3>
                <p style='font-size: 1.05rem; line-height: 1.6;'>
                نموذج متقدم لتوقع الإيرادات الضريبية الفيدرالية باستخدام تحليل السلاسل الزمنية والتعلم الآلي.
                يعتمد على بيانات ربع سنوية من 1947 إلى الوقت الحاضر مع 26 متغيرًا مميزًا.
                </p>
                <ul style='line-height: 1.8; padding-right: 1.5rem;'>
                    <li><strong>البنية:</strong> نموذج هجين يجمع بين 3 نماذج (LASSO, Ridge, XGBoost)</li>
                    <li><strong>البيانات:</strong> بيانات ربع سنوية من 1947-2024 (305 نقطة بيانات)</li>
                    <li><strong>الهدف:</strong> توقع الإيرادات الضريبية ربع السنوية القادمة</li>
                    <li><strong>المتغيرات:</strong> 26 متغيرًا مميزًا مشتقًا من 5 متغيرات أساسية</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-highlight">
                <div style='font-size: 2.5rem; font-weight: 700;'>80.6%</div>
                <div style='font-size: 1rem; opacity: 0.9;'>دقة التوقع</div>
                <div style='font-size: 0.85rem; margin-top: 0.5rem;'>R² Score (Cross-Validation)</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Model Architecture Diagram
        st.markdown('<div class="section-title">الهندسة المعمارية للنموذج</div>', unsafe_allow_html=True)
        
        # Create a visual flow diagram
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=[1, 2, 3, 4, 5, 6],
            y=[1, 1, 1, 1, 1, 1],
            mode='markers+text',
            marker=dict(size=[60, 60, 60, 60, 60, 60], 
                       color=['#3b82f6', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981', '#8b5cf6']),
            text=['البيانات\nالخام', 'هندسة\nالميزات', 'مقياس\nالتطبيع', 'نموذج\nLASSO', 'النموذج\nالهجين', 'التنبؤ'],
            textposition="top center",
            textfont=dict(size=12, color='white', weight='bold'),
        ))
        
        for i in range(5):
            fig.add_trace(go.Scatter(
                x=[i+1, i+2],
                y=[1, 1],
                mode='lines',
                line=dict(color='#94a3b8', width=3, dash='dash'),
            ))
        
        fig.update_layout(
            showlegend=False,
            height=200,
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
            plot_bgcolor='white',
            margin=dict(l=20, r=20, t=20, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Model specifications from the notebook
        st.markdown('<div class="section-title">مواصفات النموذج الحقيقي</div>', unsafe_allow_html=True)
        
        spec_cols = st.columns(4)
        
        specs = [
            ("📊 أفضل نموذج", "LASSO Regression", "alpha=0.1 - تم اختياره بناءً على الأداء"),
            ("🔢 الميزات", "26 متغيرًا", "مشتقة من 5 متغيرات اقتصادية أساسية"),
            ("🎯 Cross-Validation R²", "80.6%", "أفضل نتيجة بين 5 نماذج تم اختبارها"),
            ("📅 البيانات", "1947-2024", "بيانات ربع سنوية (305 فترة)")
        ]
        
        for idx, (title, value, desc) in enumerate(specs):
            with spec_cols[idx]:
                st.markdown(f"""
                <div class="perf-metric">
                    <div style='font-size: 0.9rem; color: #64748b; margin-bottom: 0.5rem;'>{title}</div>
                    <div class="perf-value">{value}</div>
                    <div class="perf-label">{desc}</div>
                </div>
                """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="section-title">المتغيرات الرئيسية (من الـ.ipynb)</div>', unsafe_allow_html=True)
        
        # Actual feature importance from notebook
        features = [
            'متوسط الإيرادات الضريبية (4 فترات)',
            'الناتج المحلي الإجمالي (GDP)',
            'متوسط GDP (4 فترات)',
            'الإيرادات الضريبية (تأخر ربع)',
            'الإيرادات الضريبية (تأخر نصف سنة)',
            'GDP (تأخر ربع)'
        ]
        
        importance = [0.193, 0.137, 0.087, 0.066, 0.064, 0.060]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=features,
            x=importance,
            orientation='h',
            marker=dict(
                color=['#3b82f6', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981', '#ef4444'],
                line=dict(color='white', width=1)
            ),
            text=[f'{imp*100:.1f}%' for imp in importance],
            textposition='outside',
        ))
        
        fig.update_layout(
            title='أهم 6 متغيرات تأثيراً (من أصل 26 متغيراً)',
            height=350,
            xaxis=dict(title='مستوى الأهمية', range=[0, 0.25]),
            yaxis=dict(title='', autorange='reversed'),
            plot_bgcolor='white',
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-title">القيود والتحديات الواقعية</div>', unsafe_allow_html=True)
        
        # Place the two cards side-by-side (RTL layout keeps them visually right→left)
        col_warn, col_constraints = st.columns(2)
        
        with col_warn:
            st.markdown("""
            <div class="model-card">
                <h4 style='color: #dc2626; margin-top: 0;'>🔍 تحذيرات هامة</h4>
                <ul style='line-height: 1.8; padding-right: 1.5rem;'>
                     <li>النموذج يعمل بشكل أفضل كأداة مساعدة وليس كمنظومة تنبؤية مستقلة</li>
                    <li>التنبؤات تعتمد على استمرارية العلاقات التاريخية</li>
                    <li>عدم القدرة على توقع الصدمات الاقتصادية المفاجئة</li>
                    <li>النتائج السلبية لنماذج الأشجار تشير إلى مشكلة في البيانات أو الهندسة</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col_constraints:
            st.markdown("""
            <div class="model-card">
                <h4 style='color: #ea580c; margin-top: 0;'>⚠️ قيود البيانات</h4>
                <ul style='line-height: 1.8; padding-right: 1.5rem;'>
                    <li>بيانات ربع سنوية فقط (حدود التحديث)</li>
                    <li>305 نقطة بيانات فقط للتدريب</li>
                    <li>نقص في بيانات السياسات الضريبية</li>
                    <li>عدم وجود بيانات موسمية مفصلة</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="section-title">التطوير التقني</div>', unsafe_allow_html=True)
        
        # Python libraries used
        st.markdown('<div class="section-title">المكتبات المستخدمة</div>', unsafe_allow_html=True)
        
        libs_cols = st.columns(3)
        
        libraries = {
            "🤖 تعلم الآلة": ["scikit-learn", "xgboost", "lightgbm", "shap"],
            "📊 معالجة البيانات": ["pandas", "numpy", "statsmodels", "pmdarima"],
            "📈 التصور": ["matplotlib", "seaborn", "plotly", "ipywidgets"]
        }
        
        for idx, (category, lib_list) in enumerate(libraries.items()):
            with libs_cols[idx]:
                st.markdown(f"""
                <div class="model-card">
                    <h5 style='color: #1e40af; margin-top: 0;'>{category}</h5>
                    {"".join([f'<div style="background: #e0f2fe; padding: 4px 8px; margin: 2px; border-radius: 4px; display: inline-block; font-size: 0.85rem;">{lib}</div>' for lib in lib_list])}
                </div>
                """, unsafe_allow_html=True)  
         
    # Footer with actual information
    st.markdown("""
        <div style='margin-top: 3rem; padding-top: 1.5rem; border-top: 1px solid #e2e8f0; text-align: center; color: #64748b;'>     
            <p style='font-size: 0.8rem; margin-top: 1rem;'>
                مصدر البيانات: W006RC1Q027SBEA, CPIAUCSL, GDP, EXPGS, IMPGS | 
                آخر تحديث: {}
            </p>
        </div>
    """.format(datetime.now().strftime('%Y-%m-%d')), unsafe_allow_html=True)

if __name__ == "__main__":
    main()