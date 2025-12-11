# ...existing code...
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import pathlib
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="تحليل الإيرادات الضريبية حسب القطاع",
    page_icon="💰",
    layout="wide"
)

# Custom CSS for Arabic support, Cairo font and RTL layout
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;600;700&display=swap');

    /* Global RTL and font */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"], .block-container {
        direction: rtl !important;
        text-align: right !important;
        font-family: 'Cairo', system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }

    /* Preserve LTR for numeric controls and tables */
    .stDataFrame table, .stDataFrame th, .stDataFrame td,
    [data-testid="stSlider"], input, select {
        direction: ltr !important;
    }

    /* Local RTL helper class */
    .rtl-text {
        direction: rtl;
        text-align: right;
        font-family: 'Cairo', sans-serif;
    }

    /* Card styling (light) */
    .metric-card {
        background-color: #fbfdff;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 6px rgba(13, 38, 59, 0.06);
        border-right: 4px solid #0b66c3;
    }

    .chart-container {
        background-color: white;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 8px rgba(2,6,23,0.04);
        border: 1px solid #eef3f7;
    }

    .ar-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #0b3b66;
        margin-bottom: 0.75rem;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid #e6eef8;
        text-align: right;
    }

    /* Tabs tweaks */
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] { height: 48px; padding: 8px 18px; font-weight:600; }
    .stTabs [aria-selected="true"] { background-color: #0b66c3; color: white; }

    /* Small responsive tweaks */
    @media (max-width: 640px) {
        .ar-title { font-size: 1.1rem; }
        .chart-container { padding: 1rem; }
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess the tax receipts data."""
    try:
        # Try multiple possible locations for the data file
        possible_paths = [
            pathlib.Path("data/federal_tax_receipts_complete_analysis_sectors.csv"),
            pathlib.Path("federal_tax_receipts_complete_analysis_sectors.csv"),
            pathlib.Path("tax-revenue-app/data/federal_tax_receipts_complete_analysis_sectors.csv"),
            pathlib.Path("../data/federal_tax_receipts_complete_analysis_sectors.csv")
        ]
        
        df = None
        for file_path in possible_paths:
            if file_path.exists():
                df = pd.read_csv(file_path)
                break
        
        if df is None:
            # Create sample data if file not found
            st.warning("⚠️ ملف البيانات غير موجود. سيتم استخدام بيانات نموذجية لأغراض العرض.")
            
            # Create sample data (1947-2025 quarterly)
            dates = pd.date_range(start='1947-01-01', end='2025-12-31', freq='Q')
            np.random.seed(42)
            
            df = pd.DataFrame({
                'Date': dates,
                'Total Tax Receipts (B)': np.random.uniform(50, 5000, len(dates)).cumsum(),
                'Personal Income Tax (B)': np.random.uniform(30, 3000, len(dates)).cumsum(),
                'Corporate Income Tax (B)': np.random.uniform(10, 1000, len(dates)).cumsum(),
                'Production & Import Tax (B)': np.random.uniform(5, 500, len(dates)).cumsum(),
                'Taxes from ROW (B)': np.random.uniform(2, 200, len(dates)).cumsum(),
                'Other Taxes (B)': np.random.uniform(3, 300, len(dates)).cumsum()
            })
            
            # Calculate percentages
            df['Personal Income Tax (%)'] = (df['Personal Income Tax (B)'] / df['Total Tax Receipts (B)']) * 100
            df['Corporate Income Tax (%)'] = (df['Corporate Income Tax (B)'] / df['Total Tax Receipts (B)']) * 100
            df['Production & Import Tax (%)'] = (df['Production & Import Tax (B)'] / df['Total Tax Receipts (B)']) * 100
            df['Taxes from ROW (%)'] = (df['Taxes from ROW (B)'] / df['Total Tax Receipts (B)']) * 100
            df['Other Taxes (%)'] = (df['Other Taxes (B)'] / df['Total Tax Receipts (B)']) * 100
    
    except Exception as e:
        st.error(f"خطأ في تحميل البيانات: {e}")
        return None
    
    # Convert Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Extract year and quarter for easier analysis
    df['Year'] = df['Date'].dt.year
    df['Quarter'] = df['Date'].dt.quarter
    
    # Create a Year-Quarter column for display
    df['Year-Quarter'] = df['Year'].astype(str) + '-Q' + df['Quarter'].astype(str)
    
    return df

def main():
    # Title and description in Arabic
    st.markdown('<div class="rtl-text">', unsafe_allow_html=True)
    st.title(" تحليل الإيرادات الضريبية حسب القطاع")
    st.markdown("تحليل الإيرادات الضريبية الفيدرالية الأمريكية حسب مصادر الإيرادات الرئيسية (1947-2025)")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    
    if df is None:
        st.error("تعذر تحميل البيانات. يرجى التحقق من ملف البيانات.")
        return
    
    # Create tabs for different analyses
    tab1, tab2, tab5 = st.tabs([
        "📊 النظرة العامة", 
        "📈 الاتجاهات الزمنية", 
        "🔍 تحليل متعمق"
    ])
    
    with tab1:
        st.markdown('<div class="chart-container rtl-text">', unsafe_allow_html=True)
        st.markdown('<div class="ar-title">النظرة العامة للإيرادات الضريبية</div>', unsafe_allow_html=True)
        
        # Latest data metrics in Arabic
        latest_data = df.iloc[-1]
        
        # Reverse columns so rightmost metrics appear first visually (RTL)
        col1, col2, col3, col4 = st.columns(4)[::-1]
        
        with col1:
            st.metric(
                label="ضريبة الدخل الشخصي",
                value=f"${latest_data['Personal Income Tax (B)']:.1f}B",
                delta=f"{latest_data['Personal Income Tax (%)']:.1f}%"
            )
        
        with col2:
            st.metric(
                label="ضريبة دخل الشركات",
                value=f"${latest_data['Corporate Income Tax (B)']:.1f}B",
                delta=f"{latest_data['Corporate Income Tax (%)']:.1f}%"
            )
        
        with col3:
            st.metric(
                label="ضريبة الإنتاج والاستيراد",
                value=f"${latest_data['Production & Import Tax (B)']:.1f}B",
                delta=f"{latest_data['Production & Import Tax (%)']:.1f}%"
            )
        
        with col4:
            st.metric(
                label="ضرائب أخرى",
                value=f"${latest_data['Other Taxes (B)']:.1f}B",
                delta=f"{latest_data['Other Taxes (%)']:.1f}%"
            )
        
        st.markdown('<div class="chart-container rtl-text">', unsafe_allow_html=True)
        st.markdown('<div class="ar-title">اتجاهات القطاعات عبر الزمن</div>', unsafe_allow_html=True)
        
        # Create a multi-line chart for all tax components
        fig2 = go.Figure()
        
        # Add traces for each tax component (colors chosen to be distinct and readable)
        components = [
            ('Personal Income Tax (B)', 'ضريبة الدخل الشخصي', '#1f77b4'),
            ('Corporate Income Tax (B)', 'ضريبة دخل الشركات', '#ff7f0e'),
            ('Production & Import Tax (B)', 'ضريبة الإنتاج والاستيراد', '#2ca02c'),
            ('Other Taxes (B)', 'ضرائب أخرى', '#9467bd')
        ]
        
        for col, name, color in components:
            fig2.add_trace(go.Scatter(
                x=df['Date'],
                y=df[col],
                name=name,
                mode='lines',
                line=dict(color=color, width=2),
                hovertemplate=f'{name}: $%{{y:.1f}}B<br>التاريخ: %{{x|%Y-%m}}<extra></extra>'
            ))
        
        fig2.update_layout(
            title='مكونات الإيرادات الضريبية عبر الزمن',
            xaxis_title="السنة",
            yaxis_title="مليار دولار",
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=600,
            font=dict(size=14)
        )
        
        st.plotly_chart(fig2, use_container_width=True)

        # Recent data table
        st.markdown('<div class="ar-title">بيانات الإيرادات الضريبية الحديثة</div>', unsafe_allow_html=True)
        
        # Show last 20 quarters
        recent_data = df.tail(20)[['Year-Quarter', 'Total Tax Receipts (B)', 'Personal Income Tax (B)', 
                                  'Corporate Income Tax (B)', 'Production & Import Tax (B)', 
                                  'Other Taxes (B)']].round(2)
        
        # Rename columns for Arabic display
        recent_data.columns = ['الفترة', 'الإجمالي (مليار $)', 'ضريبة الدخل الشخصي (مليار $)', 
                              'ضريبة دخل الشركات (مليار $)', 'ضريبة الإنتاج والاستيراد (مليار $)', 
                              'ضرائب أخرى (مليار $)']
        
        st.dataframe(
            recent_data,
            hide_index=True,
            use_container_width=True
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:        
        # Growth rates
        st.markdown('<div class="ar-title">معدلات النمو السنوية</div>', unsafe_allow_html=True)
        
        # Calculate year-over-year growth for each component
        df_yoy = df.copy()
        components_calc = ['Total Tax Receipts (B)', 'Personal Income Tax (B)', 
                     'Corporate Income Tax (B)', 'Production & Import Tax (B)']
        
        for col in components_calc:
            df_yoy[f'{col}_YoY'] = df_yoy[col].pct_change(4) * 100  # 4 quarters = 1 year
        
        # Create YoY growth chart
        fig3 = go.Figure()
        
        for col in components_calc:
            fig3.add_trace(go.Scatter(
                x=df_yoy['Date'],
                y=df_yoy[f'{col}_YoY'],
                name=col.replace(' (B)', ''),
                mode='lines',
                hovertemplate='%{y:.1f}%<extra></extra>'
            ))
        
        fig3.update_layout(
            title='معدلات النمو السنوية حسب القطاع',
            xaxis_title="السنة",
            yaxis_title="معدل النمو السنوي (%)",
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=500,
            font=dict(size=14)
        )
        
        # Add horizontal line at 0%
        fig3.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        
        # Historical composition analysis
        st.markdown('<div class="ar-title">التحول التاريخي في التوزيع الضريبي</div>', unsafe_allow_html=True)
        
        # Select start and end years for comparison (keep selection UI LTR for clarity)
        colA, colB = st.columns(2)[::-1]
        
        with colA:
            start_year = st.selectbox(
                "سنة البداية",
                options=sorted(df['Year'].unique()),
                index=0,
                key="start_year"
            )
        
        with colB:
            end_year = st.selectbox(
                "سنة النهاية",
                options=sorted(df['Year'].unique()),
                index=len(df['Year'].unique())-1,
                key="end_year"
            )
        
        # Get data for selected years
        start_data = df[df['Year'] == start_year].iloc[0]
        end_data = df[df['Year'] == end_year].iloc[-1]
        
        # Create comparison chart
        comparison_data = pd.DataFrame({
            'القطاع': ['ضريبة الدخل الشخصي', 'ضريبة دخل الشركات', 'ضريبة الإنتاج والاستيراد', 'ضرائب أخرى'],
            f'{start_year}': [
                start_data['Personal Income Tax (%)'],
                start_data['Corporate Income Tax (%)'],
                start_data['Production & Import Tax (%)'],
                start_data['Other Taxes (%)']
            ],
            f'{end_year}': [
                end_data['Personal Income Tax (%)'],
                end_data['Corporate Income Tax (%)'],
                end_data['Production & Import Tax (%)'],
                end_data['Other Taxes (%)']
            ]
        })
        
        fig6 = go.Figure()
        
        fig6.add_trace(go.Bar(
            name=str(start_year),
            x=comparison_data['القطاع'],
            y=comparison_data[f'{start_year}'],
            marker_color='lightblue'
        ))
        
        fig6.add_trace(go.Bar(
            name=str(end_year),
            x=comparison_data['القطاع'],
            y=comparison_data[f'{end_year}'],
            marker_color='darkblue'
        ))
        
        fig6.update_layout(
            title=f'مقارنة التوزيع الضريبي: {start_year} مقابل {end_year}',
            xaxis_title="القطاع الضريبي",
            yaxis_title="النسبة المئوية (%)",
            barmode='group',
            height=500,
            font=dict(size=14)
        )
        
        st.plotly_chart(fig6, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab5:
        st.markdown('<div class="chart-container rtl-text">', unsafe_allow_html=True)        
        # Custom analysis
        st.markdown('<div class="ar-title">تحليل مخصص</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)[::-1]
        
        with col1:
            # Select tax component to analyze
            selected_component = st.selectbox(
                "اختر مكون الضريبة",
                options=['Personal Income Tax (B)', 'Corporate Income Tax (B)', 
                        'Production & Import Tax (B)', 'Other Taxes (B)'],
                index=0,
                key="selected_component"
            )
        
        with col2:
            # Select analysis type
            analysis_type = st.selectbox(
                "اختر نوع التحليل",
                options=["القيم الفعلية", "النسبة من الإجمالي", "النمو السنوي"],
                index=0,
                key="analysis_type"
            )
        
        # Create custom analysis chart
        fig10 = go.Figure()
        
        if analysis_type == "القيم الفعلية":
            y_data = df[selected_component]
            y_title = "مليار دولار"
        elif analysis_type == "النسبة من الإجمالي":
            # Get the percentage column name
            perc_col = selected_component.replace(' (B)', ' (%)')
            y_data = df[perc_col]
            y_title = "النسبة من الإجمالي (%)"
        else:  # Year-over-Year Growth
            # Calculate YoY growth
            y_data = df[selected_component].pct_change(4) * 100
            y_title = "معدل النمو السنوي (%)"
        
        fig10.add_trace(go.Scatter(
            x=df['Date'],
            y=y_data,
            mode='lines',
            line=dict(width=2),
            name=selected_component.replace(' (B)', '')
        ))
        
        fig10.update_layout(
            title=f'{analysis_type} لـ {selected_component.replace(" (B)", "")}',
            xaxis_title="السنة",
            yaxis_title=y_title,
            hovermode='x unified',
            height=500,
            font=dict(size=14)
        )
        
        if analysis_type == "النمو السنوي":
            fig10.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        st.plotly_chart(fig10, use_container_width=True)

        # Create pivot table for seasonal analysis
        df['Quarter'] = df['Date'].dt.quarter
        seasonal_data = df.pivot_table(
            values='Total Tax Receipts (B)',
            index='Quarter',
            columns='Year',
            aggfunc='mean'
        )
        
        # Calculate average by quarter
        seasonal_data['المتوسط'] = seasonal_data.mean(axis=1)
        
        fig7 = go.Figure()
        
        # Add average line
        fig7.add_trace(go.Scatter(
            x=['الربع الأول', 'الربع الثاني', 'الربع الثالث', 'الربع الرابع'],
            y=seasonal_data['المتوسط'],
            name='المتوسط',
            mode='lines+markers',
            line=dict(color='black', width=3),
            marker=dict(size=10)
        ))
        
        # Add a few sample years
        sample_years = [df['Year'].min(), 1980, 2000, df['Year'].max()]
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightsalmon']
        
        for year, color in zip(sample_years, colors):
            if year in seasonal_data.columns:
                fig7.add_trace(go.Scatter(
                    x=['الربع الأول', 'الربع الثاني', 'الربع الثالث', 'الربع الرابع'],
                    y=seasonal_data[year],
                    name=str(year),
                    mode='lines',
                    line=dict(color=color, width=1, dash='dash'),
                    opacity=0.7
                ))
        
        st.markdown('<div class="ar-title">النمو الربعي</div>', unsafe_allow_html=True)
        
        # Calculate QoQ growth
        df_qoq = df.copy()
        df_qoq['Total_QoQ'] = df_qoq['Total Tax Receipts (B)'].pct_change() * 100
        
        # Create QoQ growth chart
        fig8 = go.Figure()
        
        fig8.add_trace(go.Bar(
            x=df_qoq['Date'],
            y=df_qoq['Total_QoQ'],
            name='النمو الربعي',
            marker_color=df_qoq['Total_QoQ'].apply(lambda x: 'green' if x > 0 else 'red')
        ))
        
        fig8.update_layout(
            title='النمو الربعي للإيرادات الضريبية الإجمالية',
            xaxis_title="السنة",
            yaxis_title="معدل النمو (%)",
            hovermode='x unified',
            height=500,
            font=dict(size=14)
        )
        
        # Add horizontal line at 0%
        fig8.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        st.plotly_chart(fig8, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Footer in Arabic
    st.markdown("---")
    st.markdown(
        f"""
        <div style='text-align: center; color: gray; font-size: 0.9em; direction: rtl; font-family: Cairo;'>
        <p>لوحة تحليل الإيرادات الضريبية الفيدرالية | بيانات من 1947 إلى 2025</p>
        <p>ملاحظة: جميع القيم بالمليار دولار. النسب المئوية تمثل الحصة من إجمالي الإيرادات الضريبية.</p>
        <p>آخر تحديث: {datetime.now().strftime('%Y-%m-%d')}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()