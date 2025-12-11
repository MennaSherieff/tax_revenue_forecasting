import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import pathlib

# Set page configuration
st.set_page_config(
    page_title="تحليل الإيرادات الضريبية حسب القطاع",
    page_icon="💰",
    layout="wide"
)

# Custom CSS for Arabic support and styling
st.markdown("""
    <style>
    /* Arabic font and RTL support */
    * {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .rtl-text {
        direction: rtl;
        text-align: right;
    }
    
    /* Card styling */
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-right: 4px solid #0068c9;
    }
    
    /* Chart containers */
    .chart-container {
        background-color: white;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border: 1px solid #e9ecef;
    }
    
    /* Arabic titles */
    .ar-title {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1a365d;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e2e8f0;
    }
    
    /* Custom tabs styling for Arabic */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 10px 20px;
        font-weight: 500;
        background-color: #f1f5f9;
        border-radius: 5px 5px 0 0;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1e40af;
        color: white;
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
    st.title("💰 تحليل الإيرادات الضريبية حسب القطاع")
    st.markdown("تحليل الإيرادات الضريبية الفيدرالية الأمريكية حسب مصادر الإيرادات الرئيسية (1947-2025)")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    
    if df is None:
        st.error("تعذر تحميل البيانات. يرجى التحقق من ملف البيانات.")
        return
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 النظرة العامة", 
        "📈 الاتجاهات الزمنية", 
        "📋 توزيع القطاعات", 
        "📅 التحليل الربعي", 
        "🔍 تحليل متعمق"
    ])
    
    with tab1:
        st.markdown('<div class="chart-container rtl-text">', unsafe_allow_html=True)
        st.markdown('<div class="ar-title">النظرة العامة للإيرادات الضريبية</div>', unsafe_allow_html=True)
        
        # Latest data metrics in Arabic
        latest_data = df.iloc[-1]
        
        col2, col3, col4, col5 = st.columns(4)
        
        
        
        with col2:
            st.metric(
                label="ضريبة الدخل الشخصي",
                value=f"${latest_data['Personal Income Tax (B)']:.1f}B",
                delta=f"{latest_data['Personal Income Tax (%)']:.1f}%"
            )
        
        with col3:
            st.metric(
                label="ضريبة دخل الشركات",
                value=f"${latest_data['Corporate Income Tax (B)']:.1f}B",
                delta=f"{latest_data['Corporate Income Tax (%)']:.1f}%"
            )
        
        with col4:
            st.metric(
                label="ضريبة الإنتاج والاستيراد",
                value=f"${latest_data['Production & Import Tax (B)']:.1f}B",
                delta=f"{latest_data['Production & Import Tax (%)']:.1f}%"
            )
        
        with col5:
            st.metric(
                label="ضرائب أخرى",
                value=f"${latest_data['Other Taxes (B)']:.1f}B",
                delta=f"{latest_data['Other Taxes (%)']:.1f}%"
            )
        
        # Time series of total tax receipts
        st.markdown('<div class="ar-title">إجمالي الإيرادات الضريبية عبر الزمن</div>', unsafe_allow_html=True)
        
        fig1 = px.line(
            df, 
            x='Date', 
            y='Total Tax Receipts (B)',
            title='الإيرادات الضريبية الفيدرالية (1947-2025)',
            labels={'Total Tax Receipts (B)': 'الإيرادات الضريبية (مليار دولار)', 'Date': 'السنة'}
        )
        fig1.update_layout(
            hovermode='x unified',
            xaxis_title="السنة",
            yaxis_title="مليار دولار",
            height=500,
            font=dict(size=14)
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        # Recent data table
        st.markdown('<div class="ar-title">بيانات الإيرادات الضريبية الحديثة</div>', unsafe_allow_html=True)
        
        # Show last 10 quarters
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
        st.markdown('<div class="chart-container rtl-text">', unsafe_allow_html=True)
        st.markdown('<div class="ar-title">اتجاهات القطاعات عبر الزمن</div>', unsafe_allow_html=True)
        
        # Create a multi-line chart for all tax components
        fig2 = go.Figure()
        
        # Add traces for each tax component
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
        
        # Growth rates
        st.markdown('<div class="ar-title">معدلات النمو السنوية</div>', unsafe_allow_html=True)
        
        # Calculate year-over-year growth for each component
        df_yoy = df.copy()
        components = ['Total Tax Receipts (B)', 'Personal Income Tax (B)', 
                     'Corporate Income Tax (B)', 'Production & Import Tax (B)']
        
        # Calculate YoY growth
        for col in components:
            df_yoy[f'{col}_YoY'] = df_yoy[col].pct_change(4) * 100  # 4 quarters = 1 year
        
        # Create YoY growth chart
        fig3 = go.Figure()
        
        for col in components:
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
    
    with tab3:
        st.markdown('<div class="chart-container rtl-text">', unsafe_allow_html=True)
        st.markdown('<div class="ar-title">تحليل توزيع القطاعات</div>', unsafe_allow_html=True)
        
        # Create two columns for composition charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="ar-title">توزيع الربع الأخير</div>', unsafe_allow_html=True)
            
            # Get percentage columns for latest data
            percentage_cols = ['Personal Income Tax (%)', 'Corporate Income Tax (%)', 
                              'Production & Import Tax (%)', 'Other Taxes (%)']
            latest_percentages = latest_data[percentage_cols]
            
            # Create pie chart
            labels = ['ضريبة الدخل الشخصي', 'ضريبة دخل الشركات', 'ضريبة الإنتاج والاستيراد', 'ضرائب أخرى']
            values = latest_percentages.values
            
            fig4 = px.pie(
                values=values,
                names=labels,
                title=f'توزيع الإيرادات الضريبية ({latest_data["Year-Quarter"]})',
                color_discrete_sequence=px.colors.sequential.RdBu
            )
            fig4.update_traces(textposition='inside', textinfo='percent+label')
            fig4.update_layout(font=dict(size=14))
            st.plotly_chart(fig4, use_container_width=True)
        
        with col2:
            st.markdown('<div class="ar-title">التوزيع عبر الزمن</div>', unsafe_allow_html=True)
            
            # Prepare data for stacked area chart
            fig5 = go.Figure()
            
            percentage_columns = [
                ('Personal Income Tax (%)', 'ضريبة الدخل الشخصي'),
                ('Corporate Income Tax (%)', 'ضريبة دخل الشركات'),
                ('Production & Import Tax (%)', 'ضريبة الإنتاج والاستيراد'),
                ('Other Taxes (%)', 'ضرائب أخرى')
            ]
            
            for col, name in percentage_columns:
                fig5.add_trace(go.Scatter(
                    x=df['Date'],
                    y=df[col],
                    name=name,
                    mode='lines',
                    stackgroup='one',
                    hovertemplate=f'{name}: %{{y:.1f}}%<extra></extra>'
                ))
            
            fig5.update_layout(
                title='توزيع الإيرادات الضريبية عبر الزمن',
                xaxis_title="السنة",
                yaxis_title="النسبة المئوية (%)",
                hovermode='x unified',
                height=500,
                font=dict(size=14)
            )
            
            st.plotly_chart(fig5, use_container_width=True)
        
        # Historical composition analysis
        st.markdown('<div class="ar-title">التحول التاريخي في التوزيع الضريبي</div>', unsafe_allow_html=True)
        
        # Select start and end years for comparison
        col1, col2 = st.columns(2)
        
        with col1:
            start_year = st.selectbox(
                "سنة البداية",
                options=sorted(df['Year'].unique()),
                index=0,
                key="start_year"
            )
        
        with col2:
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
    
    with tab4:
        
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
        
        # Quarter-over-quarter growth
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
    
    with tab5:
        st.markdown('<div class="chart-container rtl-text">', unsafe_allow_html=True)
        st.markdown('<div class="ar-title">تحليل متعمق</div>', unsafe_allow_html=True)
        
        # Correlation analysis
        st.markdown('<div class="ar-title">الارتباط بين مكونات الضرائب</div>', unsafe_allow_html=True)
        
        # Select tax components for correlation
        tax_components = [
            'Personal Income Tax (B)',
            'Corporate Income Tax (B)', 
            'Production & Import Tax (B)',
            'Other Taxes (B)'
        ]
        
        # Calculate correlation matrix
        corr_matrix = df[tax_components].corr()
        
        # Create heatmap
        fig9 = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale='RdBu',
            title='الارتباط بين مكونات الضرائب',
            labels=dict(color="معامل الارتباط")
        )
        
        # Update layout
        fig9.update_xaxes(ticktext=['ضريبة الدخل الشخصي', 'ضريبة دخل الشركات', 'ضريبة الإنتاج والاستيراد', 'ضرائب أخرى'],
                         tickvals=list(range(len(tax_components))))
        fig9.update_yaxes(ticktext=['ضريبة الدخل الشخصي', 'ضريبة دخل الشركات', 'ضريبة الإنتاج والاستيراد', 'ضرائب أخرى'],
                         tickvals=list(range(len(tax_components))))
        fig9.update_layout(font=dict(size=14))
        
        st.plotly_chart(fig9, use_container_width=True)
        
        # Custom analysis
        st.markdown('<div class="ar-title">تحليل مخصص</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
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
        
        # Summary statistics
        st.markdown('<div class="ar-title">الإحصائيات الملخصة</div>', unsafe_allow_html=True)
        
        # Calculate statistics for each component
        stats_components = ['Personal Income Tax (B)', 'Corporate Income Tax (B)', 
                           'Production & Import Tax (B)', 'Other Taxes (B)']
        
        stats_data = []
        for component in stats_components:
            stats_data.append({
                'المكون': component.replace(' (B)', ''),
                'المتوسط (مليار $)': df[component].mean(),
                'الوسيط (مليار $)': df[component].median(),
                'الانحراف المعياري (مليار $)': df[component].std(),
                'الحد الأدنى (مليار $)': df[component].min(),
                'الحد الأقصى (مليار $)': df[component].max(),
                'النمو 1947-2025 (%)': ((df[component].iloc[-1] - df[component].iloc[0]) / df[component].iloc[0]) * 100
            })
        
        stats_df = pd.DataFrame(stats_data)
        
        # Display statistics
        st.dataframe(
            stats_df.round(2),
            hide_index=True,
            use_container_width=True
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer in Arabic
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray; font-size: 0.9em; direction: rtl;'>
        <p>لوحة تحليل الإيرادات الضريبية الفيدرالية | بيانات من 1947 إلى 2025</p>
        <p>ملاحظة: جميع القيم بالمليار دولار. النسب المئوية تمثل الحصة من إجمالي الإيرادات الضريبية.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()