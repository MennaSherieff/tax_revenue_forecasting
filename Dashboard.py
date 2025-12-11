import pathlib
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Set page title
st.set_page_config(page_title="لوحة النظرة العامة", page_icon="📊", layout="wide")

# Inject RTL styles for the app (but keep sliders LTR)
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"], [data-testid="stSidebar"], .block-container {
        direction: rtl;
        unicode-bidi: embed;
        text-align: right;
    }
    .stDataFrame table, .stDataFrame th, .stDataFrame td {
        direction: ltr;
    }
    .stTitle, .stMarkdown, h1, h2, h3, h4, h5, h6 {
        text-align: right;
    }
    /* Keep sliders LTR (left-to-right) */
    [data-testid="stSlider"] {
        direction: ltr;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- UTILS ---------- #
@st.cache_resource
def load_model():
    """Load the trained Lasso model from disk."""
    models_dir = pathlib.Path("models")
    model_path = models_dir / "lasso.pkl"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. "
            "Make sure lasso.pkl is placed in the 'models/' folder."
        )

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

def predict_tax_revenue(model, gdp, cpi, exports, imports):
    """Run a single prediction."""
    X = np.array([[gdp, cpi, exports, imports]], dtype=float)
    y_pred = model.predict(X)
    return float(y_pred[0])

# ---------- MAIN PAGE ---------- #
st.title("📊 لوحة النظرة العامة")
st.markdown("### محاكاة إيرادات الضرائب باستخدام النموذج الإحصائي")

st.markdown("---")

# Top KPIs
col1, col2, col3, col4 = st.columns(4)[::-1]

with col1:
    st.metric(
        label="النموذج المستخدم",
        value="Lasso",
        delta="دقة عالية"
    )

with col2:
    st.metric(
        label="عدد المتغيرات",
        value="4",
        delta="GDP, CPI, Exports, Imports"
    )

with col3:
    st.metric(
        label="آخر تحديث",
        value="يناير 2025",
        delta="نشط"
    )

with col4:
    st.metric(
        label="السيناريوهات",
        value=f"{len(st.session_state.get('scenarios', []))}",
        delta="محفوظة"
    )

st.markdown("---")

# Model prediction section
st.subheader("🔧 إدخال متغيرات الاقتصاد الكلي")

col1, col2 = st.columns(2)[::-1]

with col1:
    st.markdown("#### المتغيرات اليمنى")
    gdp = st.slider(
        "الناتج المحلي الإجمالي (GDP)",
        min_value=10000,
        max_value=100000,
        value=35000,
        step=500
    )

    exports = st.slider(
        "الصادرات",
        min_value=1000,
        max_value=15000,
        value=5000,
        step=100
    )

with col2:
    st.markdown("#### المتغيرات اليسرى")
    cpi = st.slider(
        "مؤشر أسعار المستهلك (CPI)",
        min_value=50,
        max_value=400,
        value=200,
        step=5
    )

    imports = st.slider(
        "الواردات",
        min_value=1000,
        max_value=15000,
        value=5500,
        step=100
    )

st.markdown("---")

# Scenario name
st.subheader("💾 حفظ وتنبؤ")
scenario_name = st.text_input(
    "اسم السيناريو (اختياري)",
    placeholder="مثال: نمو اقتصادي متسارع"
)

# Initialize session_state
if "scenarios" not in st.session_state:
    st.session_state.scenarios = []

# Button row
col_predict, col_save, col_clear = st.columns([1.5, 1.5, 1])[::-1]

with col_predict:
    if st.button("🔮 تنبؤ بالإيرادات", use_container_width=True):
        try:
            model = load_model()
            pred = predict_tax_revenue(model, gdp, cpi, exports, imports)

            st.success("✅ تم التنبؤ بنجاح!")
            
            st.metric(
                label="الإيرادات الضريبية المتوقعة",
                value=f"${pred:,.0f}B",
                delta="السيناريو الحالي"
            )

            st.session_state.last_prediction = pred
            st.session_state.last_params = {
                "gdp": gdp,
                "cpi": cpi,
                "exports": exports,
                "imports": imports
            }

        except Exception as e:
            st.error(f"❌ خطأ في التنبؤ: {e}")

with col_save:
    if st.button("💾 حفظ السيناريو", use_container_width=True):
        if "last_prediction" not in st.session_state:
            st.warning("⚠️ يرجى عمل تنبؤ أولاً")
        else:
            name = scenario_name.strip() or f"سيناريو #{len(st.session_state.scenarios) + 1}"
            
            st.session_state.scenarios.append({
                "الاسم": name,
                "الناتج المحلي": gdp,
                "مؤشر الأسعار": cpi,
                "الصادرات": exports,
                "الواردات": imports,
                "الإيرادات المتوقعة": st.session_state.last_prediction,
                "التاريخ": datetime.now().strftime("%Y-%m-%d %H:%M")
            })
            st.success(f"✅ تم حفظ السيناريو **{name}**!")
            st.rerun()

with col_clear:
    if st.button("🗑️ حذف الكل", use_container_width=True):
        st.session_state.scenarios = []
        st.success("✅ تم حذف جميع السيناريوهات")
        st.rerun()

st.markdown("---")

# DISPLAY SAVED SCENARIOS
if st.session_state.scenarios:
    st.markdown("### 📊 السيناريوهات المحفوظة")
    
    df = pd.DataFrame(st.session_state.scenarios)
    
    styled = df.style.set_table_attributes('dir="rtl"').format({
            "الناتج المحلي": "${:,.0f}B",
            "مؤشر الأسعار": "{:,.0f}",
            "الصادرات": "${:,.0f}B",
            "الواردات": "${:,.0f}B",
            "الإيرادات المتوقعة": "${:,.0f}B",
        })
    
    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        height=300
    )
    
    st.markdown("---")
    
    # TWO CHARTS SIDE BY SIDE
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown("#### 📈 مقارنة الإيرادات بين السيناريوهات")
        chart_df = df[["الاسم", "الإيرادات المتوقعة"]].copy()
        chart_fig = px.bar(
            chart_df,
            x="الاسم",
            y="الإيرادات المتوقعة",
            color_discrete_sequence=["#2ecc71"],
            labels={"الإيرادات المتوقعة": "الإيرادات ($B)", "الاسم": "السيناريو"},
        )
        chart_fig.update_traces(texttemplate='$%{y:,.0f}B', textposition='outside')
        chart_fig.update_layout(height=400, showlegend=False, xaxis_title="", yaxis_title="الإيرادات ($B)")
        st.plotly_chart(chart_fig, use_container_width=True)
    
    with col_chart2:
        st.markdown("#### 💰 توزيع الإيرادات حسب السيناريو")
        pie_df = df[["الاسم", "الإيرادات المتوقعة"]].copy()
        pie_fig = px.pie(
            pie_df,
            values="الإيرادات المتوقعة",
            names="الاسم",
            color_discrete_sequence=px.colors.sequential.Greens
        )
        pie_fig.update_layout(height=400)
        st.plotly_chart(pie_fig, use_container_width=True)
    
    st.markdown("---")
    
    # REVENUE TREND LINE CHART
    st.markdown("#### 📊 اتجاه الإيرادات عبر السيناريوهات")
    trend_df = df[["الاسم", "الإيرادات المتوقعة"]].copy()
    trend_fig = px.line(
        trend_df,
        x="الاسم",
        y="الإيرادات المتوقعة",
        markers=True,
        color_discrete_sequence=["#27ae60"],
        labels={"الإيرادات المتوقعة": "الإيرادات ($B)", "الاسم": "السيناريو"}
    )
    trend_fig.update_traces(marker=dict(size=10), line=dict(width=3))
    trend_fig.update_layout(height=350, hovermode='x unified', xaxis_title="", yaxis_title="الإيرادات ($B)")
    st.plotly_chart(trend_fig, use_container_width=True)
    
else:
    st.info("📭 لا توجد سيناريوهات محفوظة بعد. قم بالتنبؤ ثم احفظ السيناريو.")

st.markdown("---")

# RECOMMENDATIONS & RISK ANALYSIS SECTION
st.markdown("## 🎯 التوصيات وتحليل المخاطر")

if st.session_state.scenarios:
    df = pd.DataFrame(st.session_state.scenarios)
    
    # Calculate statistics
    avg_revenue = df["الإيرادات المتوقعة"].mean()
    max_revenue = df["الإيرادات المتوقعة"].max()
    min_revenue = df["الإيرادات المتوقعة"].min()
    revenue_variance = df["الإيرادات المتوقعة"].var()
    revenue_std = df["الإيرادات المتوقعة"].std()
    
    # Create two columns for recommendations and risk analysis
    col_rec, col_risk = st.columns(2)[::-1]
    
    with col_rec:
        st.markdown("### 💡 التوصيات")
        
        # Find best and worst scenarios
        best_scenario_idx = df["الإيرادات المتوقعة"].idxmax()
        worst_scenario_idx = df["الإيرادات المتوقعة"].idxmin()
        
        best_name = df.loc[best_scenario_idx, "الاسم"]
        best_revenue = df.loc[best_scenario_idx, "الإيرادات المتوقعة"]
        
        worst_name = df.loc[worst_scenario_idx, "الاسم"]
        worst_revenue = df.loc[worst_scenario_idx, "الإيرادات المتوقعة"]
        
        st.success(f"✅ **أفضل سيناريو:** {best_name}")
        st.metric("", f"${best_revenue:,.0f}B", delta="الأعلى عائداً")
        
        st.info(f"⚠️ **السيناريو الأقل:** {worst_name}")
        st.metric("", f"${worst_revenue:,.0f}B", delta="الأقل عائداً")
        
        st.markdown(f"""
        **النصائح:**
        - تركيز على زيادة الناتج المحلي الإجمالي لتحسين الإيرادات
        - العمل على تقليل التضخم (مؤشر الأسعار)
        - تعزيز الصادرات لزيادة القاعدة الضريبية
        - مراقبة مستويات الواردات بعناية
        """)
    
    with col_risk:
        st.markdown("### ⚠️ تحليل المخاطر")
        
        col_risk1, col_risk2 = st.columns(2)
        
        with col_risk1:
            st.metric(
                label="متوسط الإيرادات",
                value=f"${avg_revenue:,.0f}B",
                delta=f"من {len(df)} سيناريو"
            )
        
        with col_risk2:
            st.metric(
                label="الانحراف المعياري",
                value=f"${revenue_std:,.0f}B",
                delta="مستوى التقلب"
            )
        
        # Risk assessment
        volatility_ratio = (revenue_std / avg_revenue) * 100
        
        if volatility_ratio > 20:
            st.error(f"🔴 **مخاطر عالية:** التقلب = {volatility_ratio:.1f}%")
            st.markdown("- هناك تباين كبير في الإيرادات المتوقعة بين السيناريوهات")
            st.markdown("- ينصح بوضع خطط احتياطية للسيناريوهات الأسوأ")
        elif volatility_ratio > 10:
            st.warning(f"🟡 **مخاطر متوسطة:** التقلب = {volatility_ratio:.1f}%")
            st.markdown("- مستوى تقلب معقول مع الحاجة للمراقبة")
        else:
            st.success(f"🟢 **مخاطر منخفضة:** التقلب = {volatility_ratio:.1f}%")
            st.markdown("- الإيرادات مستقرة نسبياً عبر السيناريوهات")
        
        st.markdown(f"""
        **النطاق:**
        - الحد الأقصى: ${max_revenue:,.0f}B
        - الحد الأدنى: ${min_revenue:,.0f}B
        - الفرق: ${max_revenue - min_revenue:,.0f}B
        """)

else:
    st.info("📭 لا توجد سيناريوهات محفوظة. قم بإنشاء سيناريوهات أولاً لرؤية التوصيات وتحليل المخاطر.")

st.markdown("---")

# Insights panel
st.markdown("### 📋 ملاحظات حول النموذج")
col1, col2 = st.columns(2)[::-1]

with col1:
    st.markdown("""
    **كيفية الاستخدام:**
    1. اضبط شرائح المتغيرات الاقتصادية
    2. انقر على "تنبؤ بالإيرادات"
    3. احفظ السيناريو لمقارنته لاحقاً
    4. اعرض المقارنات في الرسوم البيانية
    5. اطلع على التوصيات وتحليل المخاطر
    """)

with col2:
    st.markdown("""
    **تفسير النتائج:**
    
    يعتمد النموذج على تحليل الانحدار الخطي مع تنظيم **Lasso**. القيم الأعلى للناتج المحلي والصادرات تؤدي عادةً إلى إيرادات ضريبية أعلى، بينما التضخم قد يؤثر سلباً على الإيرادات الحقيقية.
    """)

st.markdown("---")
st.caption(f"آخر تحديث: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")