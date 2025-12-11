import pathlib
import pickle
import sys
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime

# Set page title
st.set_page_config(page_title="لوحة النظرة العامة", page_icon="📊")

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

    try:
        import sys
        import numpy as _np
        if "numpy._core.numeric" not in sys.modules:
            sys.modules["numpy._core.numeric"] = _np.core.numeric
    except Exception:
        pass

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

# Top KPIs
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="النموذج المستخدم",
        value="Lasso Regression",
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
        delta="Active"
    )

with col4:
    st.metric(
        label="حالات المحاكاة",
        value=f"{len(st.session_state.get('scenarios', []))}",
        delta="محفوظة"
    )

st.markdown("---")

# Model prediction section
st.subheader("إدخال متغيرات الاقتصاد الكلي")

col1, col2 = st.columns(2)

with col1:
    gdp = st.slider(
        "الناتج المحلي الإجمالي (GDP)",
        min_value=0.0,
        max_value=60000.0,
        value=30000.0,
        step=100.0,
        help="Gross Domestic Product",
    )

    exports = st.slider(
        "الصادرات",
        min_value=0.0,
        max_value=10000.0,
        value=3000.0,
        step=50.0,
        help="Exports value",
    )

with col2:
    cpi = st.slider(
        "مؤشر أسعار المستهلك (CPI)",
        min_value=0.0,
        max_value=500.0,
        value=250.0,
        step=1.0,
        help="Consumer Price Index",
    )

    imports = st.slider(
        "الواردات",
        min_value=0.0,
        max_value=10000.0,
        value=4000.0,
        step=50.0,
        help="Imports value",
    )

st.markdown("---")

scenario_name = st.text_input("اسم السيناريو (اختياري)", placeholder="مثال: نمو اقتصادي متسارع")

col_predict, col_save = st.columns([2, 1])

# Initialize session_state
if "scenarios" not in st.session_state:
    st.session_state.scenarios = []

# PREDICT BUTTON
with col_predict:
    if st.button("🔮 تنبؤ بالإيرادات", use_container_width=True):
        try:
            model = load_model()
            pred = predict_tax_revenue(model, gdp, cpi, exports, imports)

            st.success("تم التنبؤ بنجاح!")
            st.metric(
                label="الإيرادات الضريبية المتوقعة",
                value=f"{pred:,.2f}",
                delta_color="off"
            )

            # Store last prediction
            st.session_state.last_prediction = pred
            st.session_state.last_params = {
                "gdp": gdp,
                "cpi": cpi,
                "exports": exports,
                "imports": imports
            }

        except Exception as e:
            st.error(f"خطأ في التنبؤ: {e}")

# SAVE SCENARIO BUTTON
with col_save:
    if st.button("💾 حفظ السيناريو", use_container_width=True):
        if "last_prediction" not in st.session_state:
            st.warning("يرجى عمل تنبؤ أولاً")
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
            st.success(f"تم حفظ السيناريو **{name}**!")

# DISPLAY SAVED SCENARIOS
if st.session_state.scenarios:
    st.markdown("### 📁 السيناريوهات المحفوظة")
    
    df = pd.DataFrame(st.session_state.scenarios)
    
    # Show table
    st.dataframe(
        df.style.format({
            "الناتج المحلي": "{:,.2f}",
            "مؤشر الأسعار": "{:,.2f}",
            "الصادرات": "{:,.2f}",
            "الواردات": "{:,.2f}",
            "الإيرادات المتوقعة": "{:,.2f}",
        }),
        use_container_width=True,
        hide_index=True
    )
    
    # Chart
    if len(df) > 1:
        st.markdown("#### مقارنة بين السيناريوهات")
        chart_df = df[["الاسم", "الإيرادات المتوقعة"]].copy()
        chart_fig = px.bar(
            chart_df,
            x="الاسم",
            y="الإيرادات المتوقعة",
            color="الإيرادات المتوقعة",
            color_continuous_scale="Blues",
            labels={"الإيرادات المتوقعة": "الإيرادات المتوقعة", "الاسم": "السيناريو"}
        )
        chart_fig.update_layout(height=400)
        st.plotly_chart(chart_fig, use_container_width=True)
else:
    st.info("لا توجد سيناريوهات محفوظة بعد. قم بالتنبؤ ثم احفظ السيناريو.")

# Insights panel
st.markdown("### 📋 ملاحظات حول النموذج")
st.markdown("""
<div class="insight-box">
    <strong>كيفية الاستخدام:</strong>
    <ol dir="rtl">
        <li>اضبط شرائح المتغيرات الاقتصادية</li>
        <li>انقر على "تنبؤ بالإيرادات"</li>
        <li>احفظ السيناريو لمقارنته لاحقاً</li>
    </ol>
</div>
<div class="insight-box">
    <strong>تفسير النتائج:</strong> يعتمد النموذج على تحليل الانحدار الخطي مع تنظيم Lasso. القيم الأعلى للناتج المحلي والصادرات تؤدي عادةً إلى إيرادات ضريبية أعلى.
</div>
""", unsafe_allow_html=True)