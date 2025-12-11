import pathlib
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
import plotly.express as px

# ---------- CONFIG ----------
st.set_page_config(page_title="لوحة النظرة العامة", page_icon="📊", layout="wide")

# Inject RTL styles, Cairo font, keep sliders LTR, and light risk/recommendation styles
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;600;700&display=swap');
    [data-testid="stAppViewContainer"], [data-testid="stSidebar"], .block-container, body {
        direction: rtl;
        unicode-bidi: embed;
        text-align: right;
        font-family: 'Cairo', system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif;
    }
    .stDataFrame table, .stDataFrame th, .stDataFrame td { direction: ltr; font-family: 'Cairo', sans-serif; }
    .stTitle, .stMarkdown, h1, h2, h3, h4, h5, h6 { text-align: right; font-family: 'Cairo', sans-serif; }
    [data-testid="stSlider"] { direction: ltr; }

    /* Light risk / recommendation styles */
    .risk-high { background: linear-gradient(135deg,#fff7f7 0%,#fff2f2 100%); color:#3b2b2b; padding:14px; border-radius:10px; border-right:6px solid #f3a6a6; }
    .risk-medium { background: linear-gradient(135deg,#fffdf6 0%,#fff9f0 100%); color:#3b3b2b; padding:14px; border-radius:10px; border-right:6px solid #ffd59a; }
    .risk-low { background: linear-gradient(135deg,#f6fffb 0%,#eefef5 100%); color:#234034; padding:14px; border-radius:10px; border-right:6px solid #bfead0; }

    .recommendation-card { background:#fbfcfd; border-radius:10px; padding:12px; border:1px solid #eef3f7; }
    .recommendation-highlight { background: linear-gradient(135deg,#eef8ff 0%,#e6f4ff 100%); color:#0b3b66; padding:14px; border-radius:10px; border-left:4px solid #bcdff6; }
    .recommendation-warning { background: linear-gradient(135deg,#fff7ec 0%,#fff3e6 100%); color:#4a3520; padding:12px; border-radius:10px; border-left:4px solid #ffd5a8; }

    .risk-header { display:flex; align-items:center; gap:10px; }
    .risk-icon { font-size:22px; }
    .risk-title { font-weight:700; margin:0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- UTILS ----------
@st.cache_resource
def load_model():
    models_dir = pathlib.Path("models")
    model_path = models_dir / "lasso.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}. Place lasso.pkl in the 'models/' folder.")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

def predict_tax_revenue(model, gdp, cpi, exports, imports):
    X = np.array([[gdp, cpi, exports, imports]], dtype=float)
    y_pred = model.predict(X)
    return float(y_pred[0])

# ---------- STATE ----------
if "scenarios" not in st.session_state:
    st.session_state.scenarios = []

# ---------- MAIN PAGE ----------
st.title(" لوحة النظرة العامة")
st.markdown("### محاكاة إيرادات الضرائب باستخدام النموذج الإحصائي")
st.markdown("---")

# Top KPIs
col2, col3, col4 = st.columns(3)[::-1]

with col2:
    st.metric(label="عدد المتغيرات", value="4", delta="GDP, CPI, Exports, Imports")
with col3:
    st.metric(label="آخر تحديث", value="يناير 2025", delta="نشط")
with col4:
    st.metric(label="السيناريوهات", value=f"{len(st.session_state.scenarios)}", delta="محفوظة")

st.markdown("---")

# Input sliders (sliders remain LTR)
st.subheader(" إدخال متغيرات الاقتصاد الكلي")
col1, col2 = st.columns(2)[::-1]

with col1:
    st.markdown("#### المتغيرات اليمنى")
    gdp = st.slider("الناتج المحلي الإجمالي (GDP)", min_value=10000, max_value=100000, value=35000, step=500)
    exports = st.slider("الصادرات", min_value=1000, max_value=15000, value=5000, step=100)

with col2:
    st.markdown("#### المتغيرات اليسرى")
    cpi = st.slider("مؤشر أسعار المستهلك (CPI)", min_value=50, max_value=400, value=200, step=5)
    imports = st.slider("الواردات", min_value=1000, max_value=15000, value=5500, step=100)

st.markdown("---")

# Scenario controls
st.subheader(" حفظ وتنبؤ")
scenario_name = st.text_input("اسم السيناريو (اختياري)", placeholder="مثال: نمو اقتصادي متسارع")

col_predict, col_save, col_clear = st.columns([1.5, 1.5, 1])[::-1]

with col_predict:
    if st.button(" تنبؤ بالإيرادات", use_container_width=True):
        try:
            model = load_model()
            pred = predict_tax_revenue(model, gdp, cpi, exports, imports)
            st.success("✅ تم التنبؤ بنجاح!")
            st.metric(label="الإيرادات الضريبية المتوقعة", value=f"${pred:,.0f}B", delta="السيناريو الحالي")
            st.session_state.last_prediction = pred
            st.session_state.last_params = {"gdp": gdp, "cpi": cpi, "exports": exports, "imports": imports}
        except Exception as e:
            st.error(f"❌ خطأ في التنبؤ: {e}")

with col_save:
    if st.button(" حفظ السيناريو", use_container_width=True):
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
            st.experimental_rerun()

with col_clear:
    if st.button("🗑️ حذف الكل", use_container_width=True):
        st.session_state.scenarios = []
        st.success("✅ تم حذف جميع السيناريوهات")
        st.experimental_rerun()

st.markdown("---")

# DISPLAY SAVED SCENARIOS & CHARTS
if st.session_state.scenarios:
    st.markdown("###  السيناريوهات المحفوظة")
    df = pd.DataFrame(st.session_state.scenarios)
    styled = df.style.set_table_attributes('dir="rtl"').format({
        "الناتج المحلي": "${:,.0f}B",
        "مؤشر الأسعار": "{:,.0f}",
        "الصادرات": "${:,.0f}B",
        "الواردات": "${:,.0f}B",
        "الإيرادات المتوقعة": "${:,.0f}B",
    })
    st.dataframe(styled, use_container_width=True, hide_index=True, height=300)

    st.markdown("---")

    # Charts area
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.markdown("####  مقارنة الإيرادات بين السيناريوهات")
        chart_df = df[["الاسم", "الإيرادات المتوقعة"]].copy()
        chart_fig = px.bar(chart_df, x="الاسم", y="الإيرادات المتوقعة",
                           color_discrete_sequence=["#2ecc71"],
                           labels={"الإيرادات المتوقعة": "الإيرادات ($B)", "الاسم": "السيناريو"})
        chart_fig.update_traces(texttemplate='$%{y:,.0f}B', textposition='outside')
        chart_fig.update_layout(height=400, showlegend=False, xaxis_title="", yaxis_title="الإيرادات ($B)")
        st.plotly_chart(chart_fig, use_container_width=True)

    with col_chart2:
        st.markdown("####  توزيع الإيرادات حسب السيناريو")
        pie_df = df[["الاسم", "الإيرادات المتوقعة"]].copy()
        pie_fig = px.pie(pie_df, values="الإيرادات المتوقعة", names="الاسم",
                         color_discrete_sequence=px.colors.sequential.Blues)
        pie_fig.update_layout(height=400)
        st.plotly_chart(pie_fig, use_container_width=True)

    st.markdown("---")

    st.markdown("####  اتجاه الإيرادات عبر السيناريوهات")
    trend_df = df[["الاسم", "الإيرادات المتوقعة"]].copy()
    trend_fig = px.line(trend_df, x="الاسم", y="الإيرادات المتوقعة", markers=True,
                        color_discrete_sequence=["#27ae60"],
                        labels={"الإيرادات المتوقعة": "الإيرادات ($B)", "الاسم": "السيناريو"})
    trend_fig.update_traces(marker=dict(size=8), line=dict(width=3))
    trend_fig.update_layout(height=350, hovermode='x unified', xaxis_title="", yaxis_title="الإيرادات ($B)")
    st.plotly_chart(trend_fig, use_container_width=True)

else:
    st.info("📭 لا توجد سيناريوهات محفوظة بعد. قم بالتنبؤ ثم احفظ السيناريو.")

st.markdown("---")

# RECOMMENDATIONS & RISK ASSESSMENT (alert banner + details)
st.markdown("## 🎯 التوصيات وتحليل المخاطر")

if st.session_state.scenarios:
    df = pd.DataFrame(st.session_state.scenarios)
    avg_revenue = df["الإيرادات المتوقعة"].mean() if not df.empty else 0.0
    max_revenue = df["الإيرادات المتوقعة"].max() if not df.empty else 0.0
    min_revenue = df["الإيرادات المتوقعة"].min() if not df.empty else 0.0
    revenue_std = df["الإيرادات المتوقعة"].std() if not df.empty else 0.0
    revenue_count = len(df)

    volatility_ratio = 0.0
    if avg_revenue and not np.isnan(avg_revenue):
        volatility_ratio = float((revenue_std / abs(avg_revenue)) * 100)

    if volatility_ratio > 20:
        cls = "risk-high"; icon = "🔴"; title = "مخاطر عالية"; summary = f"التقلب مرتفع ({volatility_ratio:.1f}%). يوصى بإجراءات تحوطية."
    elif volatility_ratio > 10:
        cls = "risk-medium"; icon = "🟡"; title = "مخاطر متوسطة"; summary = f"التقلب متوسط ({volatility_ratio:.1f}%). راقب الأداء وحدث فروض السيناريو."
    else:
        cls = "risk-low"; icon = "🟢"; title = "مخاطر منخفضة"; summary = f"التقلب منخفض ({volatility_ratio:.1f}%). الإيرادات مستقرة نسبياً."

    banner_html = f"""
    <div class="{cls}">
      <div class="risk-header">
        <div class="risk-icon">{icon}</div>
        <div>
          <div class="risk-title">{title}</div>
          <div class="risk-content">{summary}</div>
        </div>
      </div>
    </div>
    """
    st.markdown(banner_html, unsafe_allow_html=True)

    with st.expander("تفاصيل التوصيات وتحليل المخاطر", expanded=False):
        col_rec, col_risk = st.columns(2)[::-1]
        with col_rec:
            st.markdown("### 💡 توصيات عملية")
            best_idx = df["الإيرادات المتوقعة"].idxmax()
            worst_idx = df["الإيرادات المتوقعة"].idxmin()
            best_name = df.loc[best_idx, "الاسم"]
            best_revenue = df.loc[best_idx, "الإيرادات المتوقعة"]
            worst_name = df.loc[worst_idx, "الاسم"]
            worst_revenue = df.loc[worst_idx, "الإيرادات المتوقعة"]

            st.markdown(f'<div class="recommendation-highlight"><strong>السيناريو الأمثل:</strong> {best_name} — <strong>${best_revenue:,.0f}B</strong></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="recommendation-warning"><strong>تحذير:</strong> السيناريو الأسوأ {worst_name} — <strong>${worst_revenue:,.0f}B</strong></div>', unsafe_allow_html=True)

            st.markdown("""
            - ركز على تحسين الناتج المحلي والدعم التصديري لرفع الإيرادات.
            - خفض التضخم سيدعم الإيرادات الحقيقية.
            - ضع خطط احتياطية للسيناريوهات الأدنى.
            """)

        with col_risk:
            st.markdown("### ⚙️ مؤشرات الخطر")
            st.metric(label="عدد السيناريوهات", value=f"{revenue_count}")
            st.metric(label="متوسط الإيرادات", value=f"${avg_revenue:,.0f}B")
            st.metric(label="الانحراف المعياري", value=f"${revenue_std:,.0f}B")
            st.markdown(f"**النطاق:** الحد الأقصى: ${max_revenue:,.0f}B — الحد الأدنى: ${min_revenue:,.0f}B — الفرق: ${max_revenue - min_revenue:,.0f}B")

            if volatility_ratio > 20:
                st.markdown("**إجراءات مقترحة:** إعداد احتياطي سيولة، مراجعة سيناريوهات الإنفاق، تفعيل أدوات التحوط.")
            elif volatility_ratio > 10:
                st.markdown("**إجراءات مقترحة:** مراقبة دورية وتحديث فروض السيناريو كل ربع سنة.")
            else:
                st.markdown("**إجراءات مقترحة:** متابعات دورية وتحسينات تدريجية.")

else:
    st.info("📭 لا توجد سيناريوهات محفوظة. قم بإنشاء سيناريوهات أولاً لرؤية التوصيات وتحليل المخاطر.")

st.markdown("---")
st.caption(f"🔄 آخر تحديث: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} — تم تحليل {len(st.session_state.get('scenarios', []))} سيناريو")