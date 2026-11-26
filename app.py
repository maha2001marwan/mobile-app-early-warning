"""
AppPulse - Early Warning System & Dynamic Competitor Intelligence for Mobile Apps
Live search -> scrape -> feature engineering -> model benchmark -> dashboard -> triage -> Gemini report
"""
import os
import re
from datetime import datetime
from html import escape

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from google_play_scraper import Sort, reviews, search
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------------
# Page + theme
# ----------------------------------------------------------------------------
st.set_page_config(page_title="AppPulse | تحليل التطبيقات والمنافسين", page_icon="📡", layout="wide")

ACCENT = "#0F9D7A"
COMP_COLORS = ["#7562C7", "#D96B4D", "#397DB8", "#C28A19"]

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;800&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Arabic:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Noto Sans Arabic', 'Inter', sans-serif; }
.stApp, .hero, .hero p, .hero h1, .hero-meta, .step, .step h4, .step p,
.stMarkdown, .stCaption, .stTabs, div[data-testid="stMetric"] {
        font-family: 'Noto Sans Arabic', 'Inter', sans-serif !important; }
:root { color-scheme: light; }
.stApp { direction: rtl; text-align: right; background: #f4f7fb; color: #233247; }
div[data-testid="stAppViewContainer"] { direction: rtl; }
.block-container { max-width: 1500px; padding-top: 2rem; padding-bottom: 3rem; }
section[data-testid="stSidebar"] { background: #ffffff; border-left: 1px solid #e1e8f0; }
section[data-testid="stSidebar"] > div { padding-top: 1.5rem; }
.stApp, .stApp p, .stApp label, .stApp [data-testid="stMarkdownContainer"],
.stApp [data-testid="stCaptionContainer"], .stApp [data-testid="stWidgetLabel"] { color: #334155; }
.stApp h1, .stApp h2, .stApp h3, .stApp h4 { color: #172b45; }
.stApp [data-testid="stMarkdownContainer"], .stApp [data-testid="stCaptionContainer"] { text-align: right; }
.stApp input, .stApp textarea, .stApp [data-baseweb="select"] > div { text-align: right; }
.stApp [data-testid="stAlert"] { direction: rtl; text-align: right; }
.stApp [data-testid="stExpander"], .stApp [data-testid="stForm"] {
        background: #ffffff; border: 1px solid #e0e8f0; border-radius: 16px; }
.stApp [data-testid="stExpander"] details summary { direction: rtl; }
.stApp [data-testid="stMetric"] { text-align: right; }
.stApp [data-testid="stMetricValue"] { font-size: 1.55rem; }
.stApp [data-testid="stMarkdownContainer"] h2 { margin-top: 1.6rem; margin-bottom: .35rem; }
.stApp [data-testid="stMarkdownContainer"] h3,
.stApp [data-testid="stMarkdownContainer"] h4 { margin-top: 1.2rem; margin-bottom: .45rem; }
.hero { position: relative; overflow: hidden; padding: 34px 38px; border-radius: 26px; margin: 4px 0 24px;
        background: linear-gradient(115deg, #ffffff, #eef5fa 58%, #e8f5f1);
        border: 1px solid #dce7ed; box-shadow: 0 14px 36px rgba(31,55,80,.08); }
.hero:after { content: ""; position: absolute; width: 230px; height: 230px; border-radius: 50%;
        left: -60px; top: -110px; background: rgba(15,157,122,.09); filter: blur(2px); }
.hero-kicker { display: inline-block; padding: 5px 12px; margin-bottom: 13px; border-radius: 999px;
        color: #08785e; background: #e4f5ef; border: 1px solid #c5e9dd;
        font-size: .82rem; font-weight: 700; }
.hero h1 { position: relative; z-index: 1; margin: 0; font-size: clamp(1.7rem, 3vw, 2.45rem);
        font-weight: 800; letter-spacing: -.5px; color: #172b45; }
.hero h1 span { color: #0f9d7a; }
.hero p { position: relative; z-index: 1; max-width: 850px; margin: 10px 0 0; color: #52657b; font-size: 1rem; line-height: 1.9; }
.hero-meta { position: relative; z-index: 1; display: flex; flex-wrap: wrap; gap: 8px; margin-top: 19px; }
.hero-meta span { padding: 6px 11px; border: 1px solid #dce7ed; border-radius: 999px;
        background: rgba(255,255,255,.8); color: #52657b; font-size: .78rem; }
.step { background: #ffffff; border: 1px solid #e0e8f0;
        border-radius: 18px; padding: 20px; min-height: 145px; box-shadow: 0 8px 22px rgba(31,55,80,.06); }
.step b { display: inline-grid; place-items: center; width: 34px; height: 34px; border-radius: 11px;
        color: #ffffff; background: #0f9d7a; font-size: 1rem; }
.step h4 { margin: 12px 0 5px; color: #172b45; font-size: 1rem; }
.step p { color: #5e7085; margin: 0; font-size: .88rem; line-height: 1.7; }
div[data-testid="stMetric"] { height: 100%; background: #ffffff;
        border: 1px solid #e0e8f0; padding: 17px 19px; border-radius: 18px; box-shadow: 0 8px 22px rgba(31,55,80,.06); }
div[data-testid="stMetricLabel"] { color: #5e7085; }
div[data-testid="stMetricValue"] { font-weight: 800; color: #172b45; }
div[data-testid="stMetricDelta"] { font-size: .8rem; }
.banner { padding: 16px 20px; border-radius: 16px; margin: 14px 0 22px; font-weight: 500; line-height: 1.8; }
.banner.good { background: #eaf7f1; border: 1px solid #b9e4d3; color: #17684f; }
.banner.bad { background: #fff0ef; border: 1px solid #f1c7c3; color: #9b3933; }
.summary-box { padding: 15px 20px; margin: 0 0 18px; border: 1px solid #dce9e5;
        border-right: 4px solid #0f9d7a; border-radius: 14px; background: #ffffff;
        box-shadow: 0 6px 18px rgba(31,55,80,.04); line-height: 1.85; }
.summary-box p { margin: 5px 0; color: #334155; }
.summary-box .summary-footnote { margin-top: 12px; padding-top: 10px; border-top: 1px solid #edf1f5;
        color: #64748b; font-size: .88rem; }
.risk { padding: 20px 22px; border-radius: 18px; font-size: 1rem; line-height: 1.9; }
.risk.low { background: #eaf7f1; border: 1px solid #b9e4d3; color: #17684f; }
.risk.med { background: #fff7e5; border: 1px solid #efd99e; color: #795b0b; }
.risk.high { background: #fff0ef; border: 1px solid #f1c7c3; color: #9b3933; }
button[kind="primary"] { background: #0f9d7a !important; color:#ffffff !important;
        font-weight: 800 !important; border: 0 !important; box-shadow: 0 6px 16px rgba(15,157,122,.18); }
button[kind="primary"]:hover { filter: brightness(1.08); transform: translateY(-1px); }
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid #dce5ee; direction: rtl; flex-wrap: wrap; }
.stTabs [data-baseweb="tab"] { min-height: 46px; padding: 0 14px; font-weight: 650; color: #5e7085;
        background: #ffffff; border: 1px solid #e0e8f0; border-radius: 12px 12px 0 0; }
.stTabs [aria-selected="true"] { color: #08785e !important; background: #eaf7f1; }
div[data-testid="stDataFrame"], div[data-testid="stTable"] { border: 1px solid #e0e8f0; border-radius: 14px; overflow: hidden; }
div[data-testid="stPlotlyChart"] { padding: 8px; border: 1px solid #e0e8f0; border-radius: 18px;
        background: #ffffff; }
@media (max-width: 700px) {
    .block-container { padding: 1rem .8rem 2rem; }
    .hero { padding: 25px 20px; border-radius: 20px; }
    .hero h1 { font-size: 1.65rem; }
    .stTabs [data-baseweb="tab"] { padding: 0 9px; font-size: .82rem; }
    .summary-box { padding: 12px 14px; }
    div[data-testid="stHorizontalBlock"] { flex-wrap: wrap; gap: .75rem; }
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
        min-width: calc(50% - .4rem) !important; flex: 1 1 calc(50% - .4rem);
    }
}
footer, #MainMenu { visibility: hidden; }
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
DEMO_APPS = {
    "Uber Eats": "com.ubercab.eats",
    "DoorDash": "com.dd.doordash",
    "Grubhub": "com.grubhub.android",
    "Deliveroo": "com.deliveroo.orderapp",
    "Just Eat": "com.justeat.app.uk",
}
COUNTRIES = {
    "السعودية": ("sa", "ar"),
    "الإمارات": ("ae", "ar"),
    "مصر": ("eg", "ar"),
    "الأردن": ("jo", "ar"),
    "الولايات المتحدة": ("us", "en"),
    "المملكة المتحدة": ("gb", "en"),
    "كندا": ("ca", "en"),
    "أستراليا": ("au", "en"),
    "ألمانيا": ("de", "en"),
    "فرنسا": ("fr", "en"),
}

CRASH_RX = r"crash|freez|froze|bug|glitch|error|broken|not working|doesn'?t work|won'?t (open|load|work)|stuck|fail|force clos|تعطل|تعطّل|انهيار|تجمّد|تجمد|خلل|خطأ|لا يعمل|لا تفتح|عالق|فشل"
ISSUES = {
    "Crashes & bugs": CRASH_RX,
    "Login & account": r"log ?in|sign ?in|password|account|verif|banned|locked|suspend|تسجيل الدخول|كلمة المرور|الحساب|تحقق|محظور|مقفل",
    "Payments & refunds": r"refund|charge|payment|credit card|money|price|fee|overcharg|promo|coupon|استرداد|استرجاع|دفع|بطاقة|رسوم|السعر|مبلغ|خصم|قسيمة",
    "Delivery & orders": r"deliver|driver|late|cold|missing|wrong order|cancel|never arrived|courier|توصيل|مندوب|متأخر|بارد|ناقص|طلب خاطئ|إلغاء|لم يصل",
    "Slow performance": r"slow|lag|loading|spinning|takes forever|hang|بطيء|بطء|تحميل|يستغرق|يعلق|تعليق",
}
FEATURES = ["text_len", "word_count", "exclam_count", "thumbs_up", "crash_kw_count", "caps_ratio", "hour"]
FEATURE_LABELS = {
    "text_len": "طول المراجعة", "word_count": "عدد الكلمات", "exclam_count": "علامات التعجب",
    "thumbs_up": "الإعجابات", "crash_kw_count": "كلمات الأعطال الحرجة",
    "caps_ratio": "نسبة الأحرف الكبيرة", "hour": "ساعة النشر",
}
ISSUE_LABELS_AR = {
    "Crashes & bugs": "الأعطال والأخطاء",
    "Login & account": "تسجيل الدخول والحساب",
    "Payments & refunds": "المدفوعات والاسترداد",
    "Delivery & orders": "التوصيل والطلبات",
    "Slow performance": "بطء الأداء",
}
TREND_COLUMN_LABELS_AR = {
    "App": "التطبيق",
    "Previous 7 days (reviews)": "مراجعات الأيام السبعة السابقة",
    "Previous critical (%)": "الشكاوى الحرجة سابقًا (%)",
    "Latest 7 days (reviews)": "مراجعات آخر 7 أيام",
    "Latest critical (%)": "الشكاوى الحرجة حاليًا (%)",
    "Change (percentage points)": "التغير (نقطة مئوية)",
    "Status": "الحالة",
}
MODEL_LABELS_AR = {
    "Logistic Regression": "الانحدار اللوجستي (Logistic Regression)",
    "Random Forest": "الغابة العشوائية (Random Forest)",
}


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def get_api_key():
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY", "") or st.session_state.get("manual_key", "")


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_reviews(app_id, n, lang, country):
    data, _ = reviews(app_id, lang=lang, country=country, sort=Sort.NEWEST, count=n)
    df = pd.DataFrame(data)
    if df.empty:
        return df
    return df[["reviewId", "content", "score", "thumbsUpCount", "at"]]


def clean_text(t):
    t = re.sub(r"http\S+", " ", str(t))
    return re.sub(r"\s+", " ", t).strip()


def normalize_for_matching(text):
    text = str(text).lower()
    text = re.sub(r"[\u064B-\u065F\u0670\u0640]", "", text)
    return text.translate(str.maketrans({
        "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",
        "ى": "ي", "ة": "ه",
    }))


def matches_issue(text, pattern):
    return bool(re.search(normalize_for_matching(pattern), normalize_for_matching(text)))


def text_features(text, thumbs=0, hour=12):
    t = clean_text(text)
    letters = [c for c in t if c.isalpha()]
    return {
        "text_len": len(t),
        "word_count": len(t.split()),
        "exclam_count": sum(t.count(mark) for mark in ("!", "！", "‼")),
        "thumbs_up": thumbs,
        "crash_kw_count": len(re.findall(normalize_for_matching(CRASH_RX), normalize_for_matching(t))),
        "caps_ratio": (sum(c.isupper() for c in letters) / len(letters)) if letters else 0.0,
        "hour": hour,
    }


def build_dataset(frames):
    df = pd.concat(frames, ignore_index=True).drop_duplicates("reviewId")
    df = df.dropna(subset=["content", "score"])
    df["content"] = df["content"].map(clean_text)
    df = df[df["content"].str.len() > 0].copy()
    df["clean"] = df["content"].str.lower()
    df["at"] = pd.to_datetime(df["at"])
    feats = pd.DataFrame(
        [text_features(t, th, h) for t, th, h in zip(df["content"], df["thumbsUpCount"], df["at"].dt.hour)],
        index=df.index,
    )
    df = pd.concat([df, feats], axis=1)
    df["critical"] = (df["score"] <= 2).astype(int)
    return df.rename(columns={"thumbsUpCount": "likes"}).reset_index(drop=True)


def train_models(df):
    y = df["critical"]
    if y.nunique() < 2 or y.value_counts().min() < 5:
        return None
    X = df[FEATURES]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1
        ),
    }
    rows, cms = [], {}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        p = m.predict(Xte)
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(yte, p),
            "Precision": precision_score(yte, p, zero_division=0),
            "Recall": recall_score(yte, p, zero_division=0),
        })
        cms[name] = confusion_matrix(yte, p, labels=[0, 1])
    table = pd.DataFrame(rows)
    best = table.sort_values(["Recall", "Precision"], ascending=False).iloc[0]["Model"]
    return {"models": models, "table": table, "cms": cms, "best": best, "n_test": len(yte)}


def feature_importance(res):
    m = res["models"][res["best"]]
    if hasattr(m, "feature_importances_"):
        imp = m.feature_importances_
    else:
        imp = np.abs(m[-1].coef_[0])
        imp = imp / imp.sum()
    return pd.DataFrame({"Feature": [FEATURE_LABELS[f] for f in FEATURES], "Importance": imp}).sort_values("Importance")


def app_summary(df):
    return (
        df.groupby("app_name")
        .agg(reviews=("score", "size"), avg_rating=("score", "mean"),
             critical_reviews=("critical", "sum"),
             critical_pct=("critical", lambda x: 100 * x.mean()), avg_likes=("likes", "mean"))
        .reset_index()
    )


def issue_table(df):
    crit = df[df["critical"] == 1]
    rows = []
    for app in df["app_name"].drop_duplicates():
        group = crit[crit["app_name"] == app]
        for issue, rx in ISSUES.items():
            share = 100 * group["content"].map(lambda text: matches_issue(text, rx)).mean() if not group.empty else 0.0
            rows.append({"app_name": app, "Issue": issue, "Share of critical reviews (%)": share})
    return pd.DataFrame(rows)


def complaint_trends(df, min_reviews=5, alert_threshold=10):
    weekly = (
        df.assign(week=df["at"].dt.strftime("%G-W%V"))
        .groupby(["week", "app_name"])
        .agg(
            reviews=("critical", "size"),
            critical_pct=("critical", lambda values: 100 * values.mean()),
        )
        .reset_index()
        .sort_values("week")
    )

    latest = df["at"].max()
    recent_start = latest - pd.Timedelta(days=7)
    previous_start = recent_start - pd.Timedelta(days=7)
    rows = []
    for app, group in df.groupby("app_name"):
        recent = group[(group["at"] > recent_start) & (group["at"] <= latest)]
        previous = group[(group["at"] > previous_start) & (group["at"] <= recent_start)]
        recent_pct = 100 * recent["critical"].mean() if not recent.empty else np.nan
        previous_pct = 100 * previous["critical"].mean() if not previous.empty else np.nan
        enough_data = len(recent) >= min_reviews and len(previous) >= min_reviews
        change = recent_pct - previous_pct if enough_data else np.nan
        if not enough_data:
            status = f"Insufficient data (need {min_reviews}+ reviews per period)"
        elif change >= alert_threshold:
            status = "Alert: complaints rising"
        elif change > 0:
            status = "Slight increase"
        else:
            status = "Stable or decreasing"
        rows.append({
            "App": app,
            "Previous 7 days (reviews)": len(previous),
            "Previous critical (%)": previous_pct,
            "Latest 7 days (reviews)": len(recent),
            "Latest critical (%)": recent_pct,
            "Change (percentage points)": change,
            "Status": status,
        })
    return weekly, pd.DataFrame(rows)


def style(fig, h=380):
    fig.update_layout(
        template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=h, margin=dict(l=10, r=10, t=50, b=10), font=dict(family="Noto Sans Arabic, Inter", color="#334155"),
        legend_title_text="",
    )
    fig.update_xaxes(gridcolor="#e7edf3", zerolinecolor="#cbd5e1")
    fig.update_yaxes(gridcolor="#e7edf3", zerolinecolor="#cbd5e1")
    return fig


def render_user_guide():
    with st.expander("📘 دليل الاستخدام وشرح النتائج — افتحي هذا القسم عند الحاجة", expanded=False):
        st.markdown(
            """
### ما وظيفة هذا المشروع؟
يجمع AppPulse أحدث مراجعات التطبيقات من **Google Play** للبلد الذي تختارينه، ثم يلخص تقييمات المستخدمين ويقارن تطبيقك بمنافسيه. كما يدرّب نموذجين آليين لتمييز المراجعات الحرجة، ويعرض مؤشرات تساعدك على تحديد ما يستحق المتابعة. النتائج مؤشرات مساعدة وليست حكمًا نهائيًا أو بديلًا عن قراءة المراجعات الأصلية.

### ابدئي التحليل خطوة بخطوة
1. من **الشريط الجانبي** اختاري بلد متجر Google Play. اختيار السعودية أو الإمارات أو مصر أو الأردن يطلب المراجعات بالعربية؛ البلدان الأخرى تستخدم الإنجليزية.
2. اختاري **تطبيقنا** من القائمة. لإضافة تطبيق غير موجود، اكتبي اسمه في خانة البحث واضغطي **بحث**، ثم اختاريه من قائمة التطبيقات بعد إضافته.
3. اختاري من 2 إلى 4 منافسين، وحددي عدد المراجعات المطلوب جلبها **لكل تطبيق**. العدد الأكبر قد يستغرق وقتًا أطول، وقد لا تتوفر مراجعات كافية لبعض التطبيقات أو البلدان.
4. اضغطي **ابدأ التحليل** وانتظري اكتمال جلب المراجعات وتجهيز المقارنة. إذا ظهرت رسالة تعذر الجلب، راجعي اسم التطبيق والبلد أو جرّبي تطبيقًا آخر.
5. اقرئي **الخلاصة التنفيذية** الظاهرة أعلى الصفحة أولًا، ثم افتحي القسم المتخصص من التبويبات عند الحاجة إلى التفاصيل أو تنزيل البيانات.

### أين أجد كل نتيجة؟
- **تحليل السوق:** جدول مقارنة التطبيقات والرسوم الخاصة بالتقييمات، ونسب الشكاوى، وفئات المشكلات وتوزيع النجوم.
- **النماذج:** مقاييس النموذجين ومصفوفة الالتباس والعوامل التي أثرت في تصنيف النموذج الأفضل.
- **فحص مراجعة:** الصقي مراجعة جديدة واضغطي زر التحليل لرؤية احتمال كونها حرجة، مستوى الخطورة، وإجراءات متابعة مقترحة.
- **التقرير:** أنشئي تقريرًا عربيًا. يمكن استخدام Gemini إذا توفر مفتاحه الاختياري في الشريط الجانبي؛ وإلا يعرض التطبيق تقريرًا محليًا مختصرًا.
- **البيانات:** راجعي المراجعات التي جُمعت وحقولها، أو نزّليها بصيغة CSV. نص المراجعة يبقى بلغته الأصلية ولا تتم ترجمته تلقائيًا.
- **الاتجاهات:** تابعي تغير نسبة الشكاوى عبر الأسابيع، وجدول مقارنة آخر 7 أيام بالـ7 أيام السابقة.

### كيف أقرأ الأرقام؟
- يعتبر التحليل المراجعة **حرجة إذا كان تقييمها نجمة أو نجمتين**؛ والتقييمات من 3 إلى 5 نجوم غير حرجة وفق هذا التعريف.
- **نسبة الشكاوى الحرجة** هي عدد المراجعات الحرجة مقسومًا على مراجعات التطبيق التي جُمعت. قارنيها بمتوسط المنافسين، لا بعدد الشكاوى وحده.
- **الدقة (Accuracy):** نسبة كل التوقعات الصحيحة من جميع المراجعات المختبرة. **الإحكام (Precision):** من المراجعات التي صنفها النموذج حرجة، كم واحدة كانت حرجة فعلًا. **الاستدعاء (Recall):** من الشكاوى الحرجة الفعلية، كم واحدة اكتشفها النموذج.
- يختار التطبيق النموذج صاحب أعلى **استدعاء** (ثم الإحكام عند التعادل)، لتقليل احتمال تفويت شكوى حرجة. الاختبار يستخدم جزءًا منفصلًا من البيانات، لذلك قد تختلف النتيجة على مراجعات جديدة.
- **مصفوفة الالتباس:** الصفوف تمثل التصنيف الفعلي والأعمدة التصنيف المتوقع؛ الخانة التي تجمع «حرجة» فعليًا و«حرجة» متوقعة هي الشكاوى التي اكتشفها النموذج.
- في فحص المراجعة، الاحتمال أقل من 35% منخفض، ومن 35% إلى أقل من 65% متوسط، و65% فأكثر مرتفع. هذا تقدير آلي وليس تأكيدًا بأن المراجعة صحيحة أو أن المشكلة حدثت.
- **تنبيهات الاتجاهات:** تقارن الأيام السبعة الأخيرة بحسب تاريخ أحدث مراجعة مع الأيام السبعة السابقة لها. يلزم 5 مراجعات على الأقل في كل فترة؛ والزيادة بمقدار 10 نقاط مئوية أو أكثر تُظهر تنبيهًا. قلة البيانات تعني «لا يمكن الحكم»، وليس أن الأداء مستقر.
- تصنيف أسباب الشكاوى يعتمد على كلمات مفتاحية، وقد تتداخل الفئات أو لا تلتقط صياغة غير متوقعة. افتحي تبويب **البيانات** واقرئي النصوص الأصلية للتحقق من أي استنتاج مهم.

### تنبيه عن العينة
المقارنة مبنية على عدد المراجعات الذي طلبتِه والمتاح من Google Play وقت تشغيل التحليل، وليست بالضرورة كل مراجعات التطبيق. اختلاف البلد واللغة وحداثة المراجعات يؤثر في النتائج؛ استخدمي المؤشرات لتحديد ما يستحق التحقيق، ثم تحققي من المراجعات والسياق قبل اتخاذ قرار.
"""
        )


def color_map(primary, apps):
    cm, i = {primary: ACCENT}, 0
    for a in apps:
        if a != primary:
            cm[a] = COMP_COLORS[i % len(COMP_COLORS)]
            i += 1
    return cm


def risk_level(p):
    if p >= 0.65:
        return "high", "🔴 خطورة مرتفعة", "صعّدي الحالة إلى مسؤول الفريق التقني، وردّي علنًا خلال 24 ساعة."
    if p >= 0.35:
        return "med", "🟠 خطورة متوسطة", "حوّلي الحالة إلى فريق الدعم وحددي مالك الميزة لمتابعتها."
    return "low", "🟢 خطورة منخفضة", "لا حاجة للتصعيد حاليًا؛ راقبي ظهور شكاوى مشابهة."


def recommendations(text):
    recs = [
        f"• تم رصد **{ISSUE_LABELS_AR[k]}**: أحِيليها للفريق المسؤول وراجعي آخر تحديث."
        for k, rx in ISSUES.items() if matches_issue(text, rx)
    ]
    return recs or ["• لم يُحدَّد نوع شكوى بعينه؛ راجعي المراجعة يدويًا وقدّمي اعتذارًا ورابط الدعم."]


def offline_report(df, summ, issues, primary):
    comps = summ[summ.app_name != primary]
    me = summ[summ.app_name == primary].iloc[0]
    lines = [f"التقرير التنفيذي (ملخص محلي) - {primary}", f"تاريخ الإنشاء: {datetime.now():%Y-%m-%d %H:%M}", ""]
    lines.append(f"{primary}: متوسط التقييم {me.avg_rating:.2f} من 5، ونسبة الشكاوى الحرجة {me.critical_pct:.1f}% من {int(me.reviews)} مراجعة.")
    lines.append(f"متوسط الشكاوى الحرجة لدى المنافسين: {comps.critical_pct.mean():.1f}%.\n")
    for app, g in issues.groupby("app_name"):
        if g["Share of critical reviews (%)"].max() == 0:
            lines.append(f"- {app}: لم تُرصد شكاوى حرجة ضمن العينة.")
            continue
        top = g.sort_values("Share of critical reviews (%)", ascending=False).iloc[0]
        lines.append(f"- {app}: أبرز فئة شكاوى هي {ISSUE_LABELS_AR.get(top['Issue'], top['Issue'])} ({top['Share of critical reviews (%)']:.0f}% من الشكاوى الحرجة)")
    lines.append("\nأضيفي مفتاح Gemini API لإنشاء تقرير استراتيجي موسّع بالذكاء الاصطناعي.")
    return "\n".join(lines)


def gemini_report(df, summ, issues, primary, key):
    from google import genai

    summary_ar = summ.rename(columns={
        "app_name": "التطبيق",
        "reviews": "عدد المراجعات",
        "avg_rating": "متوسط التقييم",
        "critical_pct": "الشكاوى الحرجة (%)",
        "avg_likes": "متوسط الإعجابات",
    })
    issues_ar = issues.assign(Issue=issues["Issue"].map(ISSUE_LABELS_AR)).rename(columns={
        "app_name": "التطبيق",
        "Issue": "فئة الشكوى",
        "Share of critical reviews (%)": "النسبة من الشكاوى الحرجة (%)",
    })
    ctx = [summary_ar.round(2).to_string(index=False), "\nنسب فئات الشكاوى الحرجة (%):", issues_ar.round(1).to_string(index=False)]
    for app, g in df[df.critical == 1].groupby("app_name"):
        top = g.sort_values("likes", ascending=False).head(3)
        ctx.append(f"\nأكثر المراجعات الحرجة إعجابًا لتطبيق {app} (النص الأصلي):")
        ctx += [f"- ({r.score} نجوم، {r.likes} إعجابًا) {r.content[:220]}" for r in top.itertuples()]
    prompt = f"""أنت محلل أول لاستخبارات المنتجات. التطبيق الرئيسي هو "{primary}".
باستخدام البيانات أدناه فقط (مراجعات Google Play)، اكتب تقريرًا تنفيذيًا باللغة العربية الفصحى، ويتضمن:
1. الملخص التنفيذي (3 نقاط)
2. نقاط الضعف التقنية لدى كل منافس، بالاستناد إلى البيانات
3. نقاط ضعف تطبيقنا مقارنة بالمنافسين
4. أهم 5 أولويات للتطوير مرتبة، مع الأثر المتوقع والجهد (صغير/متوسط/كبير)
5. خطة عمل لمدة 30 يومًا
كن محددًا وموجزًا، ولا تضف استنتاجات لا تدعمها البيانات.

البيانات:
{chr(10).join(ctx)}"""
    from google.genai import types

    client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=60_000))
    last_err = None
    for model in ("gemini-2.5-flash", "gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash", "gemini-flash-latest", "gemini-3.1-flash-lite"):
        try:
            return client.models.generate_content(model=model, contents=prompt).text
        except Exception as e:  # busy / unavailable -> try the next model
            last_err = e
    raise last_err


# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------
st.session_state.setdefault("catalog", dict(DEMO_APPS))
st.session_state.setdefault("hits", [])
st.session_state.setdefault("result", None)
st.session_state.setdefault("report", None)

# ----------------------------------------------------------------------------
# Sidebar - search & selection (Requirement 1)
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📡 AppPulse")
    st.caption("الإنذار المبكر وتحليل المنافسين")
    st.markdown("#### 1 · السوق")
    country_name = st.selectbox("بلد متجر Google Play", list(COUNTRIES), index=0)
    country, lang = COUNTRIES[country_name]
    st.caption("اختيار السعودية أو الإمارات أو مصر أو الأردن يجلب المراجعات باللغة العربية عند توفرها.")

    st.markdown("#### 2 · البحث في Google Play")
    q = st.text_input("اسم التطبيق", placeholder="مثال: Zomato أو Glovo أو Careem")
    if st.button("🔎 بحث", use_container_width=True) and q.strip():
        try:
            hits = search(q.strip(), lang=lang, country=country, n_hits=8)
            st.session_state.hits = hits
            for h in hits:
                st.session_state.catalog.setdefault(h["title"], h["appId"])
        except Exception as e:
            st.error(f"تعذّر البحث: {e}")
    if st.session_state.hits:
        st.caption("التطبيقات المطابقة التي أضيفت للقائمة: " + " · ".join(h["title"] for h in st.session_state.hits[:6]))

    st.markdown("#### 3 · اختيار التطبيقات")
    names = list(st.session_state.catalog)
    primary = st.selectbox("تطبيقنا", names, index=names.index("Uber Eats") if "Uber Eats" in names else 0)
    opts = [n for n in names if n != primary]
    default = [d for d in ["DoorDash", "Grubhub", "Deliveroo"] if d in opts]
    comps = st.multiselect("المنافسون (2–4)", opts, default=default, max_selections=4)
    n_rev = st.slider("عدد المراجعات لكل تطبيق", 200, 3000, 600, 100)
    go_btn = st.button("🚀 ابدأ التحليل", type="primary", use_container_width=True, disabled=len(comps) < 2)
    if len(comps) < 2:
        st.warning("اختاري منافسَين على الأقل.")

    st.markdown("---")
    if not get_api_key():
        st.text_input("مفتاح Gemini API (اختياري)", type="password", key="manual_key")

# ----------------------------------------------------------------------------
# Pipeline (Requirements 2 & 3)
# ----------------------------------------------------------------------------
if go_btn:
    frames, warns = [], []
    bar = st.progress(0.0, text="جارٍ البدء…")
    all_apps = [primary] + comps
    for i, name in enumerate(all_apps):
        bar.progress(i / len(all_apps), text=f"جارٍ جلب المراجعات: {name}")
        try:
            d = fetch_reviews(st.session_state.catalog[name], n_rev, lang, country)
        except Exception as e:
            warns.append(f"{name}: تعذّر جلب المراجعات ({e})")
            continue
        if d.empty:
            warns.append(f"{name}: لم تُعثر على مراجعات في هذا المتجر أو البلد؛ تم التخطي.")
            continue
        d = d.copy()
        d["app_name"] = name
        frames.append(d)
    bar.progress(0.9, text="جارٍ تجهيز الميزات وتدريب النماذج…")
    if primary not in [f["app_name"].iloc[0] for f in frames] or len(frames) < 2:
        bar.empty()
        st.session_state.result = None
        st.error("البيانات غير كافية: نحتاج مراجعات لتطبيقنا ولمنافس واحد على الأقل. جرّبي تطبيقات أو بلدًا آخر.")
        for w in warns:
            st.warning(w)
    else:
        df = build_dataset(frames)
        res = train_models(df)
        st.session_state.result = {"df": df, "res": res, "primary": primary, "warns": warns, "country": country_name}
        st.session_state.report = None
        bar.empty()

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
st.markdown(
    """<div class="hero"><span class="hero-kicker">منصة ذكاء المنتجات · Google Play</span>
<h1><span>AppPulse</span> | رؤية أوضح لصوت المستخدم</h1>
<p>حلّلي مراجعات التطبيقات، قارني أداء المنافسين، واكتشفي مؤشرات الشكاوى قبل أن تتحول إلى مشكلة أكبر.</p>
<div class="hero-meta"><span>تحليل المراجعات</span><span>مقارنة المنافسين</span><span>تنبيهات مبكرة</span><span>تقارير تنفيذية</span></div></div>""",
    unsafe_allow_html=True,
)

R = st.session_state.result
if not R:
    st.markdown("## ابدئي من هنا")
    c = st.columns(4)
    steps = [("1", "اختيار البلد", "حددي سوق Google Play الذي تريدين تحليل مراجعاته."),
             ("2", "اختيار التطبيقات", "اختاري تطبيقك ومنافسَين على الأقل للمقارنة."),
             ("3", "بدء التحليل", "اضغطي الزر في الشريط الجانبي وانتظري جمع البيانات."),
             ("4", "قراءة النتائج", "ابدئي بالخلاصة ثم افتحي التبويب المناسب للتفاصيل.")]
    for col, (n, t, d) in zip(c, steps):
        col.markdown(f'<div class="step"><b>{n}</b><h4>{t}</h4><p>{d}</p></div>', unsafe_allow_html=True)
    st.info("👈 ابدئي بإعداد التطبيق والمنافسين من الشريط الجانبي. الدليل أدناه يشرح كل خطوة ومكان ظهور نتائجها.")
    render_user_guide()
    st.stop()

df, res, primary = R["df"], R["res"], R["primary"]
for w in R["warns"]:
    st.warning(w)
apps = df["app_name"].unique().tolist()
cmap = color_map(primary, apps)
summ = app_summary(df)
issues = issue_table(df)
weekly, comparisons = complaint_trends(df)

primary_summary = summ[summ.app_name == primary].iloc[0]
competitor_avg = float(summ[summ.app_name != primary].critical_pct.mean())
critical_gap = float(primary_summary.critical_pct - competitor_avg)
critical_difference = abs(critical_gap)
if critical_gap < 0:
    comparison_text = f"نسبة الشكاوى الحرجة في {escape(primary)} أقل من متوسط المنافسين بمقدار {critical_difference:.1f} نقطة مئوية."
elif critical_gap > 0:
    comparison_text = f"نسبة الشكاوى الحرجة في {escape(primary)} أعلى من متوسط المنافسين بمقدار {critical_difference:.1f} نقطة مئوية؛ راجعي الشكاوى الأكثر تكرارًا."
else:
    comparison_text = f"نسبة الشكاوى الحرجة في {escape(primary)} تساوي متوسط المنافسين تقريبًا."

primary_issues = issues[issues.app_name == primary].sort_values("Share of critical reviews (%)", ascending=False)
top_issue = primary_issues.iloc[0]
if int(primary_summary.critical_reviews) == 0 or float(top_issue["Share of critical reviews (%)"]) == 0:
    issue_text = "لم تظهر مراجعات حرجة كافية لتحديد فئة شكوى بارزة."
else:
    issue_text = (
        f"أكثر الفئات المطابقة للكلمات المفتاحية لدى {escape(primary)} هي "
        f"{escape(ISSUE_LABELS_AR.get(top_issue['Issue'], top_issue['Issue']))} "
        f"({top_issue['Share of critical reviews (%)']:.1f}% من الشكاوى الحرجة)."
    )

alerts = comparisons[comparisons["Status"] == "Alert: complaints rising"]
insufficient_trend_data = comparisons["Status"].str.startswith("Insufficient").any()
if not alerts.empty:
    alert_names = "، ".join(escape(name) for name in alerts["App"].tolist())
    trend_summary = f"تنبيه اتجاهات للتطبيقات: {alert_names}. راجعي تبويب الاتجاهات للتفاصيل."
elif insufficient_trend_data:
    trend_summary = "بيانات آخر أسبوعين غير كافية لإصدار حكم على اتجاه الشكاوى."
else:
    trend_summary = "لم يظهر ارتفاع يبلغ حد التنبيه في الفترات التي أمكن مقارنتها."

st.markdown("## الخلاصة التنفيذية")
st.caption(
    f"قراءة سريعة لنتيجة {escape(primary)} في {escape(R.get('country', 'البلد المحدد'))} "
    f"مقابل {len(summ) - 1} منافسين، اعتمادًا على {int(summ.reviews.sum()):,} مراجعة جُمعت."
)
summary_metrics = st.columns(4)
summary_metrics[0].metric("متوسط تقييم تطبيقك", f"{primary_summary.avg_rating:.2f} / 5")
summary_metrics[1].metric("الشكاوى الحرجة", f"{primary_summary.critical_pct:.1f}%", f"{int(primary_summary.critical_reviews):,} مراجعة")
summary_metrics[2].metric("متوسط المنافسين", f"{competitor_avg:.1f}%")
summary_metrics[3].metric(
    "الفارق عن المنافسين",
    f"{critical_difference:.1f} نقطة",
    "أقل" if critical_gap < 0 else ("أعلى" if critical_gap > 0 else "مماثل"),
    delta_color="inverse" if critical_gap > 0 else "normal",
)

if res is None:
    model_summary = "لم تتوفر فئتا مراجعات كافيتان لتدريب نموذج موثوق؛ يمكنك متابعة مقارنة السوق والبيانات."
else:
    selected_model = res["table"].loc[res["table"]["Model"] == res["best"]].iloc[0]
    model_summary = (
        f"النموذج المختار: {MODEL_LABELS_AR[res['best']]} "
        f"(استدعاء {selected_model['Recall']:.1%} على بيانات الاختبار)."
    )
st.markdown(
    f'<div class="summary-box">'
    f'<p><b>مقارنة المنافسين:</b> {comparison_text}</p>'
    f'<p><b>أبرز فئة شكاوى:</b> {issue_text}</p>'
    f'<p><b>النموذج والاتجاهات:</b> {escape(model_summary)} {trend_summary}</p>'
    f'<p class="summary-footnote">تُحسب الشكوى الحرجة من تقييم نجمة أو نجمتين. هذه خلاصة للعينة المتاحة وليست حكمًا نهائيًا؛ افتحي دليل الاستخدام أدناه لفهم الحدود والتفاصيل.</p>'
    f'</div>',
    unsafe_allow_html=True,
)
render_user_guide()

t1, t2, t3, t4, t5, t6 = st.tabs([
    "📊 تحليل السوق", "🤖 النماذج", "🚨 فحص مراجعة",
    "🧠 التقرير", "🗂 البيانات", "📈 الاتجاهات",
])

# ---- Dashboard (Requirement 4) --------------------------------------------
with t1:
    st.markdown("### تفاصيل السوق")
    st.caption("تفاصيل المقارنة بين التطبيق الرئيسي والمنافسين المختارين.")
    st.markdown("#### جدول مقارنة التطبيقات")
    comparison = summ.rename(columns={
        "app_name": "التطبيق",
        "reviews": "عدد المراجعات",
        "avg_rating": "متوسط التقييم",
        "critical_reviews": "المراجعات الحرجة",
        "critical_pct": "الشكاوى الحرجة (%)",
        "avg_likes": "متوسط الإعجابات",
    }).copy()
    comparison.insert(1, "الدور", ["التطبيق الرئيسي" if name == primary else "منافس" for name in summ["app_name"]])
    st.dataframe(
        comparison.style.format({
            "عدد المراجعات": "{:,.0f}",
            "متوسط التقييم": "{:.2f} / 5",
            "المراجعات الحرجة": "{:,.0f}",
            "الشكاوى الحرجة (%)": "{:.1f}%",
            "متوسط الإعجابات": "{:.1f}",
        }).highlight_max(subset=["متوسط التقييم"], color="#dff3ea")
          .highlight_min(subset=["الشكاوى الحرجة (%)"], color="#dff3ea"),
        use_container_width=True,
        hide_index=True,
    )
    st.markdown("#### نظرة بصرية على السوق")
    a, b = st.columns(2)
    f1 = px.bar(summ, x="app_name", y="avg_rating", color="app_name", color_discrete_map=cmap, text=summ.avg_rating.round(2), title="متوسط التقييم")
    f1.update_yaxes(range=[0, 5], title="التقييم من 5"); f1.update_layout(showlegend=False, xaxis_title="")
    a.plotly_chart(style(f1), use_container_width=True)
    f2 = px.bar(summ, x="app_name", y="critical_pct", color="app_name", color_discrete_map=cmap, text=summ.critical_pct.round(1), title="نسبة الشكاوى الحرجة (≤ نجمتين)")
    f2.update_layout(showlegend=False, xaxis_title="", yaxis_title="النسبة المئوية")
    b.plotly_chart(style(f2), use_container_width=True)
    c, d = st.columns(2)
    chart_issues = issues.assign(Issue=issues["Issue"].map(ISSUE_LABELS_AR))
    f3 = px.bar(chart_issues, x="Issue", y="Share of critical reviews (%)", color="app_name", barmode="group", color_discrete_map=cmap, title="أبرز أسباب الشكاوى")
    f3.update_layout(xaxis_title="فئة الشكوى", yaxis_title="من الشكاوى الحرجة (%)")
    c.plotly_chart(style(f3), use_container_width=True)
    dist = df.groupby(["app_name", "score"]).size().reset_index(name="n")
    dist["pct"] = 100 * dist["n"] / dist.groupby("app_name")["n"].transform("sum")
    f4 = px.bar(dist, x="app_name", y="pct", color=dist["score"].astype(str), title="توزيع التقييمات (%)",
                color_discrete_sequence=["#d85c55", "#df8952", "#c79c36", "#76a94b", "#0f9d7a"],
                category_orders={"color": ["1", "2", "3", "4", "5"]})
    f4.update_layout(xaxis_title="التطبيق", yaxis_title="النسبة المئوية", legend_title_text="عدد النجوم")
    d.plotly_chart(style(f4), use_container_width=True)

# ---- Benchmark (Requirement 3) --------------------------------------------
with t2:
    if res is None:
        st.error("لا توجد مراجعات حرجة وغير حرجة كافية لتدريب النماذج. زيدي عدد المراجعات أو اختاري تطبيقات أخرى.")
    else:
        st.markdown(f"**تقسيم التدريب والاختبار:** 80/20 مع الحفاظ على نسب الفئات · `random_state=42` · حجم الاختبار = {res['n_test']} مراجعة · أوزان الفئات متوازنة")
        tbl = res["table"].copy()
        tbl["Model"] = tbl["Model"].map(MODEL_LABELS_AR)
        tbl = tbl.rename(columns={
            "Model": "النموذج",
            "Accuracy": "الدقة",
            "Precision": "الإحكام",
            "Recall": "الاستدعاء",
        })
        st.dataframe(
            tbl.style.format({"الدقة": "{:.3f}", "الإحكام": "{:.3f}", "الاستدعاء": "{:.3f}"}).highlight_max(subset=["الدقة", "الإحكام", "الاستدعاء"], color="#dff3ea"),
            use_container_width=True, hide_index=True,
        )
        st.markdown(f'<div class="banner good">🏆 تم اختيار <b>{MODEL_LABELS_AR[res["best"]]}</b> تلقائيًا لأنه حقق أعلى <b>استدعاء</b>. '
                    'فقدان شكوى حرجة فعلية أخطر من إطلاق إنذار زائد؛ لذلك نُعطي الاستدعاء أولوية على الدقة.</div>', unsafe_allow_html=True)
        cc = st.columns(2)
        for col, (name, cm) in zip(cc, res["cms"].items()):
            fig = px.imshow(cm, text_auto=True, x=["غير حرجة", "حرجة"], y=["غير حرجة", "حرجة"],
                            color_continuous_scale=["#e8f5ef", "#0f9d7a"],
                            title=f"مصفوفة الالتباس · {MODEL_LABELS_AR[name]}")
            fig.update_layout(xaxis_title="التصنيف المتوقع", yaxis_title="التصنيف الفعلي", coloraxis_showscale=False)
            col.plotly_chart(style(fig, 340), use_container_width=True)
        imp = feature_importance(res)
        fi = px.bar(imp, x="Importance", y="Feature", orientation="h",
                    title=f"أهم العوامل المؤثرة في الشكاوى ({MODEL_LABELS_AR[res['best']]})")
        fi.update_layout(xaxis_title="الأهمية", yaxis_title="الميزة")
        fi.update_traces(marker_color=ACCENT)
        st.plotly_chart(style(fi, 360), use_container_width=True)

# ---- Live triage (Requirement 4) ------------------------------------------
with t3:
    if res is None:
        st.error("لا يتوفر نموذج مدرّب لهذا الاختيار.")
    else:
        st.markdown(f"اختبري أي مراجعة جديدة باستخدام نموذج **{MODEL_LABELS_AR[res['best']]}**.")
        with st.form("triage"):
            txt = st.text_area("نص المراجعة", height=130, placeholder="يتعطل التطبيق كلما حاولت الدفع! الدعم لا يرد.")
            c1, c2 = st.columns(2)
            likes = c1.number_input("عدد الإعجابات", 0, 10000, 0)
            hour = c2.slider("ساعة النشر", 0, 23, datetime.now().hour)
            sub = st.form_submit_button("حلّلي مستوى الخطورة", type="primary")
        if sub:
            if not txt.strip():
                st.warning("أدخلي نص المراجعة أولًا.")
            else:
                X = pd.DataFrame([text_features(txt, likes, hour)])[FEATURES]
                p = float(res["models"][res["best"]].predict_proba(X)[0, 1])
                cls, label, action = risk_level(p)
                g, m = st.columns([1, 1.3])
                gauge = go.Figure(go.Indicator(
                    mode="gauge+number", value=p * 100, number={"suffix": "%"},
                    gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#334155"},
                           "steps": [{"range": [0, 35], "color": "#dff3ea"}, {"range": [35, 65], "color": "#fff1c9"}, {"range": [65, 100], "color": "#fde1df"}]},
                    title={"text": "احتمال أن تكون الشكوى حرجة"}))
                g.plotly_chart(style(gauge, 300), use_container_width=True)
                m.markdown(f'<div class="risk {cls}"><b>{label}</b><br>{action}</div>', unsafe_allow_html=True)
                m.markdown("**إجراءات مقترحة**")
                for r in recommendations(txt):
                    m.markdown(r)
                if p >= 0.65:
                    m.markdown("• أنشئي تذكرة بأولوية قصوى وأبلغي مديرة المنتج.")
                other = [n for n in res["models"] if n != res["best"]][0]
                po = float(res["models"][other].predict_proba(X)[0, 1])
                m.caption(f"مقارنة بالنموذج الآخر · {MODEL_LABELS_AR[other]}: {po * 100:.1f}%")

# ---- Executive report (Requirement 5) --------------------------------------
with t4:
    key = get_api_key()
    st.markdown("أنشئي تقريرًا مقارنًا باللغة العربية باستخدام **Gemini Flash**.")
    if st.button("✨ إنشاء التقرير التنفيذي", type="primary"):
        with st.spinner("جارٍ تحليل البيانات…"):
            try:
                if not key:
                    raise ValueError("no key")
                st.session_state.report = gemini_report(df, summ, issues, primary, key)
            except Exception as e:
                st.session_state.report = offline_report(df, summ, issues, primary)
                st.warning("تعذّر استخدام Gemini (تحققي من المفتاح)؛ يُعرض ملخص محلي بدلًا منه." if key else "لم يُعثر على مفتاح Gemini؛ يُعرض ملخص محلي بدلًا منه.")
    if st.session_state.report:
        st.markdown(st.session_state.report)
        st.download_button("⬇️ تنزيل التقرير (TXT)", st.session_state.report, "executive_report.txt", "text/plain; charset=utf-8")

# ---- Data ------------------------------------------------------------------
with t5:
    st.caption("تعرض نصوص المراجعات بلغتها الأصلية كما وردت من Google Play؛ لا تتم ترجمة نص المراجعة تلقائيًا.")
    show = df[["app_name", "score", "critical", "likes", "text_len", "word_count", "exclam_count", "crash_kw_count", "caps_ratio", "hour", "at", "content"]].rename(columns={
        "app_name": "التطبيق",
        "score": "التقييم (نجوم)",
        "critical": "شكوى حرجة",
        "likes": "الإعجابات",
        "text_len": "طول النص",
        "word_count": "عدد الكلمات",
        "exclam_count": "علامات التعجب",
        "crash_kw_count": "كلمات الأعطال",
        "caps_ratio": "نسبة الأحرف الكبيرة",
        "hour": "ساعة النشر",
        "at": "تاريخ المراجعة",
        "content": "نص المراجعة (باللغة الأصلية)",
    })
    show["شكوى حرجة"] = show["شكوى حرجة"].map({0: "لا", 1: "نعم"})
    st.dataframe(show, use_container_width=True, height=420)
    st.download_button("⬇️ تنزيل البيانات المنظفة (CSV)", show.to_csv(index=False).encode("utf-8-sig"), "cleaned_reviews.csv", "text/csv; charset=utf-8")

# ---- Complaint trend alerts -----------------------------------------------
with t6:
    st.markdown("تابعي تغيّر الشكاوى الحرجة بمرور الوقت لاكتشاف أي ارتفاع مفاجئ.")
    st.caption(
        "يجمع الرسم المراجعات حسب الأسبوع. يقارن الجدول آخر 7 أيام بالسبعة السابقة، "
        "ويشترط 5 مراجعات على الأقل في كل فترة. يظهر التنبيه عند زيادة قدرها 10 نقاط "
        "مئوية أو أكثر؛ وهو إشارة للمراقبة وليس اختبارًا للدلالة الإحصائية."
    )
    weekly_view = weekly.rename(columns={
        "week": "الأسبوع",
        "app_name": "التطبيق",
        "reviews": "عدد المراجعات",
        "critical_pct": "الشكاوى الحرجة (%)",
    })
    trend_fig = px.line(
        weekly_view, x="الأسبوع", y="الشكاوى الحرجة (%)", color="التطبيق", markers=True,
        color_discrete_map=cmap, hover_data=["عدد المراجعات"],
        title="الشكاوى الحرجة حسب الأسبوع",
    )
    trend_fig.update_layout(xaxis_title="الأسبوع", yaxis_title="الشكاوى الحرجة (%)")
    trend_fig.update_yaxes(rangemode="tozero")
    st.plotly_chart(style(trend_fig), use_container_width=True)
    comparisons_view = comparisons.rename(columns=TREND_COLUMN_LABELS_AR).copy()
    comparisons_view["الحالة"] = comparisons_view["الحالة"].replace({
        "Alert: complaints rising": "تنبيه: ارتفاع الشكاوى",
        "Slight increase": "ارتفاع طفيف",
        "Stable or decreasing": "مستقرة أو في انخفاض",
    })
    comparisons_view.loc[
        comparisons_view["الحالة"].str.startswith("Insufficient data"),
        "الحالة",
    ] = "بيانات غير كافية (يلزم 5 مراجعات على الأقل لكل فترة)"
    st.dataframe(
        comparisons_view.style.format(
            {
                "الشكاوى الحرجة سابقًا (%)": "{:.1f}",
                "الشكاوى الحرجة حاليًا (%)": "{:.1f}",
                "التغير (نقطة مئوية)": "{:+.1f}",
            },
            na_rep="—",
        ),
        use_container_width=True,
        hide_index=True,
    )
    alerts = comparisons[comparisons["Status"] == "Alert: complaints rising"]
    if not alerts.empty:
        for _, alert in alerts.iterrows():
            st.warning(
                f"{alert['App']}: ارتفعت الشكاوى الحرجة بمقدار "
                f"{alert['Change (percentage points)']:.1f} نقطة مئوية. "
                "راجعي التحديثات الأخيرة وملاحظات المستخدمين."
            )
    elif comparisons["Status"].str.startswith("Insufficient").any():
        st.info(
            "لم يصدر تنبيه. لا يمكن اعتبار قلة المراجعات دليلًا على استقرار مستوى الشكاوى."
        )
    else:
        st.success("لم يسجل أي تطبيق ارتفاعًا في الشكاوى الحرجة يبلغ حد التنبيه أو يتجاوزه.")
