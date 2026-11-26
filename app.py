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

ACCENT = "#00E5A8"
COMP_COLORS = ["#7C6CFF", "#FF7A59", "#4DA3FF", "#FFC857"]

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;800&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Arabic:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Noto Sans Arabic', 'Inter', sans-serif; }
.stApp, .hero, .hero p, .hero h1, .hero-meta, .step, .step h4, .step p,
.stMarkdown, .stCaption, .stTabs, div[data-testid="stMetric"] {
        font-family: 'Noto Sans Arabic', 'Inter', sans-serif !important; }
:root { color-scheme: dark; }
.stApp { direction: rtl; background: radial-gradient(900px 500px at 95% -8%, #17304a 0%, transparent 62%), #080d19; }
.block-container { max-width: 1500px; padding-top: 2rem; padding-bottom: 3rem; }
section[data-testid="stSidebar"] { background: #0b1220; border-left: 1px solid #202c41; }
section[data-testid="stSidebar"] > div { padding-top: 1.5rem; }
.hero { position: relative; overflow: hidden; padding: 34px 38px; border-radius: 26px; margin: 4px 0 24px;
        background: linear-gradient(115deg, rgba(17,31,53,.98), rgba(20,48,66,.94) 56%, rgba(12,82,80,.72));
        border: 1px solid rgba(133,190,211,.2); box-shadow: 0 22px 60px rgba(0,0,0,.24); }
.hero:after { content: ""; position: absolute; width: 230px; height: 230px; border-radius: 50%;
        left: -60px; top: -110px; background: rgba(0,229,168,.10); filter: blur(2px); }
.hero-kicker { display: inline-block; padding: 5px 12px; margin-bottom: 13px; border-radius: 999px;
        color: #9ef5dc; background: rgba(0,229,168,.10); border: 1px solid rgba(0,229,168,.22);
        font-size: .82rem; font-weight: 700; }
.hero h1 { position: relative; z-index: 1; margin: 0; font-size: clamp(1.7rem, 3vw, 2.45rem);
        font-weight: 800; letter-spacing: -.5px; color: #f4f8ff; }
.hero h1 span { color: #48e6bc; }
.hero p { position: relative; z-index: 1; max-width: 850px; margin: 10px 0 0; color: #b8c8dd; font-size: 1rem; line-height: 1.9; }
.hero-meta { position: relative; z-index: 1; display: flex; flex-wrap: wrap; gap: 8px; margin-top: 19px; }
.hero-meta span { padding: 6px 11px; border: 1px solid rgba(196,215,238,.15); border-radius: 999px;
        background: rgba(4,12,24,.25); color: #d3deed; font-size: .78rem; }
.step { background: linear-gradient(145deg, #121d30, #0d1626); border: 1px solid #24344b;
        border-radius: 18px; padding: 20px; min-height: 145px; box-shadow: 0 10px 28px rgba(0,0,0,.13); }
.step b { display: inline-grid; place-items: center; width: 34px; height: 34px; border-radius: 11px;
        color: #06121e; background: linear-gradient(135deg,#62f0c7,#43bce5); font-size: 1rem; }
.step h4 { margin: 12px 0 5px; color: #f2f6ff; font-size: 1rem; }
.step p { color: #9eafc7; margin: 0; font-size: .88rem; line-height: 1.7; }
div[data-testid="stMetric"] { height: 100%; background: linear-gradient(145deg, #121d30, #0e1726);
        border: 1px solid #25354b; padding: 17px 19px; border-radius: 18px; box-shadow: 0 10px 26px rgba(0,0,0,.12); }
div[data-testid="stMetricLabel"] { color: #aabbd1; }
div[data-testid="stMetricValue"] { font-weight: 800; color: #f3f7ff; }
div[data-testid="stMetricDelta"] { font-size: .8rem; }
.banner { padding: 16px 20px; border-radius: 16px; margin: 14px 0 22px; font-weight: 500; line-height: 1.8; }
.banner.good { background: linear-gradient(90deg,rgba(0,229,168,.12),rgba(0,229,168,.04)); border: 1px solid rgba(0,229,168,.45); color: #c4ffed; }
.banner.bad { background: linear-gradient(90deg,rgba(255,90,90,.13),rgba(255,90,90,.04)); border: 1px solid rgba(255,110,110,.48); color: #ffe0e0; }
.risk { padding: 20px 22px; border-radius: 18px; font-size: 1rem; line-height: 1.9; }
.risk.low { background: rgba(0,229,168,.09); border: 1px solid rgba(0,229,168,.5); }
.risk.med { background: rgba(255,200,87,.09); border: 1px solid rgba(255,200,87,.55); }
.risk.high { background: rgba(255,90,90,.10); border: 1px solid rgba(255,90,90,.58); }
button[kind="primary"] { background: linear-gradient(100deg,#49e2b9,#4cbfe4) !important; color:#06121e !important;
        font-weight: 800 !important; border: 0 !important; box-shadow: 0 8px 22px rgba(0,210,174,.18); }
button[kind="primary"]:hover { filter: brightness(1.08); transform: translateY(-1px); }
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid #25334a; }
.stTabs [data-baseweb="tab"] { min-height: 48px; padding: 0 15px; font-weight: 650; color: #aebed2; }
.stTabs [aria-selected="true"] { color: #68eac7 !important; }
div[data-testid="stDataFrame"], div[data-testid="stTable"] { border: 1px solid #26364d; border-radius: 14px; overflow: hidden; }
div[data-testid="stPlotlyChart"] { padding: 8px; border: 1px solid rgba(45,63,86,.65); border-radius: 18px;
        background: rgba(12,20,34,.45); }
@media (max-width: 700px) {
    .block-container { padding: 1rem .8rem 2rem; }
    .hero { padding: 25px 20px; border-radius: 20px; }
    .hero h1 { font-size: 1.65rem; }
    .stTabs [data-baseweb="tab"] { padding: 0 9px; font-size: .82rem; }
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
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=h, margin=dict(l=10, r=10, t=50, b=10), font=dict(family="Noto Sans Arabic, Inter", color="#dbe4ff"),
        legend_title_text="",
    )
    fig.update_xaxes(gridcolor="#1f2a4d")
    fig.update_yaxes(gridcolor="#1f2a4d")
    return fig


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
    c = st.columns(4)
    steps = [("1", "البحث", "ابحثي عن أي تطبيق في Google Play."), ("2", "جمع البيانات", "اجلبي أحدث المراجعات لكل تطبيق."),
             ("3", "المقارنة", "قارني الانحدار اللوجستي بالغابة العشوائية."), ("4", "التقرير", "أنشئي تقريرًا تنفيذيًا ذكيًا.")]
    for col, (n, t, d) in zip(c, steps):
        col.markdown(f'<div class="step"><b>{n}</b><h4>{t}</h4><p>{d}</p></div>', unsafe_allow_html=True)
    st.info("👈 اختاري تطبيقك ومنافسيه من الشريط الجانبي، ثم اضغطي **ابدأ التحليل**.")
    st.stop()

df, res, primary = R["df"], R["res"], R["primary"]
for w in R["warns"]:
    st.warning(w)
apps = df["app_name"].unique().tolist()
cmap = color_map(primary, apps)
summ = app_summary(df)
issues = issue_table(df)

t1, t2, t3, t4, t5, t6 = st.tabs([
    "📊 لوحة المعلومات", "🤖 مقارنة النماذج", "🚨 تقييم المراجعات",
    "🧠 التقرير التنفيذي", "🗂 البيانات", "📈 تنبيهات الاتجاهات",
])

# ---- Dashboard (Requirement 4) --------------------------------------------
with t1:
    me = summ[summ.app_name == primary].iloc[0]
    comp_avg = summ[summ.app_name != primary].critical_pct.mean()
    diff = me.critical_pct - comp_avg
    cls = "good" if diff <= 0 else "bad"
    word = "أقل من" if diff <= 0 else "أعلى من"
    st.markdown("### ملخص الأداء")
    st.caption(f"مقارنة {primary} مع {len(summ) - 1} منافسين · {int(summ.reviews.sum()):,} مراجعة ضمن العينة")
    metrics = st.columns(4)
    metrics[0].metric("متوسط تقييم تطبيقك", f"{me.avg_rating:.2f} / 5", f"{int(me.reviews):,} مراجعة")
    metrics[1].metric("الشكاوى الحرجة", f"{me.critical_pct:.1f}%", f"{int(me.critical_reviews):,} مراجعة حرجة")
    metrics[2].metric("متوسط المنافسين", f"{comp_avg:.1f}%", f"{len(summ) - 1} تطبيقات")
    metrics[3].metric(
        "الفارق عن المنافسين",
        f"{abs(diff):.1f} نقطة",
        "أداء أفضل" if diff < 0 else ("مماثل" if diff == 0 else "يحتاج متابعة"),
        delta_color="normal" if diff <= 0 else "inverse",
    )
    st.markdown(
        f'<div class="banner {cls}"><b>{escape(primary)}</b>: نسبة الشكاوى الحرجة <b>{abs(diff):.1f} نقطة مئوية {word}</b> متوسط المنافسين ({comp_avg:.1f}%).</div>',
        unsafe_allow_html=True,
    )
    st.markdown("#### مقارنة التطبيقات")
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
        }).highlight_max(subset=["متوسط التقييم"], color="#164c44")
          .highlight_min(subset=["الشكاوى الحرجة (%)"], color="#164c44"),
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
                color_discrete_sequence=["#ff5a5a", "#ff9f5a", "#ffd25a", "#9be564", "#00E5A8"],
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
            tbl.style.format({"الدقة": "{:.3f}", "الإحكام": "{:.3f}", "الاستدعاء": "{:.3f}"}).highlight_max(subset=["الدقة", "الإحكام", "الاستدعاء"], color="#0b5d49"),
            use_container_width=True, hide_index=True,
        )
        st.markdown(f'<div class="banner good">🏆 تم اختيار <b>{MODEL_LABELS_AR[res["best"]]}</b> تلقائيًا لأنه حقق أعلى <b>استدعاء</b>. '
                    'فقدان شكوى حرجة فعلية أخطر من إطلاق إنذار زائد؛ لذلك نُعطي الاستدعاء أولوية على الدقة.</div>', unsafe_allow_html=True)
        cc = st.columns(2)
        for col, (name, cm) in zip(cc, res["cms"].items()):
            fig = px.imshow(cm, text_auto=True, x=["غير حرجة", "حرجة"], y=["غير حرجة", "حرجة"],
                            color_continuous_scale=["#131A2E", "#00E5A8"],
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
                    gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#fff"},
                           "steps": [{"range": [0, 35], "color": "#0b5d49"}, {"range": [35, 65], "color": "#8a6d1c"}, {"range": [65, 100], "color": "#8a2b2b"}]},
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
    weekly, comparisons = complaint_trends(df)
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
