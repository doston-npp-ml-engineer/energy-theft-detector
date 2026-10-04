import streamlit as st
import pandas as pd
import numpy as np
import requests
from xgboost import XGBClassifier

st.set_page_config(page_title="Elektr O'g'irligini Aniqlash", page_icon="⚡")

st.title("⚡ Elektr Energiyasi O'g'irligini Aniqlash Tizimi")
st.write("XGBoost va early stopping asosida qurilgan model — SGCC dataset + ob-havo")

FEATURE_ORDER = [
    'mean_consumption', 'std_consumption', 'median_consumption',
    'min_consumption', 'max_consumption', 'cv', 'zero_ratio',
    'missing_ratio', 'q25', 'q75', 'iqr', 'mean_first_half',
    'mean_second_half', 'trend_diff', 'mean_daily_diff', 'max_daily_diff',
    'weekly_std', 'weather_temp_corr'
]

# Jiangsu provinsiyasi markazi (Nankin) — SGCC shu hududda joylashgan
LAT, LON = 32.06, 118.78


@st.cache_resource
def load_model():
    model = XGBClassifier()
    model.load_model("theft_model.json")
    return model


@st.cache_data(show_spinner=False)
def fetch_weather(start_date, end_date):
    """Open-Meteo tarixiy arxividan kunlik o'rtacha haroratni oladi."""
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={LAT}&longitude={LON}&start_date={start_date}&end_date={end_date}"
        "&daily=temperature_2m_mean&timezone=Asia%2FShanghai"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()
    return pd.Series(
        data["daily"]["temperature_2m_mean"],
        index=pd.to_datetime(data["daily"]["time"]),
    )


def compute_weather_corr(consumption, date_cols):
    """Har bir mijoz uchun iste'mol-harorat korrelyatsiyasini hisoblaydi.
    Sana ustunlarini aniqlab bo'lmasa yoki ob-havo yuklanmasa, 0 qaytaradi
    va foydalanuvchiga ogohlantirish ko'rsatadi."""
    parsed = pd.to_datetime(pd.Series(date_cols), errors="coerce")
    if parsed.isna().any():
        st.warning(
            "⚠️ Ustun nomlari sana sifatida tanilmadi, shuning uchun "
            "ob-havo xususiyati 0 deb olindi (natijaga ozgina ta'sir qiladi)."
        )
        return pd.Series(0.0, index=consumption.index)

    try:
        weather = fetch_weather(
            parsed.min().strftime("%Y-%m-%d"), parsed.max().strftime("%Y-%m-%d")
        )
    except Exception:
        st.warning(
            "⚠️ Ob-havo ma'lumotini yuklab bo'lmadi (internet yoki xizmat "
            "muammosi), ob-havo xususiyati 0 deb olindi."
        )
        return pd.Series(0.0, index=consumption.index)

    temp_series = weather.reindex(parsed.values).values
    temp_centered = temp_series - np.nanmean(temp_series)

    X = consumption.values.astype(float)
    X_centered = X - np.nanmean(X, axis=1, keepdims=True)

    num = np.nansum(X_centered * temp_centered, axis=1)
    den = np.sqrt(np.nansum(X_centered ** 2, axis=1) * np.nansum(temp_centered ** 2))
    corr = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    return pd.Series(corr, index=consumption.index)


model = load_model()
st.success("Model muvaffaqiyatli yuklandi ✅")

tab1, tab2 = st.tabs(["📝 Qo'lda kiritish", "📁 CSV yuklash"])

with tab1:
    st.subheader("Mijoz statistikalarini qo'lda kiriting")

    col1, col2 = st.columns(2)

    with col1:
        mean_consumption = st.number_input("O'rtacha kunlik iste'mol (kWh)", min_value=0.0, value=5.0)
        std_consumption = st.number_input("Iste'mol standart og'ishi", min_value=0.0, value=3.0)
        median_consumption = st.number_input("Median iste'mol", min_value=0.0, value=4.0)
        min_consumption = st.number_input("Minimal kunlik iste'mol", min_value=0.0, value=0.0)
        max_consumption = st.number_input("Maksimal kunlik iste'mol", min_value=0.0, value=20.0)
        cv = st.number_input("Variatsiya koeffitsienti (CV)", min_value=0.0, value=0.5)
        zero_ratio = st.slider("Nol iste'molli kunlar ulushi", 0.0, 1.0, 0.05)
        missing_ratio = st.slider("Yo'q ma'lumotlar ulushi", 0.0, 1.0, 0.1)

    with col2:
        q25 = st.number_input("25-kvartil", min_value=0.0, value=2.0)
        q75 = st.number_input("75-kvartil", min_value=0.0, value=7.0)
        iqr = st.number_input("IQR (q75-q25)", min_value=0.0, value=5.0)
        mean_first_half = st.number_input("Birinchi yarim davr o'rtachasi", min_value=0.0, value=4.0)
        mean_second_half = st.number_input("Ikkinchi yarim davr o'rtachasi", min_value=0.0, value=5.0)
        trend_diff = st.number_input("Trend farqi (2-yarim - 1-yarim)", value=1.0)
        mean_daily_diff = st.number_input("O'rtacha kunlik o'zgarish", min_value=0.0, value=1.5)
        max_daily_diff = st.number_input("Maksimal kunlik o'zgarish", min_value=0.0, value=10.0)
        weekly_std = st.number_input("Haftalik std", min_value=0.0, value=3.0)

    weather_temp_corr = st.slider(
        "Ob-havoga sezgirlik (iste'mol-harorat korrelyatsiyasi)",
        -1.0, 1.0, 0.3,
        help="Halol abonentda odatda musbat yoki manfiy tomonga kuchliroq bog'liqlik bo'ladi (masalan yozda konditsioner). 0 ga yaqin qiymat — iste'mol haroratdan deyarli mustaqil."
    )

    if st.button("🔍 Xulosa chiqarish", key="manual_predict"):
        input_values = {
            'mean_consumption': mean_consumption,
            'std_consumption': std_consumption,
            'median_consumption': median_consumption,
            'min_consumption': min_consumption,
            'max_consumption': max_consumption,
            'cv': cv,
            'zero_ratio': zero_ratio,
            'missing_ratio': missing_ratio,
            'q25': q25,
            'q75': q75,
            'iqr': iqr,
            'mean_first_half': mean_first_half,
            'mean_second_half': mean_second_half,
            'trend_diff': trend_diff,
            'mean_daily_diff': mean_daily_diff,
            'max_daily_diff': max_daily_diff,
            'weekly_std': weekly_std,
            'weather_temp_corr': weather_temp_corr,
        }

        input_df = pd.DataFrame([input_values])[FEATURE_ORDER]

        proba = model.predict_proba(input_df)[0, 1]
        pred = int(proba >= 0.5)

        st.markdown("---")
        if pred == 1:
            st.error(f"⚠️ O'g'irlik ehtimoli YUQORI: {proba:.1%}")
        else:
            st.success(f"✅ Normal iste'mol, o'g'irlik ehtimoli: {proba:.1%}")

        st.progress(float(proba))

with tab2:
    st.subheader("Xom kunlik iste'mol ma'lumotlarini yuklang")
    st.write("CSV formatida: 1-ustun mijoz ID, 2-ustun mahalla kodi, keyingi ustunlar — har bir kun uchun iste'mol (sana nomlar bilan)")

    uploaded_file = st.file_uploader("CSV faylni tanlang", type=["csv"])

    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        st.write(f"Yuklandi: {raw_df.shape[0]} mijoz, {raw_df.shape[1]} ustun")
        st.dataframe(raw_df.head())

        id_col = raw_df.columns[0]
        mahalla_col = raw_df.columns[1]
        date_cols = [c for c in raw_df.columns if c not in (id_col, mahalla_col)]
        consumption = raw_df[date_cols]

        feats = pd.DataFrame(index=raw_df.index)
        feats['mean_consumption'] = consumption.mean(axis=1)
        feats['std_consumption'] = consumption.std(axis=1)
        feats['median_consumption'] = consumption.median(axis=1)
        feats['min_consumption'] = consumption.min(axis=1)
        feats['max_consumption'] = consumption.max(axis=1)
        feats['cv'] = feats['std_consumption'] / (feats['mean_consumption'] + 1e-6)
        feats['zero_ratio'] = (consumption == 0).sum(axis=1) / consumption.shape[1]
        feats['missing_ratio'] = consumption.isna().sum(axis=1) / consumption.shape[1]
        feats['q25'] = consumption.quantile(0.25, axis=1)
        feats['q75'] = consumption.quantile(0.75, axis=1)
        feats['iqr'] = feats['q75'] - feats['q25']

        mid = len(date_cols) // 2
        feats['mean_first_half'] = consumption.iloc[:, :mid].mean(axis=1)
        feats['mean_second_half'] = consumption.iloc[:, mid:].mean(axis=1)
        feats['trend_diff'] = feats['mean_second_half'] - feats['mean_first_half']

        diffs = consumption.diff(axis=1)
        feats['mean_daily_diff'] = diffs.abs().mean(axis=1)
        feats['max_daily_diff'] = diffs.abs().max(axis=1)
        feats['weekly_std'] = consumption.std(axis=1)

        with st.spinner("Ob-havo ma'lumoti yuklanmoqda..."):
            feats['weather_temp_corr'] = compute_weather_corr(consumption, date_cols)

        feats = feats.fillna(0).replace([np.inf, -np.inf], 0)
        X_new = feats[FEATURE_ORDER]

        probas = model.predict_proba(X_new)[:, 1]
        results = pd.DataFrame({
            id_col: raw_df[id_col],
            mahalla_col: raw_df[mahalla_col],
            'ogirlik_ehtimoli': probas,
            'xulosa': np.where(probas >= 0.5, "O'G'IRLIK", "Normal")
        }).sort_values('ogirlik_ehtimoli', ascending=False)

        st.subheader("Natijalar")
        st.dataframe(results)

        st.subheader("Mahalla bo'yicha xavf darajasi")
        mahalla_stats = results.groupby(mahalla_col)['ogirlik_ehtimoli'].agg(
            ortacha_xavf='mean',
            abonentlar_soni='count',
            yuqori_xavfli=lambda x: (x >= 0.5).sum()
        ).sort_values('ortacha_xavf', ascending=False)
        st.dataframe(mahalla_stats)
        st.bar_chart(mahalla_stats['ortacha_xavf'])

        st.subheader("Energiya balansi (transformator darajasida) — ixtiyoriy")
        st.write("Mahalladagi abonentlar yig'indisini transformatordan chiqqan energiya bilan solishtiradi. Bu qatlam hozircha namoyish uchun — real transformator ma'lumoti kerak.")

        transformer_file = st.file_uploader(
            "Transformator ma'lumotini yuklang (mahalla_id, transformator_kwh ustunlari bilan)",
            type=["csv"], key="transformer_upload"
        )

        if transformer_file is not None:
            transformer_df = pd.read_csv(transformer_file)
            abonent_jami = consumption.sum(axis=1)
            jami_df = pd.DataFrame({mahalla_col: raw_df[mahalla_col], 'abonentlar_jami_kwh': abonent_jami})
            mahalla_jami = jami_df.groupby(mahalla_col)['abonentlar_jami_kwh'].sum()

            balans = transformer_df.set_index(mahalla_col).join(mahalla_jami)
            balans['yoqotish_kwh'] = balans['transformator_kwh'] - balans['abonentlar_jami_kwh']
            balans['yoqotish_foiz'] = (balans['yoqotish_kwh'] / balans['transformator_kwh'] * 100).round(1)

            st.dataframe(balans.sort_values('yoqotish_foiz', ascending=False))
            st.bar_chart(balans['yoqotish_foiz'])

        csv_out = results.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Natijani CSV sifatida yuklab olish", csv_out, "natijalar.csv", "text/csv")
