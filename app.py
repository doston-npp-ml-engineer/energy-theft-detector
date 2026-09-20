import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier

st.set_page_config(page_title="Elektr O'g'irligini Aniqlash", page_icon="⚡")

st.title("⚡ Elektr Energiyasi O'g'irligini Aniqlash Tizimi")
st.write("XGBoost va early stopping asosida qurilgan model — SGCC dataset")

# Modelni yuklash
@st.cache_resource
def load_model():
    model = XGBClassifier()
    model.load_model("theft_model.json")
    return model

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
    
    if st.button("🔍 Bashorat qilish", key="manual_predict"):
        feature_order = [
            'mean_consumption', 'std_consumption', 'median_consumption',
            'min_consumption', 'max_consumption', 'cv', 'zero_ratio',
            'missing_ratio', 'q25', 'q75', 'iqr', 'mean_first_half',
            'mean_second_half', 'mean_daily_diff', 'max_daily_diff', 'weekly_std',
            'trend_diff'
        ]

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
            'mean_daily_diff': mean_daily_diff,
            'max_daily_diff': max_daily_diff,
            'weekly_std': weekly_std,
            'trend_diff': trend_diff,
        }

        input_df = pd.DataFrame([input_values])[feature_order]

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
    st.write("CSV formatida: birinchi ustun mijoz ID, keyingi ustunlar — har bir kun uchun iste'mol (sana nomlar bilan)")

    uploaded_file = st.file_uploader("CSV faylni tanlang", type=["csv"])

    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        st.write(f"Yuklandi: {raw_df.shape[0]} mijoz, {raw_df.shape[1]} ustun")
        st.dataframe(raw_df.head())

        id_col = raw_df.columns[0]
        date_cols = [c for c in raw_df.columns if c != id_col]

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
        feats['weekly_std'] = consumption.std(axis=1)  # sodda variant

        feats = feats.fillna(0).replace([np.inf, -np.inf], 0)

        feature_order = [
            'mean_consumption', 'std_consumption', 'median_consumption',
            'min_consumption', 'max_consumption', 'cv', 'zero_ratio',
            'missing_ratio', 'q25', 'q75', 'iqr', 'mean_first_half',
            'mean_second_half', 'mean_daily_diff', 'max_daily_diff', 'weekly_std',
            'trend_diff'
        ]
        X_new = feats[feature_order]

        probas = model.predict_proba(X_new)[:, 1]
        results = pd.DataFrame({
            id_col: raw_df[id_col],
            'ogirlik_ehtimoli': probas,
            'bashorat': np.where(probas >= 0.5, "O'G'IRLIK", "Normal")
        }).sort_values('ogirlik_ehtimoli', ascending=False)

        st.subheader("Natijalar")
        st.dataframe(results)

        csv_out = results.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Natijani CSV sifatida yuklab olish", csv_out, "natijalar.csv", "text/csv")        