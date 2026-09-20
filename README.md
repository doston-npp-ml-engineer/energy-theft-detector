# ⚡ Elektr Energiyasi O'g'irligini Aniqlash Tizimi

XGBoost va early stopping texnikasi asosida qurilgan, elektr energiyasi o'g'irligini (non-technical loss) aniqlaydigan machine learning loyihasi.

## 📊 Dataset

**SGCC (State Grid Corporation of China) Electricity Theft Detection**
- 42,372 mijozning kunlik elektr iste'moli (2014-2016)
- Binary classification: normal iste'mol (0) vs o'g'irlik (1)
- Nomutanosib dataset: ~8.5% theft holatlari

## 🧠 Model

- **Algoritm:** XGBoost Classifier
- **Early stopping:** validation to'plamdagi `aucpr` (Precision-Recall AUC) metrikasi bo'yicha, overfitting'ning oldini olish uchun
- **Imbalance bilan ishlash:** `scale_pos_weight` parametri orqali
- **Feature engineering:** xom kunlik iste'mol ma'lumotlaridan 17 ta statistik feature (o'rtacha, standart og'ish, kvartillar, trend, haftalik barqarorlik va h.k.) hisoblab olindi

### Natijalar
- ROC-AUC: ~0.79
- PR-AUC: ~0.34

## 🖥️ Ilova (Streamlit)

Ikki xil kiritish rejimi:
1. **Qo'lda kiritish** — 17 ta statistik ko'rsatkichni qo'lda kiritib, bitta mijoz uchun bashorat olish
2. **CSV yuklash** — bir nechta mijozning xom kunlik iste'mol ma'lumotlarini yuklab, ommaviy bashorat olish va natijani CSV sifatida yuklab olish

## 🚀 Ishga tushirish

\`\`\`bash
git clone https://github.com/SIZNING_USERNAME/energy-theft-detector.git
cd energy-theft-detector
python -m venv venv
venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
\`\`\`

## 🛠️ Texnologiyalar

Python · XGBoost · Streamlit · Pandas · Scikit-learn
