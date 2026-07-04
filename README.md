# 🏥 Insurance Charges Prediction

Sug'urta kompaniyasining mijozlar ma'lumotlari asosida tibbiy sug'urta to'lovlarini (`charges`) bashorat qiluvchi machine learning loyihasi.

## 📂 Loyiha tarkibi

```
├── insurance.csv          # Dataset
├── insurance_csv.ipynb    # Asosiy notebook (EDA + modellashtirish)
└── README.md
```

## 📊 Dataset haqida

Dataset quyidagi ustunlardan iborat:

| Ustun | Tavsif |
|-------|--------|
| `age` | Yoshi |
| `sex` | Jinsi (male/female) |
| `bmi` | Tana massa indeksi |
| `children` | Bolalar soni |
| `smoker` | Chekuvchi yoki yo'q (yes/no) |
| `region` | Yashash hududi |
| `charges` | Sug'urta to'lovi ($) — bashorat qilinadigan qiymat |

## ⚙️ Ishlatilgan texnologiyalar

- **Python** — pandas, numpy
- **Vizualizatsiya** — matplotlib, seaborn
- **ML** — scikit-learn (Linear/Ridge/Lasso Regression, Decision Tree, Random Forest, Gradient Boosting, KNN, SVR)

## 🔍 Ish jarayoni

1. **EDA** — ma'lumotlarni tahlil qilish, taqsimotlarni ko'rish, korrelyatsiya matritsasi
2. **Preprocessing** — kategorik ustunlarni one-hot encoding orqali raqamga aylantirish
3. **Train/Test split** — 80/20 nisbatda, scaling (SVR va KNN uchun)
4. **Modellarni solishtirish** — 8 xil regression modeli sinovdan o'tkazildi
5. **Eng yaxshi modelni tanlash va tahlil qilish** — Gradient Boosting
6. **Cross-validation** — 5-fold orqali modelni tekshirish
7. **Inference** — yangi mijoz uchun to'lovni bashorat qilish funksiyasi

## 🏆 Natijalar

| Model | R² | MAE ($) | RMSE ($) |
|-------|-----|---------|----------|
| **Gradient Boosting** | **0.8793** | **2,443** | **4,330** |
| Random Forest | 0.8651 | 2,550 | 4,576 |
| KNN | 0.8038 | 3,495 | 5,519 |
| Linear Regression | 0.7836 | 4,181 | 5,796 |
| Lasso Regression | 0.7835 | 4,182 | 5,797 |
| Ridge Regression | 0.7833 | 4,194 | 5,800 |
| Decision Tree | 0.7266 | 3,195 | 6,515 |

Sozlangan (tuned) Gradient Boosting modeli test to'plamida **R² = 0.8697** va 5-fold cross-validation'da o'rtacha **R² = 0.8463 ± 0.0389** natija ko'rsatdi.

### Eng muhim omillar (feature importance)
1. **smoker** (65.7%) — chekuvchilar sug'urtaga taxminan 3 baravar ko'p to'laydi
2. **bmi** (20.0%)
3. **age** (12.1%)
4. `children`, `region`, `sex` — ta'siri sezilarli darajada kichik

## 🚀 Ishga tushirish

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
jupyter notebook insurance_csv.ipynb
```

## 🔮 Bashorat misoli

```python
predict_charges(age=35, sex='male', bmi=27.5, children=2, smoker='no', region='southeast')
# → $8,169.22
```
