# football-injury-prediction

[README.md](https://github.com/user-attachments/files/27750389/README.md)
# ⚽ Football Player Injury Prediction & Readiness System

A machine learning system that predicts injury risk for football players based on physical and wellness metrics. Built with Random Forest (recall-optimised) and deployed as an interactive Streamlit dashboard.

## Demo

Three-tab dashboard:
- **🩺 Player Assessment** — enter daily metrics, get a traffic-light readiness result
- **📊 Model Performance** — confusion matrix, feature importance, classification report
- **ℹ️ Methodology** — system design and architecture overview

## Model Performance

| Metric | Score |
|--------|-------|
| Recall | 1.000 |
| Precision | 0.860 |
| F1 Score | 0.925 |
| ROC-AUC | 0.993 |

Optimised for **maximum recall** — a missed injury is always worse than an unnecessary rest day.

## Setup

```bash
git clone https://github.com/adlut/football-injury-prediction.git
cd football-injury-prediction
pip install -r requirements.txt
streamlit run app.py
```

Place `data.csv` in the same directory as `app.py` before running.

## Dataset

800 football players with 18 features covering:
- Physical profile (age, height, weight, position)
- Biomechanical tests (knee strength, hamstring flexibility, balance, agility, sprint speed)
- Wellness metrics (sleep, stress, nutrition, warmup adherence)
- Workload history (training hours, matches played, previous injuries)

**Target:** `Injury_Next_Season` (0 = no injury, 1 = injury)

## Feature Engineering

Three composite features were engineered:

| Feature | Description |
|---------|-------------|
| `Wellness_Score` | Normalised blend of sleep, stress (inverted), nutrition, warmup |
| `Physical_Resilience` | Mean of knee strength, flexibility, balance, agility |
| `Workload_Risk` | Training hrs/week × matches played last season |

## Tech Stack

- **ML:** scikit-learn (Random Forest)
- **UI:** Streamlit
- **Visualisation:** Matplotlib, Seaborn
- **Data:** Pandas, NumPy

## Project Structure

```
football-injury-prediction/
├── app.py              # Main Streamlit app + model logic
├── data.csv            # Player dataset
├── requirements.txt    # Python dependencies
└── README.md
```
