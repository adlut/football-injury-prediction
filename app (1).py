# Football Player Injury Prediction & Readiness System
# Model: Random Forest Classifier (recall-optimised)
# Run:   streamlit run app.py

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, classification_report,
    recall_score, precision_score, f1_score, roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder
import os

st.set_page_config(
    page_title="⚽ Injury Prediction System",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

GREEN  = "#2ECC71"
YELLOW = "#F39C12"
RED    = "#E74C3C"
DARK   = "#1A1A2E"
CARD   = "#16213E"
ACCENT = "#0F3460"


# --- Data Loading & Feature Engineering ---

@st.cache_data
def load_and_preprocess(csv_path="data.csv"):
    df = pd.read_csv(csv_path)

    le = LabelEncoder()
    df["Position_enc"] = le.fit_transform(df["Position"])
    position_classes = le.classes_
    df.drop(columns=["BMI", "Position"], inplace=True)

    # Composite wellness score (sleep, stress, nutrition, warmup)
    sleep_norm = (df["Sleep_Hours_Per_Night"] / 10) * 100
    stress_inv = 100 - df["Stress_Level_Score"]
    df["Wellness_Score"] = (
        sleep_norm + stress_inv +
        df["Nutrition_Quality_Score"] +
        df["Warmup_Routine_Adherence"] * 10
    ) / 4

    # Physical readiness: mean of key strength/flexibility test scores
    df["Physical_Resilience"] = df[[
        "Knee_Strength_Score", "Hamstring_Flexibility",
        "Balance_Test_Score",  "Agility_Score",
    ]].mean(axis=1)

    # Chronic workload proxy
    df["Workload_Risk"] = df["Training_Hours_Per_Week"] * df["Matches_Played_Past_Season"]

    y = df["Injury_Next_Season"]
    X = df.drop(columns=["Injury_Next_Season"])
    return X, y, X.columns.tolist(), position_classes, df


# --- Model Training ---

@st.cache_resource
def train_model(csv_path="data.csv"):
    X, y, feature_names, position_classes, df = load_and_preprocess(csv_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # class_weight='balanced' + lowered threshold = recall-first strategy
    # A missed injury (False Negative) is always worse than unnecessary rest (False Positive)
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    THRESHOLD = 0.35
    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= THRESHOLD).astype(int)

    metrics = {
        "recall":    recall_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "f1":        f1_score(y_test, y_pred),
        "roc_auc":   roc_auc_score(y_test, y_prob),
        "threshold": THRESHOLD,
    }
    return clf, X_test, y_test, y_pred, y_prob, metrics, feature_names, position_classes


# --- Visualisations ---

def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor(DARK)
    ax.set_facecolor(DARK)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="YlOrRd",
        linewidths=2, linecolor=DARK, ax=ax,
        xticklabels=["Ready (0)", "High Risk (1)"],
        yticklabels=["Ready (0)", "High Risk (1)"],
        annot_kws={"size": 16, "weight": "bold"},
    )
    ax.set_xlabel("Predicted", color="white", fontsize=12)
    ax.set_ylabel("Actual",    color="white", fontsize=12)
    ax.set_title("Confusion Matrix", color="white", fontsize=14, pad=12)
    ax.tick_params(colors="white")
    plt.tight_layout()
    return fig


def plot_feature_importance(clf, feature_names, top_n=15):
    importances = clf.feature_importances_
    indices     = np.argsort(importances)[-top_n:]
    feat_labels = [feature_names[i] for i in indices]
    vals        = importances[indices]
    colours     = [RED if v > 0.08 else YELLOW if v > 0.04 else "#4ECDC4" for v in vals]

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor(DARK)
    ax.set_facecolor(DARK)
    bars = ax.barh(feat_labels, vals, color=colours, edgecolor="none", height=0.6)
    ax.set_xlabel("Importance Score", color="white", fontsize=11)
    ax.set_title(f"Top {top_n} Feature Importances", color="white", fontsize=14, pad=12)
    ax.tick_params(colors="white", labelsize=10)
    ax.spines[:].set_visible(False)
    ax.set_xlim(0, vals.max() * 1.2)
    for bar, val in zip(bars, vals):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", color="white", fontsize=9)
    legend_elements = [
        mpatches.Patch(color=RED,       label="High  (>0.08)"),
        mpatches.Patch(color=YELLOW,    label="Medium (0.04–0.08)"),
        mpatches.Patch(color="#4ECDC4", label="Low  (<0.04)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right",
              facecolor=CARD, labelcolor="white", fontsize=9)
    plt.tight_layout()
    return fig


def traffic_light_html(label, colour, prob):
    return f"""
    <div style="background:{CARD}; border-radius:16px; padding:28px 24px;
                text-align:center; border:2px solid {colour};
                box-shadow:0 0 20px {colour}55;">
      <div style="font-size:64px; margin-bottom:8px;">
        {'🟢' if colour==GREEN else '🟡' if colour==YELLOW else '🔴'}
      </div>
      <div style="font-size:28px; font-weight:800; color:{colour}; letter-spacing:1px;">
        {label}
      </div>
      <div style="font-size:16px; color:#aaa; margin-top:8px;">
        Injury probability: <b style="color:{colour}">{prob:.1%}</b>
      </div>
    </div>
    """


# --- Main App ---

def main():
    csv_path = "data.csv"
    if not os.path.exists(csv_path):
        st.error("data.csv not found. Place it in the same directory as app.py.")
        return

    clf, X_test, y_test, y_pred, y_prob, metrics, feature_names, pos_classes = \
        train_model(csv_path)

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,{DARK},{ACCENT});
                padding:32px; border-radius:16px; margin-bottom:24px;
                border-left:5px solid {GREEN};">
      <h1 style="color:white; margin:0; font-size:2.2rem;">
        ⚽ Football Injury Prediction & Readiness System
      </h1>
      <p style="color:#aaa; margin:8px 0 0 0; font-size:1rem;">
        ML-powered daily readiness assessment · Random Forest · Recall-optimised
      </p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(
        ["🩺 Player Assessment", "📊 Model Performance", "ℹ️ Methodology"]
    )

    with tab1:
        st.markdown("### Enter today's player metrics")
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**🏃 Physical Profile**")
            age      = st.slider("Age (years)",      18, 35, 21)
            height   = st.slider("Height (cm)",      155, 200, 177)
            weight   = st.slider("Weight (kg)",      55, 100, 75)
            position = st.selectbox("Position", list(pos_classes))

        with c2:
            st.markdown("**💪 Physical Tests**")
            knee_str = st.slider("Knee Strength Score",    40.0, 100.0, 75.0)
            ham_flex = st.slider("Hamstring Flexibility",  40.0, 100.0, 75.0)
            balance  = st.slider("Balance Test Score",     40.0, 100.0, 80.0)
            agility  = st.slider("Agility Score",          40.0, 100.0, 78.0)
            sprint   = st.slider("Sprint Speed 10m (s)",   4.5,   7.5,  5.8)
            reaction = st.slider("Reaction Time (ms)",    150.0, 400.0, 260.0)

        with c3:
            st.markdown("**😴 Wellness & Workload**")
            sleep     = st.slider("Sleep Hours / Night",       4.0, 10.0, 7.5)
            stress    = st.slider("Stress Level (0-100)",      0.0, 100.0, 40.0)
            nutrition = st.slider("Nutrition Quality (0-100)", 0.0, 100.0, 70.0)
            warmup    = st.slider("Warmup Adherence (0-10)",   0, 10, 7)
            train_hrs = st.slider("Training hrs / Week",       4.0, 20.0, 11.0)
            matches   = st.slider("Matches Last Season",       0, 50, 25)
            prev_inj  = st.slider("Previous Injury Count",     0,  6,  1)

        st.markdown("---")
        if st.button("🔍  Run Injury Risk Assessment", use_container_width=True):
            pos_enc       = list(pos_classes).index(position)
            sleep_norm    = (sleep / 10) * 100
            wellness      = (sleep_norm + (100 - stress) + nutrition + warmup * 10) / 4
            resilience    = (knee_str + ham_flex + balance + agility) / 4
            workload_risk = train_hrs * matches

            input_dict = {
                "Age": age, "Height_cm": height, "Weight_kg": weight,
                "Training_Hours_Per_Week": train_hrs,
                "Matches_Played_Past_Season": matches,
                "Previous_Injury_Count": prev_inj,
                "Knee_Strength_Score": knee_str, "Hamstring_Flexibility": ham_flex,
                "Reaction_Time_ms": reaction,   "Balance_Test_Score": balance,
                "Sprint_Speed_10m_s": sprint,   "Agility_Score": agility,
                "Sleep_Hours_Per_Night": sleep, "Stress_Level_Score": stress,
                "Nutrition_Quality_Score": nutrition, "Warmup_Routine_Adherence": warmup,
                "Position_enc": pos_enc, "Wellness_Score": wellness,
                "Physical_Resilience": resilience, "Workload_Risk": workload_risk,
            }
            input_df = pd.DataFrame([input_dict])[feature_names]
            prob = clf.predict_proba(input_df)[0, 1]

            if prob < metrics["threshold"]:
                colour, label = GREEN,  "READY TO TRAIN"
            elif prob < 0.55:
                colour, label = YELLOW, "CAUTION - Monitor Closely"
            else:
                colour, label = RED,    "HIGH RISK - REST RECOMMENDED"

            _, r2, _ = st.columns([1, 1.5, 1])
            with r2:
                st.markdown(traffic_light_html(label, colour, prob), unsafe_allow_html=True)

            st.markdown("#### Risk Factor Breakdown")
            factors = {
                "Wellness Score":      (wellness,       60,  True),
                "Physical Resilience": (resilience,     65,  True),
                "Sleep Hours":         (sleep,           7,  True),
                "Stress Level":        (stress,          50, False),
                "Previous Injuries":   (prev_inj,        2,  False),
                "Workload Risk Index": (workload_risk,  300, False),
            }
            cols = st.columns(3)
            for idx, (name, (val, thr, good_high)) in enumerate(factors.items()):
                risky = (val < thr) if good_high else (val > thr)
                icon  = "🔴" if risky else "🟢"
                cols[idx % 3].metric(f"{icon} {name}", f"{val:.1f}",
                                     delta="Risk factor" if risky else "OK",
                                     delta_color="inverse" if risky else "normal")

            st.markdown("#### Recommendation")
            if colour == GREEN:
                st.success("Player appears ready. Proceed with planned training session.")
            elif colour == YELLOW:
                st.warning("Elevated risk. Consider reduced-intensity session and re-assess tomorrow.")
            else:
                st.error("High injury risk. Recommend full rest or physio-only work.")

    with tab2:
        st.markdown("### Model Performance Report")
        m1, m2, m3, m4 = st.columns(4)
        kpi = f"background:{CARD}; border-radius:12px; padding:20px; text-align:center;"
        for col, name, val, clr in [
            (m1, "Recall",    metrics["recall"],    GREEN),
            (m2, "Precision", metrics["precision"], YELLOW),
            (m3, "F1 Score",  metrics["f1"],        "#4ECDC4"),
            (m4, "ROC-AUC",   metrics["roc_auc"],   "#9B59B6"),
        ]:
            col.markdown(
                f'<div style="{kpi} border:2px solid {clr};">'
                f'<div style="font-size:2rem;font-weight:800;color:{clr}">{val:.3f}</div>'
                f'<div style="color:#aaa;font-size:.9rem">{name}</div></div>',
                unsafe_allow_html=True,
            )

        st.caption(f"Decision threshold: {metrics['threshold']} — lowered from 0.5 to maximise recall.")
        st.markdown("---")

        v1, v2 = st.columns(2)
        with v1:
            st.markdown("#### Confusion Matrix")
            st.pyplot(plot_confusion_matrix(y_test, y_pred))
        with v2:
            st.markdown("#### Feature Importance")
            st.pyplot(plot_feature_importance(clf, feature_names))

        st.markdown("#### Full Classification Report")
        report = classification_report(y_test, y_pred,
                                       target_names=["Ready (0)", "High Risk (1)"],
                                       output_dict=True)
        st.dataframe(
            pd.DataFrame(report).T.round(3).style.background_gradient(cmap="YlOrRd"),
            use_container_width=True,
        )

    with tab3:
        st.markdown("### Methodology & System Design")
        st.markdown("""
#### Problem Statement
Football players who train through soreness or poor wellness face elevated injury risk.
This system provides daily readiness assessment using an ML model trained on player
monitoring data, enabling coaches to make data-driven decisions before each session.

---
#### Dataset
| Attribute | Value |
|-----------|-------|
| Records | 800 players |
| Raw features | 18 |
| Engineered features | 3 |
| Target | Injury_Next_Season (binary) |
| Class balance | 50 / 50 |

---
#### Feature Engineering
| Feature | Formula | Rationale |
|---------|---------|-----------|
| Wellness_Score | (sleep_norm + stress_inv + nutrition + warmup x10) / 4 | Daily readiness proxy |
| Physical_Resilience | mean(knee, hamstring, balance, agility) | Muscular readiness |
| Workload_Risk | Training hrs/wk x Matches last season | Chronic load exposure |

---
#### Model & Recall Optimisation
Random Forest was selected for robustness to mixed feature types and native feature importance.
Two mechanisms maximise recall:
1. class_weight='balanced' — upweights the injured class during training
2. Decision threshold = 0.35 — flags High Risk at lower probability than default 0.50

A False Negative (missed injury) causes real harm. A False Positive (unnecessary rest) costs one session.

---
#### System Architecture
```
data.csv -> Preprocessing -> Feature Engineering -> Train/Test Split (80/20)
  -> Random Forest (balanced) -> Threshold Calibration (0.35)
  -> prob < 0.35: READY | 0.35-0.55: CAUTION | >0.55: HIGH RISK
  -> Streamlit Dashboard
```

---
#### Production Notes
- Coaching staff enter morning wellness values; output appears in under 10 seconds.
- Model should be retrained each season with updated player data.
- Integrating GPS, sRPE, and daily soreness logs would further improve accuracy.
        """)


if __name__ == "__main__":
    main()
