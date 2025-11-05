# streamlit_app.py – Extended Cultural Ethics Simulator

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import entropy, pearsonr

st.set_page_config(page_title="Ethics Sim", layout="wide")
st.title("🌍 Cultural AI Ethics Simulator (Extended)")

CULTURES = {
    "USA":     {"emotion": 0.3, "social": 0.1, "identity": 0.3, "moral": 0.3},
    "CHINA":   {"emotion": 0.1, "social": 0.5, "identity": 0.2, "moral": 0.2},
    "EUROPE":  {"emotion": 0.3, "social": 0.2, "identity": 0.2, "moral": 0.3},
    "KOREA":   {"emotion": 0.2, "social": 0.2, "identity": 0.4, "moral": 0.2},
    "LATIN_AM": {"emotion": 0.4, "social": 0.2, "identity": 0.2, "moral": 0.2},
    "MIDDLE_E": {"emotion": 0.1, "social": 0.2, "identity": 0.2, "moral": 0.5},
    "AFRICA":  {"emotion": 0.2, "social": 0.4, "identity": 0.2, "moral": 0.2},
}

scenario = st.sidebar.selectbox("시나리오", ["Classic Trolley", "Medical Triage", "AI Regulation"])
selected = st.sidebar.multiselect("문화권 선택", list(CULTURES.keys()), default=list(CULTURES.keys()))
steps = st.sidebar.slider("반복 수", 50, 500, 200, step=50)
manual = st.sidebar.checkbox("🎮 사용자 정의 가중치", False)

AGENTS = selected
AGENT_WEIGHTS = {}
for a in AGENTS:
    if manual:
        st.sidebar.markdown(f"**{a}**")
        w1 = st.sidebar.slider(f"{a} - Emotion", 0.0, 1.0, CULTURES[a]["emotion"])
        w2 = st.sidebar.slider(f"{a} - Social", 0.0, 1.0, CULTURES[a]["social"])
        w3 = st.sidebar.slider(f"{a} - Identity", 0.0, 1.0, CULTURES[a]["identity"])
        w4 = st.sidebar.slider(f"{a} - Moral", 0.0, 1.0, CULTURES[a]["moral"])
        total = sum([w1, w2, w3, w4])
        AGENT_WEIGHTS[a] = {"emotion": w1/total, "social": w2/total, "identity": w3/total, "moral": w4/total}
    else:
        AGENT_WEIGHTS[a] = dict(CULTURES[a])

AGENT_SCORES = {a: [] for a in AGENTS}
AGENT_HISTORY = {a: [dict(AGENT_WEIGHTS[a])] for a in AGENTS}
AGENT_ENTROPIES = {a: [] for a in AGENTS}
AGENT_MOVEMENT = {a: [] for a in AGENTS}
GROUP_DIVERGENCE = []
GROUP_AVG_REWARDS = []

@st.cache_data(show_spinner=False)
def simulate():
    for _ in range(steps):
        for a in AGENTS:
            prev = list(AGENT_WEIGHTS[a].values())
            rewards = np.random.rand(4)
            keys = list(AGENT_WEIGHTS[a].keys())
            score = sum(AGENT_WEIGHTS[a][k] * v for k, v in zip(keys, rewards))
            AGENT_SCORES[a].append(score)
            max_i, min_i = np.argmax(rewards), np.argmin(rewards)
            AGENT_WEIGHTS[a][keys[max_i]] += 0.05
            AGENT_WEIGHTS[a][keys[min_i]] -= 0.05
            total = sum(AGENT_WEIGHTS[a].values())
            for k in keys:
                AGENT_WEIGHTS[a][k] = max(0.001, AGENT_WEIGHTS[a][k]) / total
            curr = list(AGENT_WEIGHTS[a].values())
            AGENT_HISTORY[a].append(dict(AGENT_WEIGHTS[a]))
            AGENT_ENTROPIES[a].append(entropy(curr))
            AGENT_MOVEMENT[a].append(np.linalg.norm(np.array(curr) - np.array(prev)))
        mat = np.array([list(AGENT_WEIGHTS[a].values()) for a in AGENTS])
        GROUP_DIVERGENCE.append(np.mean(pdist(mat)))
        GROUP_AVG_REWARDS.append(np.mean([np.mean(AGENT_SCORES[a]) for a in AGENTS]))

if st.button("▶️ 시뮬레이션 시작"):
    simulate()

    st.subheader("📊 국가별 전략 궤적")
    for dim in ["emotion", "social", "identity", "moral"]:
        fig, ax = plt.subplots()
        for a in AGENT_HISTORY:
            ax.plot([w[dim] for w in AGENT_HISTORY[a]], label=a)
        ax.set_title(f"{dim.capitalize()} Weight")
        ax.legend(); st.pyplot(fig)

    st.subheader("📈 전략 엔트로피 / 이동량")
    fig1, ax1 = plt.subplots()
    for a in AGENT_ENTROPIES:
        ax1.plot(AGENT_ENTROPIES[a], label=a)
    ax1.set_title("Entropy of Strategy Distribution")
    ax1.legend(); st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    for a in AGENT_MOVEMENT:
        ax2.plot(np.cumsum(AGENT_MOVEMENT[a]), label=a)
    ax2.set_title("Cumulative Strategic Change")
    ax2.legend(); st.pyplot(fig2)

    st.subheader("📉 전략 다양성과 평균 보상")
    fig3, ax3 = plt.subplots()
    ax3.plot(GROUP_DIVERGENCE, label="Divergence")
    ax3.set_title("Group Ethical Divergence")
    ax3.legend(); st.pyplot(fig3)

    fig4, ax4 = plt.subplots()
    ax4.scatter(GROUP_DIVERGENCE, GROUP_AVG_REWARDS)
    r, p = pearsonr(GROUP_DIVERGENCE, GROUP_AVG_REWARDS)
    ax4.set_title(f"Divergence vs Avg Reward (r={r:.2f}, p={p:.3f})")
    st.pyplot(fig4)

    st.subheader("📄 국가별 최종 전략")
    df = pd.DataFrame([{"Agent": a, **AGENT_HISTORY[a][-1]} for a in AGENTS])
    st.dataframe(df.set_index("Agent"))

    st.download_button("📥 전략 데이터 저장 (CSV)", data=df.to_csv(index=False), file_name="final_strategies.csv")
