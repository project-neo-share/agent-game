# streamlit_app.py – Cultural Scenario-Based AI Ethics Simulator

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.stats import entropy, pearsonr

st.set_page_config(page_title="Ethical Divergence Simulator", layout="wide")
st.title("🌐 Cultural Scenario-Based AI Ethics Simulator")

# ------------------------- Cultural Presets -------------------------
CULTURES = {
    "USA":     {"emotion": 0.3, "social": 0.1, "identity": 0.3, "moral": 0.3},
    "CHINA":   {"emotion": 0.1, "social": 0.5, "identity": 0.2, "moral": 0.2},
    "EUROPE":  {"emotion": 0.3, "social": 0.2, "identity": 0.2, "moral": 0.3},
    "KOREA":   {"emotion": 0.2, "social": 0.2, "identity": 0.4, "moral": 0.2},
    "LATIN_AM":{"emotion": 0.4, "social": 0.2, "identity": 0.2, "moral": 0.2},
    "MIDDLE_E":{"emotion": 0.1, "social": 0.2, "identity": 0.2, "moral": 0.5},
    "AFRICA":  {"emotion": 0.2, "social": 0.4, "identity": 0.2, "moral": 0.2},
}

# ------------------------- Sidebar Configuration -------------------------
st.sidebar.header("🧭 시뮬레이션 설정")
scenario = st.sidebar.selectbox("시나리오 선택", options=["Classic Trolley", "Medical Triage", "Self-driving Dilemma"])
selected_cultures = st.sidebar.multiselect("분석할 문화권 선택", options=list(CULTURES.keys()), default=list(CULTURES.keys()))
sim_steps = st.sidebar.slider("시뮬레이션 반복 수", 50, 500, 200, step=50)

st.sidebar.markdown("---")

# ------------------------- Simulation State -------------------------
AGENTS = selected_cultures
AGENT_WEIGHTS = {a: dict(CULTURES[a]) for a in AGENTS}
AGENT_SCORES = {a: [] for a in AGENTS}
AGENT_WEIGHT_HISTORY = {a: [dict(AGENT_WEIGHTS[a])] for a in AGENTS}
AGENT_ENTROPIES = {a: [] for a in AGENTS}
AGENT_MOVEMENT = {a: [] for a in AGENTS}
GROUP_DIVERGENCE = []
GROUP_AVG_REWARDS = []

# ------------------------- Simulation Engine -------------------------
def run_simulation():
    for _ in range(sim_steps):
        for agent in AGENTS:
            prev = list(AGENT_WEIGHTS[agent].values())
            rewards = np.random.rand(4)
            keys = list(AGENT_WEIGHTS[agent].keys())
            score = sum(AGENT_WEIGHTS[agent][k]*r for k, r in zip(keys, rewards))
            AGENT_SCORES[agent].append(score)
            max_idx, min_idx = np.argmax(rewards), np.argmin(rewards)
            AGENT_WEIGHTS[agent][keys[max_idx]] += 0.05
            AGENT_WEIGHTS[agent][keys[min_idx]] -= 0.05
            total = sum(AGENT_WEIGHTS[agent].values())
            for k in keys:
                AGENT_WEIGHTS[agent][k] = max(0.001, AGENT_WEIGHTS[agent][k]) / total
            curr = list(AGENT_WEIGHTS[agent].values())
            AGENT_WEIGHT_HISTORY[agent].append(dict(AGENT_WEIGHTS[agent]))
            AGENT_ENTROPIES[agent].append(entropy(curr))
            AGENT_MOVEMENT[agent].append(np.linalg.norm(np.array(curr) - np.array(prev)))
        mat = np.array([list(AGENT_WEIGHTS[a].values()) for a in AGENTS])
        GROUP_DIVERGENCE.append(np.mean(pdist(mat)))
        GROUP_AVG_REWARDS.append(np.mean([np.mean(AGENT_SCORES[a]) for a in AGENTS]))

# ------------------------- Visualization -------------------------
def plot_trajectories():
    st.header("📊 Figure 1. 전략 궤적 (Strategic Trajectories)")
    for dim in ["emotion", "social", "identity", "moral"]:
        fig, ax = plt.subplots(figsize=(10, 3))
        for a in AGENT_WEIGHT_HISTORY:
            values = [w[dim] for w in AGENT_WEIGHT_HISTORY[a]]
            ax.plot(values, label=a)
        ax.set_title(f"{dim.capitalize()} Weight Trajectory")
        ax.set_xlabel("Step")
        ax.set_ylabel("Weight")
        ax.legend()
        st.pyplot(fig)

def plot_entropy_movement():
    st.header("📈 Figure 2. 엔트로피 및 전략 이동량")
    fig1, ax1 = plt.subplots()
    for a in AGENT_ENTROPIES:
        ax1.plot(AGENT_ENTROPIES[a], label=a)
    ax1.set_title("Entropy of Strategy Distribution")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Entropy")
    ax1.legend()
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    for a in AGENT_MOVEMENT:
        ax2.plot(np.cumsum(AGENT_MOVEMENT[a]), label=a)
    ax2.set_title("Cumulative Strategic Change Distance")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Distance")
    ax2.legend()
    st.pyplot(fig2)

def plot_summary():
    st.header("📉 Figure 3. 집단 분산 및 보상 상관")
    fig3, ax3 = plt.subplots()
    ax3.plot(GROUP_DIVERGENCE, label="Ethical Divergence")
    ax3.set_title("Group Ethical Divergence Over Time")
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Avg Pairwise Distance")
    ax3.legend()
    st.pyplot(fig3)

    fig4, ax4 = plt.subplots()
    ax4.scatter(GROUP_DIVERGENCE, GROUP_AVG_REWARDS)
    r, p = pearsonr(GROUP_DIVERGENCE, GROUP_AVG_REWARDS)
    ax4.set_title("Divergence vs Avg Reward")
    ax4.set_xlabel("Divergence")
    ax4.set_ylabel("Avg Reward")
    ax4.text(min(GROUP_DIVERGENCE), max(GROUP_AVG_REWARDS), f"r = {r:.2f}, p = {p:.3f}")
    st.pyplot(fig4)

# ------------------------- App Launch -------------------------
if st.button("▶️ 시뮬레이션 실행"):
    run_simulation()
    plot_trajectories()
    plot_entropy_movement()
    plot_summary()
    st.success("시뮬레이션 완료: 국가별 전략 궤적 및 집단 동역학 분석 완료")
