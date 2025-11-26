# app.py — Cultural AI Ethics: Single Culture & Scenario Config
# 작성자: Prof. Songhee Kang
# Update: Restored Scenario Reward Config + Diversity-Reward Correlation

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from scipy.stats import pearsonr
from dataclasses import dataclass
from typing import Dict, List

# ==================== 설정 ====================
st.set_page_config(page_title="AI Ethics: Environment & Agent", page_icon="🎛️", layout="wide")

# ==================== 데이터 모델 ====================
@dataclass
class Scenario:
    sid: str
    title: str
    setup: str
    options: Dict[str, str]
    rewards: Dict[str, Dict[str, float]]

# 기본 시나리오 데이터 (Default Presets)
DEFAULT_SCENARIOS = [
    Scenario(
        sid="S1", title="1단계: 고전적 트롤리",
        setup="선로 위 5명 vs 1명. 레버를 당길 것인가?",
        options={"A": "1명 희생 (개입)", "B": "방관 (현상 유지)"},
        rewards={"A": {"emotion": 1.0, "social": -0.5, "moral": -1.0, "identity": 0.5},
                 "B": {"emotion": -1.0, "social": 0.5, "moral": 1.0, "identity": -0.5}}
    ),
    Scenario(
        sid="S2", title="2단계: 맥락적 요소",
        setup="무단 침입자 5명 vs 관리자 자녀 1명.",
        options={"A": "5명 구조 (자녀 희생)", "B": "규정 준수 (5명 방관)"},
        rewards={"A": {"emotion": 0.6, "social": -0.8, "moral": -0.7, "identity": 0.3},
                 "B": {"emotion": -0.5, "social": 0.9, "moral": 0.6, "identity": 0.4}}
    ),
    Scenario(
        sid="S3", title="3단계: 의료 재난 분류",
        setup="일반 부상자 다수 vs 숙련된 의사 1명.",
        options={"A": "의사 우선 (공리주의)", "B": "동등 대우 (평등주의)"},
        rewards={"A": {"emotion": 0.7, "social": -0.4, "moral": -0.6, "identity": 0.8},
                 "B": {"emotion": -0.3, "social": 0.7, "moral": 0.9, "identity": 0.5}}
    ),
]

FRAMEWORKS = ["emotion", "social", "moral", "identity"]

# 문화권 프리셋
CULTURES_PRESETS = {
    "USA":      {"emotion": 0.3, "social": 0.1, "identity": 0.3, "moral": 0.3},
    "CHINA":    {"emotion": 0.1, "social": 0.5, "identity": 0.2, "moral": 0.2},
    "EUROPE":   {"emotion": 0.3, "social": 0.2, "identity": 0.2, "moral": 0.3},
    "KOREA":    {"emotion": 0.2, "social": 0.2, "identity": 0.4, "moral": 0.2},
    "LATIN_AM": {"emotion": 0.4, "social": 0.2, "identity": 0.2, "moral": 0.2},
    "MIDDLE_E": {"emotion": 0.1, "social": 0.2, "identity": 0.2, "moral": 0.5},
    "AFRICA":   {"emotion": 0.2, "social": 0.4, "identity": 0.2, "moral": 0.2},
}

# ==================== 강화학습 에이전트 ====================
class QLearningAgent:
    def __init__(self, name, weights, scenarios, learning_rate=0.1, epsilon=0.5):
        self.name = name
        self.weights = weights      # 문화권 가중치 (Agent Internal)
        self.scenarios = scenarios  # 시나리오 보상 환경 (Environment External)
        self.lr = learning_rate
        self.epsilon = epsilon
        self.q_table = {s.sid: {"A": 0.0, "B": 0.0} for s in scenarios}
        
    def get_action(self, sid):
        # Epsilon-Greedy Strategy
        if random.random() < self.epsilon:
            return random.choice(["A", "B"])
        qs = self.q_table[sid]
        if qs["A"] > qs["B"]: return "A"
        elif qs["B"] > qs["A"]: return "B"
        return random.choice(["A", "B"])

    def calculate_reward(self, sid, action):
        # 핵심 로직: 시나리오가 주는 보상 벡터(Env)와 내 가치관(Agent)의 내적
        scn = next(s for s in self.scenarios if s.sid == sid)
        r_vec = scn.rewards[action]
        reward = sum(r_vec.get(k, 0) * self.weights.get(k, 0) for k in FRAMEWORKS) * 10
        return reward

    def update(self, sid, action, reward):
        old_q = self.q_table[sid][action]
        self.q_table[sid][action] = old_q + self.lr * (reward - old_q)

    def decay_epsilon(self):
        self.epsilon = max(0.01, self.epsilon * 0.99)

# ==================== 분석 함수 ====================
def calculate_diversity(actions_list: List[str]) -> float:
    """행동 다양성 계산 (1.0 = A/B 균형, 0.0 = 한쪽 쏠림)"""
    if not actions_list: return 0.0
    a_count = actions_list.count("A")
    ratio = a_count / len(actions_list)
    return 1.0 - (2 * abs(0.5 - ratio))

def run_simulation(culture_name, weights, episodes, custom_scenarios):
    agent = QLearningAgent(culture_name, weights, custom_scenarios)
    
    history = {
        "episode": [],
        "reward": [],
        "diversity": []
    }
    
    progress = st.progress(0)
    
    for ep in range(episodes):
        ep_actions = []
        ep_reward = 0
        
        for scn in custom_scenarios:
            action = agent.get_action(scn.sid)
            reward = agent.calculate_reward(scn.sid, action)
            agent.update(scn.sid, action, reward)
            
            ep_actions.append(action)
            ep_reward += reward
            
        agent.decay_epsilon()
        
        history["episode"].append(ep + 1)
        history["reward"].append(ep_reward)
        history["diversity"].append(calculate_diversity(ep_actions))
        
        if (ep + 1) % 10 == 0:
            progress.progress((ep + 1) / episodes)
            
    progress.empty()
    return pd.DataFrame(history)

# ==================== UI 구성 ====================
st.title("🎛️ AI Ethics Simulation: Config & Analysis")
st.markdown("""
**1단계 (환경 설정):** 각 시나리오의 선택지가 주는 보상(Reward Vector)을 설정합니다.<br>
**2단계 (에이전트 설정):** 특정 문화권의 가치관 가중치(Weights)를 설정합니다.<br>
**3단계 (분석):** 행동 다양성과 보상 간의 상관관계를 확인합니다.
""", unsafe_allow_html=True)

# --- [사이드바] 에이전트(문화권) 설정 ---
st.sidebar.header("👤 2. Agent (Culture) Setup")
selected_culture = st.sidebar.selectbox("문화권 프리셋 선택", list(CULTURES_PRESETS.keys()), index=3)
episodes = st.sidebar.slider("학습 횟수 (Episodes)", 100, 1000, 300, step=50)

st.sidebar.subheader("가치관 가중치 미세조정")
st.sidebar.caption("문화권의 기본 성향을 수정할 수 있습니다.")
culture_weights = CULTURES_PRESETS[selected_culture].copy()

# 사이드바에서 가중치 조정 UI
mod_weights = {}
for k in FRAMEWORKS:
    mod_weights[k] = st.sidebar.slider(f"{k.capitalize()}", 0.0, 1.0, culture_weights[k])

# 가중치 정규화 (합이 1이 되도록)
total_w = sum(mod_weights.values()) or 1
final_weights = {k: v/total_w for k, v in mod_weights.items()}

st.sidebar.markdown("---")
st.sidebar.write("📊 **적용된 가중치:**")
st.sidebar.json(final_weights)

# --- [메인 화면] 시나리오 보상 벡터 설정 (복구된 기능) ---
st.header("🌍 1. Environment (Scenario) Setup")
st.info("각 시나리오에서 A/B 선택지가 주는 보상(성격)을 정의합니다. (-1.0: 부정적 ~ 1.0: 긍정적)")

custom_scenarios = []

# 시나리오 루프 (3개)
cols = st.columns(3) # 가로로 배치
for i, default_scn in enumerate(DEFAULT_SCENARIOS):
    with cols[i]:
        with st.expander(f"📝 {default_scn.title}", expanded=True):
            st.caption(default_scn.setup)
            
            # Option A 설정
            st.markdown(f"**🅰 {default_scn.options['A']}**")
            r_a = default_scn.rewards["A"].copy()
            # 공간 절약을 위해 Emotion과 Moral만 예시로 표시 (필요시 추가 가능)
            r_a["emotion"] = st.slider(f"S{i+1}-A Emotion", -1.0, 1.0, r_a["emotion"], key=f"s{i}a_em")
            r_a["moral"] = st.slider(f"S{i+1}-A Moral", -1.0, 1.0, r_a["moral"], key=f"s{i}a_mo")
            
            # Option B 설정
            st.markdown(f"**🅱 {default_scn.options['B']}**")
            r_b = default_scn.rewards["B"].copy()
            r_b["emotion"] = st.slider(f"S{i+1}-B Emotion", -1.0, 1.0, r_b["emotion"], key=f"s{i}b_em")
            r_b["moral"] = st.slider(f"S{i+1}-B Moral", -1.0, 1.0, r_b["moral"], key=f"s{i}b_mo")
            
            # 나머지 값들은 기본값 유지하면서 커스텀 시나리오 객체 생성
            custom_scenarios.append(Scenario(
                default_scn.sid, default_scn.title, default_scn.setup, 
                default_scn.options, {"A": r_a, "B": r_b}
            ))

# --- [분석 실행] ---
st.divider()
st.header("🚀 3. Simulation & Analysis")

if st.button("시뮬레이션 시작 (Run Analysis)", type="primary"):
    with st.spinner(f"'{selected_culture}' 에이전트가 커스텀 환경에서 학습 중..."):
        df = run_simulation(selected_culture, final_weights, episodes, custom_scenarios)
    
    st.success("분석 완료!")
    
    # 1. 그래프 영역 (학습 곡선 & 다양성)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📈 Reward Curve")
        st.line_chart(df, x="episode", y="reward", color="#FF4B4B")
    with c2:
        st.subheader("🔀 Diversity Curve")
        st.line_chart(df, x="episode", y="diversity", color="#1F77B4")
        
    # 2. 상관관계 분석 영역
    st.markdown("---")
    st.subheader("🔗 Correlation: Diversity vs Reward")
    
    # 피어슨 상관계수
    r_val, p_val = pearsonr(df["diversity"], df["reward"])
    
    col_plot, col_stat = st.columns([2, 1])
    
    with col_plot:
        # Scatter Plot
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.scatter(df["diversity"], df["reward"], alpha=0.6, c='purple', edgecolors='w')
        
        # 추세선 추가
        z = np.polyfit(df["diversity"], df["reward"], 1)
        p = np.poly1d(z)
        ax.plot(df["diversity"], p(df["diversity"]), "r--", alpha=0.8, label="Trend")
        
        ax.set_xlabel("Behavioral Diversity (0=Rigid, 1=Flexible)")
        ax.set_ylabel("Total Reward")
        ax.set_title(f"Diversity vs Reward (r={r_val:.2f})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
    with col_stat:
        st.markdown("### 📊 통계 요약")
        st.metric("상관계수 (Pearson r)", f"{r_val:.3f}")
        st.metric("P-value", f"{p_val:.3e}")
        
        st.markdown("#### 해석")
        if r_val > 0.3:
            st.success("✅ **양의 상관관계**\n\n다양한 전략을 시도할수록 보상이 높아지는 환경입니다.")
        elif r_val < -0.3:
            st.warning("⚠️ **음의 상관관계**\n\n특정 행동(규칙)을 고수해야 보상이 높은 환경입니다.")
        else:
            st.info("⏺ **상관없음**\n\n다양성과 보상 간에 뚜렷한 관계가 없습니다.")

    # 데이터 다운로드
    with st.expander("📥 로우 데이터 다운로드"):
        st.dataframe(df.head())
        st.download_button("CSV Save", df.to_csv(index=False), "ethics_sim_data.csv")
