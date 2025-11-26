# app.py — Cultural AI Ethics: Intra-Cultural Correlation Analysis
# 작성자: Prof. Songhee Kang
# Update: Single Culture Focus, Diversity-Reward Correlation

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from scipy.stats import pearsonr
from dataclasses import dataclass
from typing import Dict, List

# ==================== 설정 ====================
st.set_page_config(page_title="AI Ethics: Culture & Correlation", page_icon="🔬", layout="wide")

# ==================== 데이터 모델 ====================
@dataclass
class Scenario:
    sid: str
    title: str
    setup: str
    options: Dict[str, str]
    rewards: Dict[str, Dict[str, float]]

# 기본 시나리오
SCENARIOS = [
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
        self.weights = weights
        self.scenarios = scenarios
        self.lr = learning_rate
        self.epsilon = epsilon
        self.q_table = {s.sid: {"A": 0.0, "B": 0.0} for s in scenarios}
        
    def get_action(self, sid):
        if random.random() < self.epsilon:
            return random.choice(["A", "B"])
        qs = self.q_table[sid]
        return "A" if qs["A"] > qs["B"] else "B"

    def calculate_reward(self, sid, action):
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
    """행동 다양성 계산 (0.0: 획일적 ~ 1.0: 완전 균형)"""
    if not actions_list: return 0.0
    a_count = actions_list.count("A")
    ratio = a_count / len(actions_list)
    # 0.5(반반)일 때 1.0, 0 또는 1일 때 0.0이 되도록 정규화
    return 1.0 - (2 * abs(0.5 - ratio))

def run_single_culture_simulation(culture_name, weights, episodes):
    agent = QLearningAgent(culture_name, weights, SCENARIOS)
    
    history = {
        "episode": [],
        "reward": [],
        "diversity": []
    }
    
    progress = st.progress(0)
    
    for ep in range(episodes):
        ep_actions = []
        ep_reward = 0
        
        for scn in SCENARIOS:
            action = agent.get_action(scn.sid)
            reward = agent.calculate_reward(scn.sid, action)
            agent.update(scn.sid, action, reward)
            
            ep_actions.append(action)
            ep_reward += reward
            
        agent.decay_epsilon()
        
        # 지표 기록
        history["episode"].append(ep + 1)
        history["reward"].append(ep_reward)
        # 이번 에피소드의 행동 다양성 (S1, S2, S3의 선택이 얼마나 섞였는지)
        history["diversity"].append(calculate_diversity(ep_actions))
        
        if (ep + 1) % 10 == 0:
            progress.progress((ep + 1) / episodes)
            
    progress.empty()
    return pd.DataFrame(history)

# ==================== UI 구성 ====================
st.title("🔬 AI Ethics: 강화학습 에이전트 시뮬레이션")
st.markdown("""
특정 문화권 내에서 AI가 학습할 때 '행동의 다양성(Behavioral Diversity)'과 '획득한 보상(Reward)' 간에 
어떤 상관관계가 있는지 분석합니다.
""")

# --- 사이드바 ---
st.sidebar.header("⚙️ 설정 (Settings)")

# 1. 문화권 선택 (단일 선택)
selected_culture = st.sidebar.selectbox(
    "분석할 문화권 선택", 
    list(CULTURES_PRESETS.keys()),
    index=3 # Default to KOREA
)

# 2. 파라미터
episodes = st.sidebar.slider("학습 에피소드 수", 100, 1000, 300, step=50)

# 가중치 확인 및 커스텀 (문화권 내 미세 조정)
st.sidebar.markdown("---")
st.sidebar.subheader(f"{selected_culture} 가중치 상세")
current_weights = CULTURES_PRESETS[selected_culture].copy()
use_custom = st.sidebar.checkbox("가중치 미세 조정하기", False)

if use_custom:
    for k in FRAMEWORKS:
        current_weights[k] = st.sidebar.slider(f"{k}", 0.0, 1.0, current_weights[k])
    # 정규화
    total_w = sum(current_weights.values()) or 1
    current_weights = {k: v/total_w for k, v in current_weights.items()}
else:
    st.sidebar.json(current_weights)

# --- 메인 실행 ---
if st.button("🚀 분석 시작 (Analyze)", type="primary"):
    with st.spinner(f"'{selected_culture}' 문화권 시뮬레이션 중..."):
        df = run_single_culture_simulation(selected_culture, current_weights, episodes)
    
    st.success("분석 완료!")
    
    # 1. 시계열 그래프 (Reward & Diversity)
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 학습 곡선 (Learning Curve)")
        st.caption("에피소드가 진행됨에 따라 AI가 획득한 총 보상의 변화")
        st.line_chart(df, x="episode", y="reward", color="#FF4B4B")
        
    with col2:
        st.subheader("🔀 행동 다양성 (Behavioral Diversity)")
        st.caption("선택의 다양성 (0: 한쪽으로 쏠림, 1: A/B 골고루 선택)")
        st.line_chart(df, x="episode", y="diversity", color="#1F77B4")
        
    st.markdown("---")
    
    # 2. 상관관계 분석 (Scatter Plot)
    st.subheader("🔗 상관관계 분석: 다양성 vs 보상")
    
    # 상관계수 계산
    r_val, p_val = pearsonr(df["diversity"], df["reward"])
    
    col_corr1, col_corr2 = st.columns([2, 1])
    
    with col_corr1:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(df["diversity"], df["reward"], alpha=0.5, c='purple')
        ax.set_xlabel("Behavioral Diversity")
        ax.set_ylabel("Total Reward")
        ax.set_title(f"Diversity vs Reward (Scatter)")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
    with col_corr2:
        st.markdown("### 📊 통계 요약")
        st.metric("피어슨 상관계수 (r)", f"{r_val:.3f}")
        st.metric("P-value", f"{p_val:.3e}")
        
        st.markdown("---")
        st.markdown("**💡 해석 가이드**")
        if r_val > 0.3:
            st.info("양의 상관관계: 다양하게 시도할수록 더 높은 보상을 얻는 경향이 있습니다.")
        elif r_val < -0.3:
            st.warning("음의 상관관계: 특정 행동에 집중(다양성 낮음)해야 보상이 높아집니다.")
        else:
            st.write("뚜렷한 상관관계가 없습니다. 보상은 다양성과 무관하게 결정됩니다.")

    # 3. 데이터 다운로드
    with st.expander("📥 로우 데이터(Raw Data) 보기"):
        st.dataframe(df)
        st.download_button(
            "CSV 다운로드", 
            df.to_csv(index=False), 
            file_name=f"{selected_culture}_simulation_data.csv"
        )
