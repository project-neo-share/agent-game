# app.py — Ethical Crossroads (DNA 2.0 ready) with RL & Analytics
# 작성자: Prof. Songhee Kang
# Optimized for: Automated RL Simulation & Strategy Analysis

import os, json, math, csv, io, datetime as dt, re
import random
import numpy as np
import pandas as pd
import streamlit as st
import httpx
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

# ==================== 설정 ====================
st.set_page_config(page_title="Ethical Crossroads RL", page_icon="🧭", layout="wide")

# ==================== 데이터 모델 ====================
@dataclass
class Scenario:
    sid: str
    title: str
    setup: str
    options: Dict[str, str]
    votes: Dict[str, str]
    base: Dict[str, Dict[str, float]]
    accept: Dict[str, float]
    rewards: Dict[str, Dict[str, float]]

# 5개 시나리오 정의 (보상 벡터 포함)
SCENARIOS: List[Scenario] = [
    Scenario(
        sid="S1", title="1단계: 고전적 트롤리",
        setup="선로 위 5명 vs 1명. 레버를 당길 것인가?",
        options={"A": "1명 희생, 5명 구조 (개입)", "B": "방관 (현상 유지)"},
        votes={"emotion":"A","social":"B","moral":"B","identity":"A"},
        base={"A": {"lives_saved":5, "lives_harmed":1, "fairness_gap":0.35, "rule_violation":0.60, "regret_risk":0.40},
              "B": {"lives_saved":0, "lives_harmed":5, "fairness_gap":0.50, "rule_violation":0.20, "regret_risk":0.60}},
        accept={"A":0.70, "B":0.50},
        rewards={"A": {"emotion": 1.0, "social": -0.5, "moral": -1.0, "identity": 0.5},
                 "B": {"emotion": -1.0, "social": 0.5, "moral": 1.0, "identity": -0.5}}
    ),
    Scenario(
        sid="S2", title="2단계: 맥락적 요소",
        setup="무단 침입자 5명 vs 관리자 자녀 1명.",
        options={"A": "5명 구조 (자녀 희생)", "B": "규정 준수 (5명 방관)"},
        votes={"emotion":"A","social":"B","moral":"B","identity":"B"},
        base={"A": {"lives_saved":5, "lives_harmed":1, "fairness_gap":0.65, "rule_violation":0.60, "regret_risk":0.70},
              "B": {"lives_saved":0, "lives_harmed":5, "fairness_gap":0.45, "rule_violation":0.25, "regret_risk":0.50}},
        accept={"A":0.35, "B":0.60},
        rewards={"A": {"emotion": 0.6, "social": -0.8, "moral": -0.7, "identity": 0.3},
                 "B": {"emotion": -0.5, "social": 0.9, "moral": 0.6, "identity": 0.4}}
    ),
    Scenario(
        sid="S3", title="3단계: 의료 재난 분류",
        setup="일반 부상자 vs 숙련된 의사(잠재 가치).",
        options={"A": "의사 우선 (공리주의)", "B": "동등 대우 (평등주의)"},
        votes={"emotion":"A","social":"B","moral":"B","identity":"A"},
        base={"A": {"lives_saved":7, "lives_harmed":3, "fairness_gap":0.45, "rule_violation":0.35, "regret_risk":0.45},
              "B": {"lives_saved":6, "lives_harmed":4, "fairness_gap":0.30, "rule_violation":0.10, "regret_risk":0.35}},
        accept={"A":0.55, "B":0.65},
        rewards={"A": {"emotion": 0.7, "social": -0.4, "moral": -0.6, "identity": 0.8},
                 "B": {"emotion": -0.3, "social": 0.7, "moral": 0.9, "identity": 0.5}}
    ),
    Scenario(
        sid="S4", title="4단계: 자율주행 딜레마",
        setup="탑승자(개발자) 1명 vs 보행자 3명.",
        options={"A": "보행자 보호 (탑승자 희생)", "B": "탑승자 보호 (보행자 희생)"},
        votes={"emotion":"A","social":"B","moral":"A","identity":"A"},
        base={"A": {"lives_saved":3, "lives_harmed":1, "fairness_gap":0.35, "rule_violation":0.50, "regret_risk":0.55},
              "B": {"lives_saved":1, "lives_harmed":3, "fairness_gap":0.70, "rule_violation":0.60, "regret_risk":0.65}},
        accept={"A":0.60, "B":0.30},
        rewards={"A": {"emotion": 0.8, "social": -0.7, "moral": 0.6, "identity": -0.5},
                 "B": {"emotion": -0.9, "social": 0.8, "moral": -0.7, "identity": 0.9}}
    ),
    Scenario(
        sid="S5", title="5단계: 규제 vs 자율",
        setup="안전 규제 강화 vs 자율성 보장.",
        options={"A": "규제 강화 (안전)", "B": "자율성 보장 (혁신)"},
        votes={"emotion":"B","social":"A","moral":"A","identity":"B"},
        base={"A": {"lives_saved":0, "lives_harmed":0, "fairness_gap":0.20, "rule_violation":0.10, "regret_risk":0.30},
              "B": {"lives_saved":0, "lives_harmed":0, "fairness_gap":0.40, "rule_violation":0.40, "regret_risk":0.40}},
        accept={"A":0.55, "B":0.55},
        rewards={"A": {"emotion": -0.3, "social": 0.9, "moral": 0.8, "identity": -0.6},
                 "B": {"emotion": 0.7, "social": -0.4, "moral": -0.5, "identity": 0.9}}
    )
]

FRAMEWORKS = ["emotion", "social", "moral", "identity"]

# ==================== 강화학습 (RL) 에이전트 ====================
class QLearningAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.q_table = {s.sid: {"A": 0.0, "B": 0.0} for s in SCENARIOS}
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.episode_count = 0
        
    def get_action(self, state_id: str, explore: bool = True) -> str:
        # Epsilon-Greedy
        if explore and random.random() < self.epsilon:
            return random.choice(["A", "B"])
        
        qs = self.q_table[state_id]
        if qs["A"] > qs["B"]: return "A"
        elif qs["B"] > qs["A"]: return "B"
        return random.choice(["A", "B"])

    def update(self, state_id: str, action: str, reward: float):
        # Q(s,a) <- Q(s,a) + alpha * (reward - Q(s,a)) 
        # (단일 스텝이므로 gamma=0 혹은 다음 상태 maxQ 생략 가능하나 일반성을 위해 유지)
        old_q = self.q_table[state_id][action]
        self.q_table[state_id][action] = old_q + self.lr * (reward - old_q)

    def decay_epsilon(self):
        self.epsilon = max(0.01, self.epsilon * 0.995)

    def get_policy_entropy(self) -> float:
        """현재 Q값 기준 Softmax 확률 분포의 엔트로피 계산 (전략의 불확실성)"""
        total_entropy = 0
        temperature = 1.0
        for sid in self.q_table:
            qs = np.array(list(self.q_table[sid].values()))
            # Softmax
            exp_qs = np.exp(qs / temperature)
            probs = exp_qs / np.sum(exp_qs)
            # Entropy = -sum(p * log(p))
            entropy = -np.sum(probs * np.log(probs + 1e-9))
            total_entropy += entropy
        return total_entropy / len(SCENARIOS)

# ==================== 분석 및 시뮬레이션 엔진 ====================
def calculate_reward_vector(scn: Scenario, choice: str, weights: Dict[str, float]) -> float:
    """프레임워크 가중치와 시나리오 보상을 내적(Dot Product)하여 보상 계산"""
    r_vector = scn.rewards[choice]
    # 가중합 (Reward)
    base_reward = sum(r_vector.get(fw, 0) * weights.get(fw, 0) for fw in FRAMEWORKS) * 10
    
    # 추가 페널티/보너스 (구조적 요소)
    meta = scn.base[choice]
    lives_score = (meta["lives_saved"] - meta["lives_harmed"]) * 2
    penalty = (meta["rule_violation"] + meta["fairness_gap"]) * 2
    
    return base_reward + lives_score - penalty

def calculate_diversity(choices_history: List[str]) -> float:
    """선택의 다양성 (A/B 비율의 분산 역수 개념)"""
    if not choices_history: return 0.0
    a_count = choices_history.count("A")
    ratio = a_count / len(choices_history)
    # 0.5에 가까울수록 다양함. 0이나 1이면 획일적.
    # 정규화: 0.5일 때 1.0, 0 or 1일 때 0.0
    return 1.0 - (2 * abs(0.5 - ratio))

def run_simulation(episodes: int, weights: Dict[str, float]):
    agent = QLearningAgent(epsilon=0.5) # 초기 탐험 높게
    history = []
    
    progress_bar = st.progress(0)
    
    for ep in range(episodes):
        ep_data = {"episode": ep + 1, "total_reward": 0, "actions": [], "consistency_sum": 0}
        
        for scn in SCENARIOS:
            # 1. 행동 선택
            action = agent.get_action(scn.sid)
            ep_data["actions"].append(action)
            
            # 2. 보상 계산
            reward = calculate_reward_vector(scn, action, weights)
            ep_data["total_reward"] += reward
            
            # 3. 학습
            agent.update(scn.sid, action, reward)
            
            # 4. 일관성 지표 (선택이 가중치 가장 높은 프레임워크와 일치하는지)
            top_fw = max(weights, key=weights.get)
            match = 1.0 if scn.votes[top_fw] == action else 0.0
            ep_data["consistency_sum"] += match

        # 에피소드 종료 후 처리
        agent.decay_epsilon()
        
        # 지표 저장
        avg_consistency = ep_data["consistency_sum"] / len(SCENARIOS)
        entropy = agent.get_policy_entropy()
        diversity = calculate_diversity(ep_data["actions"])
        
        history.append({
            "Episode": ep + 1,
            "Total Reward": ep_data["total_reward"],
            "Strategy Entropy": entropy,
            "Diversity": diversity,
            "Ethical Consistency": avg_consistency,
            "Epsilon": agent.epsilon
        })
        
        if (ep + 1) % 10 == 0:
            progress_bar.progress((ep + 1) / episodes)
            
    progress_bar.empty()
    return pd.DataFrame(history), agent

# ==================== UI 구성 ====================
st.title("🤖 윤리적 강화학습 시뮬레이터 (RL Analytics)")
st.markdown("""
이 시스템은 AI가 **주어진 윤리 가중치(보상 벡터)**를 기반으로 스스로 학습하는 과정을 시각화합니다.
- **Entropy**: 전략의 불확실성 (낮을수록 확고한 신념 형성)
- **Diversity**: 선택의 다양성 (상황에 따른 유연한 대처)
- **Consistency**: 설정된 윤리관과의 일치도
""")

# --- 사이드바: 가중치 설정 ---
st.sidebar.header("⚖️ 가중치 설정 (Reward Weights)")
w_emotion = st.sidebar.slider("감정 (Emotion)", 0.0, 1.0, 0.5)
w_social = st.sidebar.slider("사회 (Social)", 0.0, 1.0, 0.2)
w_moral = st.sidebar.slider("도덕 (Moral)", 0.0, 1.0, 0.2)
w_identity = st.sidebar.slider("정체성 (Identity)", 0.0, 1.0, 0.1)

# 정규화
total_w = w_emotion + w_social + w_moral + w_identity
if total_w == 0: weights = {k: 0.25 for k in FRAMEWORKS}
else: weights = {"emotion": w_emotion/total_w, "social": w_social/total_w, "moral": w_moral/total_w, "identity": w_identity/total_w}

st.sidebar.markdown("---")
st.sidebar.write("📊 **입력된 가중치 비율**")
st.sidebar.json(weights)

# --- 메인 탭 ---
tab1, tab2 = st.tabs(["🚀 자동 시뮬레이션", "🎮 수동 플레이"])

with tab1:
    st.subheader("고속 학습 시뮬레이션")
    col1, col2 = st.columns([1, 3])
    
    with col1:
        n_episodes = st.number_input("에피소드 수", min_value=10, max_value=2000, value=200, step=10)
        start_sim = st.button("시뮬레이션 시작", type="primary")

    if start_sim:
        with st.spinner("AI가 윤리적 딜레마를 학습 중입니다..."):
            df_res, trained_agent = run_simulation(n_episodes, weights)
        
        st.success("학습 완료!")
        
        # 1. 보상 및 일관성 그래프
        st.subheader("📈 학습 곡선")
        chart_data = df_res[["Episode", "Total Reward", "Ethical Consistency"]].melt('Episode')
        st.line_chart(
            df_res, x="Episode", y=["Total Reward", "Ethical Consistency"],
            color=["#FF5733", "#33FF57"], height=300
        )
        
        # 2. 고급 지표 (엔트로피 & 다양성)
        st.subheader("🧠 전략 분석 지표")
        col_m1, col_m2 = st.columns(2)
        
        with col_m1:
            st.markdown("**전략 엔트로피 (Strategy Entropy)**")
            st.caption("값이 낮아질수록 AI가 확고한 윤리적 판단 기준을 세웠음을 의미합니다.")
            st.line_chart(df_res, x="Episode", y="Strategy Entropy", color="#3357FF", height=250)
            
        with col_m2:
            st.markdown("**행동 다양성 (Action Diversity)**")
            st.caption("1.0에 가까울수록 A/B를 상황에 맞게 섞어서 선택합니다.")
            st.line_chart(df_res, x="Episode", y="Diversity", color="#FF33A1", height=250)
            
        # 3. 최종 Q-Table 히트맵 유사 시각화
        st.subheader("🎯 최종 학습 결과 (Q-Values)")
        q_data = []
        for sid, q in trained_agent.q_table.items():
            best = "A" if q["A"] > q["B"] else "B"
            q_data.append({
                "Scenario": sid, 
                "Score A": round(q["A"], 2), 
                "Score B": round(q["B"], 2),
                "Choice": best
            })
        st.dataframe(pd.DataFrame(q_data).set_index("Scenario"), use_container_width=True)

with tab2:
    st.info("기존의 수동 플레이 모드입니다. (학습된 에이전트 테스트용)")
    if 'rl_agent' not in st.session_state:
        st.session_state.rl_agent = QLearningAgent()
        
    # 간단한 플레이 UI (기존 코드의 축약판)
    current_scn_idx = st.session_state.get('scn_idx', 0)
    
    if current_scn_idx < len(SCENARIOS):
        s = SCENARIOS[current_scn_idx]
        st.markdown(f"**{s.title}**")
        st.write(s.setup)
        
        c1, c2 = st.columns(2)
        if c1.button("A 선택"):
            r = calculate_reward_vector(s, "A", weights)
            st.session_state.rl_agent.update(s.sid, "A", r)
            st.toast(f"보상: {r:.1f}")
            st.session_state.scn_idx = current_scn_idx + 1
            st.rerun()
            
        if c2.button("B 선택"):
            r = calculate_reward_vector(s, "B", weights)
            st.session_state.rl_agent.update(s.sid, "B", r)
            st.toast(f"보상: {r:.1f}")
            st.session_state.scn_idx = current_scn_idx + 1
            st.rerun()
    else:
        st.success("모든 라운드 종료")
        if st.button("다시 하기"):
            st.session_state.scn_idx = 0
            st.rerun()
