# app.py — Ethical Crossroads RL (Customizable Scenario Rewards)
# 작성자: Prof. Songhee Kang
# Update: Culture Input & Per-Scenario Reward Vector Configuration

import os, json, math, datetime as dt
import random
import numpy as np
import pandas as pd
import streamlit as st
from dataclasses import dataclass, field
from typing import Dict, List

# ==================== 설정 ====================
st.set_page_config(page_title="Ethical Crossroads RL (Custom)", page_icon="🧭", layout="wide")

# ==================== 데이터 모델 ====================
@dataclass
class Scenario:
    sid: str
    title: str
    setup: str
    options: Dict[str, str]
    votes: Dict[str, str]
    base: Dict[str, Dict[str, float]]
    rewards: Dict[str, Dict[str, float]] # 이 부분이 사용자 입력에 따라 변경됨

# 기본 시나리오 데이터 (Default Presets)
DEFAULT_SCENARIOS = [
    Scenario(
        sid="S1", title="1단계: 고전적 트롤리",
        setup="선로 위 5명 vs 1명. 레버를 당길 것인가?",
        options={"A": "1명 희생, 5명 구조", "B": "방관 (현상 유지)"},
        votes={"emotion":"A","social":"B","moral":"B","identity":"A"},
        base={"A": {"lives_saved":5, "lives_harmed":1, "fairness_gap":0.35, "rule_violation":0.60},
              "B": {"lives_saved":0, "lives_harmed":5, "fairness_gap":0.50, "rule_violation":0.20}},
        rewards={"A": {"emotion": 1.0, "social": -0.5, "moral": -1.0, "identity": 0.5},
                 "B": {"emotion": -1.0, "social": 0.5, "moral": 1.0, "identity": -0.5}}
    ),
    Scenario(
        sid="S2", title="2단계: 맥락적 요소",
        setup="무단 침입자 5명 vs 관리자 자녀 1명.",
        options={"A": "5명 구조 (자녀 희생)", "B": "규정 준수 (5명 방관)"},
        votes={"emotion":"A","social":"B","moral":"B","identity":"B"},
        base={"A": {"lives_saved":5, "lives_harmed":1, "fairness_gap":0.65, "rule_violation":0.60},
              "B": {"lives_saved":0, "lives_harmed":5, "fairness_gap":0.45, "rule_violation":0.25}},
        rewards={"A": {"emotion": 0.6, "social": -0.8, "moral": -0.7, "identity": 0.3},
                 "B": {"emotion": -0.5, "social": 0.9, "moral": 0.6, "identity": 0.4}}
    ),
    Scenario(
        sid="S3", title="3단계: 의료 재난 분류",
        setup="일반 부상자 vs 숙련된 의사(잠재 가치).",
        options={"A": "의사 우선 (공리주의)", "B": "동등 대우 (평등주의)"},
        votes={"emotion":"A","social":"B","moral":"B","identity":"A"},
        base={"A": {"lives_saved":7, "lives_harmed":3, "fairness_gap":0.45, "rule_violation":0.35},
              "B": {"lives_saved":6, "lives_harmed":4, "fairness_gap":0.30, "rule_violation":0.10}},
        rewards={"A": {"emotion": 0.7, "social": -0.4, "moral": -0.6, "identity": 0.8},
                 "B": {"emotion": -0.3, "social": 0.7, "moral": 0.9, "identity": 0.5}}
    ),
]

FRAMEWORKS = ["emotion", "social", "moral", "identity"]

# ==================== 강화학습 (RL) 에이전트 ====================
class QLearningAgent:
    def __init__(self, scenarios, learning_rate=0.1, epsilon=0.1):
        # 시나리오 ID가 동적으로 변하지 않는다고 가정
        self.q_table = {s.sid: {"A": 0.0, "B": 0.0} for s in scenarios}
        self.lr = learning_rate
        self.epsilon = epsilon
        
    def get_action(self, state_id: str, explore: bool = True) -> str:
        if explore and random.random() < self.epsilon:
            return random.choice(["A", "B"])
        qs = self.q_table[state_id]
        if qs["A"] > qs["B"]: return "A"
        elif qs["B"] > qs["A"]: return "B"
        return random.choice(["A", "B"])

    def update(self, state_id: str, action: str, reward: float):
        old_q = self.q_table[state_id][action]
        self.q_table[state_id][action] = old_q + self.lr * (reward - old_q)

    def decay_epsilon(self):
        self.epsilon = max(0.01, self.epsilon * 0.995)

    def get_policy_entropy(self) -> float:
        total_entropy = 0
        for sid in self.q_table:
            qs = np.array(list(self.q_table[sid].values()))
            exp_qs = np.exp(qs) # Softmax logic simplified
            probs = exp_qs / np.sum(exp_qs)
            entropy = -np.sum(probs * np.log(probs + 1e-9))
            total_entropy += entropy
        return total_entropy / len(self.q_table)

# ==================== 보상 계산 엔진 ====================
def calculate_reward_vector(scn: Scenario, choice: str, weights: Dict[str, float]) -> float:
    """
    핵심 로직:
    사용자가 설정한 '가치관 가중치(Weights)'와 
    사용자가 설정한 '시나리오 보상 벡터(Rewards)'의 내적(Dot Product)
    """
    r_vector = scn.rewards[choice]
    
    # 내적 계산 (Weights · Rewards)
    base_reward = sum(r_vector.get(fw, 0) * weights.get(fw, 0) for fw in FRAMEWORKS) * 10
    
    # 구조적 페널티 (Optional: 생명/규칙 등)
    meta = scn.base[choice]
    lives_score = (meta["lives_saved"] - meta["lives_harmed"]) * 2
    
    return base_reward + lives_score

def calculate_diversity(choices_history: List[str]) -> float:
    if not choices_history: return 0.0
    a_count = choices_history.count("A")
    ratio = a_count / len(choices_history)
    return 1.0 - (2 * abs(0.5 - ratio))

# ==================== 시뮬레이션 실행 함수 ====================
def run_simulation(episodes: int, weights: Dict[str, float], custom_scenarios: List[Scenario]):
    agent = QLearningAgent(custom_scenarios, epsilon=0.5)
    history = []
    progress_bar = st.progress(0)
    
    for ep in range(episodes):
        ep_data = {"episode": ep + 1, "total_reward": 0, "actions": [], "consistency_sum": 0}
        
        for scn in custom_scenarios:
            action = agent.get_action(scn.sid)
            ep_data["actions"].append(action)
            
            # 커스텀된 시나리오 보상으로 계산
            reward = calculate_reward_vector(scn, action, weights)
            ep_data["total_reward"] += reward
            
            agent.update(scn.sid, action, reward)
            
            # 일관성 (가장 높은 가중치 프레임워크와 선택의 정합성)
            top_fw = max(weights, key=weights.get)
            # 해당 선택지가 top_fw에서 양수 보상을 주는지 확인
            is_consistent = 1.0 if scn.rewards[action][top_fw] > 0 else 0.0
            ep_data["consistency_sum"] += is_consistent

        agent.decay_epsilon()
        
        avg_consistency = ep_data["consistency_sum"] / len(custom_scenarios)
        entropy = agent.get_policy_entropy()
        diversity = calculate_diversity(ep_data["actions"])
        
        history.append({
            "Episode": ep + 1,
            "Total Reward": ep_data["total_reward"],
            "Strategy Entropy": entropy,
            "Diversity": diversity,
            "Ethical Consistency": avg_consistency
        })
        
        if (ep + 1) % 10 == 0:
            progress_bar.progress((ep + 1) / episodes)
            
    progress_bar.empty()
    return pd.DataFrame(history), agent

# ==================== UI 구성 ====================
st.title("🧩 시나리오별 보상 벡터 커스텀 시뮬레이션")
st.markdown("문화권에 따른 가중치 설정과 시나리오별 보상 구조를 입력하여 AI의 윤리적 학습 양상을 분석합니다.")

# --- 사이드바: 문화권 및 가치관 설정 ---
st.sidebar.header("1️⃣ 문화권 및 가치관 (Agent Context)")
culture_context = st.sidebar.text_input("🌍 문화권 입력", value="현대 한국 사회 (일반)", help="분석 결과의 라벨로 사용됩니다.")

st.sidebar.subheader("가치관 가중치 (Weights)")
st.sidebar.caption("AI 에이전트가 어떤 가치를 중요하게 여기는지 설정")
w_emotion = st.sidebar.slider("감정 (Emotion)", 0.0, 1.0, 0.5)
w_social = st.sidebar.slider("사회 (Social)", 0.0, 1.0, 0.2)
w_moral = st.sidebar.slider("도덕 (Moral)", 0.0, 1.0, 0.2)
w_identity = st.sidebar.slider("정체성 (Identity)", 0.0, 1.0, 0.1)

# 가중치 정규화
total_w = w_emotion + w_social + w_moral + w_identity
if total_w == 0: weights = {k: 0.25 for k in FRAMEWORKS}
else: weights = {"emotion": w_emotion/total_w, "social": w_social/total_w, "moral": w_moral/total_w, "identity": w_identity/total_w}

st.sidebar.divider()
st.sidebar.write(f"🏷 **설정된 문화권:** {culture_context}")
st.sidebar.json(weights)

# --- 메인 영역: 시나리오별 보상 벡터 설정 ---
st.header("2️⃣ 시나리오별 보상 벡터 설정 (Environment Setup)")
st.caption("각 시나리오의 선택지(A/B)가 주는 보상 값을 직접 수정할 수 있습니다. (-1.0: 매우 부정적, +1.0: 매우 긍정적)")

custom_scenarios = []

# 시나리오 입력 루프
for idx, default_scn in enumerate(DEFAULT_SCENARIOS):
    with st.expander(f"📝 {default_scn.title} 설정 펼치기", expanded=(idx==0)):
        st.write(f"**상황**: {default_scn.setup}")
        
        col_a, col_b = st.columns(2)
        
        # 선택지 A 보상 입력
        with col_a:
            st.markdown(f"**🅰 선택지 A: {default_scn.options['A']}**")
            r_a = {}
            for fw in FRAMEWORKS:
                default_val = default_scn.rewards["A"].get(fw, 0.0)
                r_a[fw] = st.slider(f"[A] {fw} 보상", -1.0, 1.0, default_val, 0.1, key=f"s{idx}_a_{fw}")
        
        # 선택지 B 보상 입력
        with col_b:
            st.markdown(f"**🅱 선택지 B: {default_scn.options['B']}**")
            r_b = {}
            for fw in FRAMEWORKS:
                default_val = default_scn.rewards["B"].get(fw, 0.0)
                r_b[fw] = st.slider(f"[B] {fw} 보상", -1.0, 1.0, default_val, 0.1, key=f"s{idx}_b_{fw}")
        
        # 수정된 시나리오 객체 생성
        new_scn = Scenario(
            sid=default_scn.sid,
            title=default_scn.title,
            setup=default_scn.setup,
            options=default_scn.options,
            votes=default_scn.votes,
            base=default_scn.base,
            rewards={"A": r_a, "B": r_b}  # 사용자가 입력한 보상 벡터 적용
        )
        custom_scenarios.append(new_scn)

# --- 시뮬레이션 실행 ---
st.divider()
st.header("3️⃣ 시뮬레이션 및 결과 분석")

col_run1, col_run2 = st.columns([1, 3])
with col_run1:
    episodes = st.number_input("학습 에피소드 수", 10, 2000, 200, step=50)
    btn_start = st.button("🚀 시뮬레이션 시작", type="primary")

if btn_start:
    with st.spinner(f"'{culture_context}' 문화권 설정으로 학습 중..."):
        df_res, trained_agent = run_simulation(episodes, weights, custom_scenarios)
    
    st.success("분석 완료!")
    
    # 결과 시각화
    tab1, tab2 = st.tabs(["📊 학습 지표 (Metrics)", "🧠 최종 학습 상태 (Q-Table)"])
    
    with tab1:
        st.subheader(f"📈 학습 곡선 ({culture_context})")
        
        # 1. 보상 및 일관성
        st.line_chart(
            df_res, x="Episode", y=["Total Reward", "Ethical Consistency"],
            color=["#FF5733", "#33FF57"]
        )
        
        col_m1, col_m2 = st.columns(2)
        # 2. 엔트로피
        with col_m1:
            st.write("**전략 엔트로피 (Entropy)** - 판단의 불확실성")
            st.line_chart(df_res, x="Episode", y="Strategy Entropy", color="#3357FF", height=200)
            
        # 3. 다양성
        with col_m2:
            st.write("**행동 다양성 (Diversity)** - 선택의 유연성")
            st.line_chart(df_res, x="Episode", y="Diversity", color="#FF33A1", height=200)

    with tab2:
        st.subheader("🎯 시나리오별 최종 선호도")
        q_data = []
        for s in custom_scenarios:
            q = trained_agent.q_table[s.sid]
            choice = "A" if q["A"] > q["B"] else "B"
            q_data.append({
                "Scenario": s.title,
                "Option A (Score)": f"{q['A']:.2f}",
                "Option B (Score)": f"{q['B']:.2f}",
                "Final Choice": choice
            })
        st.table(pd.DataFrame(q_data))
