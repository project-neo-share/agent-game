# app.py — TU Korea AI Management: Ethical AI Simulation
# 작성자: Prof. Songhee Kang
# Update: Simple E-Greedy RL & Korean Comments

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from scipy.stats import pearsonr
from dataclasses import dataclass
from typing import Dict, List

# ==================== 1. 기본 설정 ====================
st.set_page_config(
    page_title="한국공학대 인공지능경영: 윤리 AI 에이전트 강화학습 시뮬레이션", 
    page_icon="🎓", 
    layout="wide"
)

# ==================== 2. 데이터 모델 (환경) ====================
@dataclass
class Scenario:
    sid: str
    title: str
    setup: str
    options: Dict[str, str]
    rewards: Dict[str, Dict[str, float]]

# 4대 윤리 프레임워크 정의
FRAMEWORKS = ["emotion", "social", "moral", "identity"]

# 기본 시나리오 데이터
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

# 문화권 프리셋 (Agent 성향)
CULTURES_PRESETS = {
    "USA":      {"emotion": 0.3, "social": 0.1, "identity": 0.3, "moral": 0.3},
    "CHINA":    {"emotion": 0.1, "social": 0.5, "identity": 0.2, "moral": 0.2},
    "EUROPE":   {"emotion": 0.3, "social": 0.2, "identity": 0.2, "moral": 0.3},
    "KOREA":    {"emotion": 0.2, "social": 0.2, "identity": 0.4, "moral": 0.2},
    "LATIN_AM": {"emotion": 0.4, "social": 0.2, "identity": 0.2, "moral": 0.2},
    "MIDDLE_E": {"emotion": 0.1, "social": 0.2, "identity": 0.2, "moral": 0.5},
    "AFRICA":   {"emotion": 0.2, "social": 0.4, "identity": 0.2, "moral": 0.2},
}

# ==================== 3. 단순 강화학습 에이전트 (Simple E-Greedy) ====================
class SimpleEGreedyAgent:
    """
    아주 기초적인 강화학습 에이전트 클래스입니다.
    복잡한 Q-Learning(미래 가치 고려) 대신, 현재 행동의 평균 보상값을 학습합니다.
    """
    def __init__(self, name, weights, scenarios, learning_rate=0.1, epsilon=0.5):
        self.name = name
        self.weights = weights       # 문화권(Agent)의 가치관 가중치
        self.scenarios = scenarios   # 환경(Environment) 정보
        self.lr = learning_rate      # 학습률 (alpha): 새로운 정보를 얼마나 반영할지 (0~1)
        self.epsilon = epsilon       # 탐험률 (epsilon): 랜덤하게 행동할 확률
        
        # 가치 테이블 초기화 (Q-Table 역할)
        # 예: {'S1': {'A': 0.0, 'B': 0.0}, ...}
        self.q_table = {s.sid: {"A": 0.0, "B": 0.0} for s in scenarios}
        
    def get_action(self, sid):
        """
        [행동 선택: Epsilon-Greedy 정책]
        동전 던지기처럼 epsilon 확률로는 무작위 행동(탐험)을 하고,
        나머지 확률로는 현재 가장 점수가 높은 행동(활용)을 선택합니다.
        """
        # 1. 탐험 (Exploration): 새로운 가능성을 찾아 무작위 선택
        if random.random() < self.epsilon:
            return random.choice(["A", "B"])
        
        # 2. 활용 (Exploitation): 현재 지식 중 최고의 선택
        qs = self.q_table[sid]
        if qs["A"] > qs["B"]: return "A"
        elif qs["B"] > qs["A"]: return "B"
        
        # 점수가 같으면 무작위
        return random.choice(["A", "B"])

    def calculate_reward(self, sid, action):
        """
        [보상 계산]
        환경(시나리오)이 주는 보상 벡터와 에이전트(문화권)의 가치관을 내적(Dot Product)합니다.
        Reward = Sum(시나리오_점수 * 내_가중치) * 10
        """
        scn = next(s for s in self.scenarios if s.sid == sid)
        r_vec = scn.rewards[action]
        
        # 4개 프레임워크 점수 합산
        reward = sum(r_vec.get(k, 0) * self.weights.get(k, 0) for k in FRAMEWORKS) * 10
        return reward

    def update(self, sid, action, reward):
        """
        [학습: 가치 업데이트]
        단순 갱신 공식 (Incremental Mean):
        새로운_가치 = 기존_가치 + 학습률 * (실제_보상 - 기존_가치)
        
        * Q-Learning과 달리 미래 상태(Gamma)를 고려하지 않습니다.
        """
        old_val = self.q_table[sid][action]
        
        # 예측 오차(Error) = 실제 받은 보상 - 내가 예상한 보상
        error = reward - old_val
        
        # 가치 업데이트
        self.q_table[sid][action] = old_val + self.lr * error

    def decay_epsilon(self):
        """
        [탐험률 감소]
        시간이 지날수록 랜덤 선택(탐험)을 줄이고, 학습된 결과(활용)를 더 믿습니다.
        """
        self.epsilon = max(0.01, self.epsilon * 0.99)

# ==================== 4. 분석 도구 ====================
def calculate_diversity(actions_list: List[str]) -> float:
    """행동 다양성 지표 (0.0: 한쪽 쏠림 ~ 1.0: 완벽한 균형)"""
    if not actions_list: return 0.0
    a_count = actions_list.count("A")
    ratio = a_count / len(actions_list)
    return 1.0 - (2 * abs(0.5 - ratio))

def run_simulation(culture_name, weights, episodes, custom_scenarios):
    # 단순 에이전트 인스턴스 생성
    agent = SimpleEGreedyAgent(culture_name, weights, custom_scenarios)
    
    history = {
        "episode": [],
        "reward": [],
        "diversity": []
    }
    
    progress = st.progress(0)
    
    for ep in range(episodes):
        ep_actions = []
        ep_reward = 0
        
        # 모든 시나리오 순회
        for scn in custom_scenarios:
            # 1. 행동 선택 (E-Greedy)
            action = agent.get_action(scn.sid)
            
            # 2. 보상 계산 (내적)
            reward = agent.calculate_reward(scn.sid, action)
            
            # 3. 학습 (값 업데이트)
            agent.update(scn.sid, action, reward)
            
            ep_actions.append(action)
            ep_reward += reward
        
        # 에피소드 종료 후 탐험률 감소
        agent.decay_epsilon()
        
        # 기록
        history["episode"].append(ep + 1)
        history["reward"].append(ep_reward)
        history["diversity"].append(calculate_diversity(ep_actions))
        
        if (ep + 1) % 10 == 0:
            progress.progress((ep + 1) / episodes)
            
    progress.empty()
    return pd.DataFrame(history)

# ==================== 5. UI 구성 ====================
st.title("🎓 한국공학대 인공지능경영: 윤리 AI 에이전트 강화학습 시뮬레이션")
st.markdown("""
이 시뮬레이터는 **초기 형태의 강화학습**(E-Greedy)을 사용하여 AI 에이전트가 문화적 가치관에 따라 윤리적 딜레마를 어떻게 학습하는지 보여줍니다.
1. **환경 설정**: 각 시나리오의 선택지가 주는 보상을 정의합니다.
2. **에이전트 설정**: AI가 중요하게 여기는 가치(문화권)를 설정합니다.
3. **결과 분석**: 학습 과정에서 '행동의 다양성'과 '보상'의 관계를 분석합니다.
""")

# --- [사이드바] 에이전트 설정 ---
st.sidebar.header("👤 2. 에이전트(문화권) 설정")
selected_culture = st.sidebar.selectbox("문화권 프리셋", list(CULTURES_PRESETS.keys()), index=3)
episodes = st.sidebar.slider("학습 횟수 (Episodes)", 100, 1000, 300, step=50)

st.sidebar.subheader("가치관 가중치 조정")
mod_weights = {}
culture_defaults = CULTURES_PRESETS[selected_culture]

# 4대 프레임워크 가중치 입력
for k in FRAMEWORKS:
    mod_weights[k] = st.sidebar.slider(f"{k.capitalize()}", 0.0, 1.0, culture_defaults[k])

# 가중치 정규화
total_w = sum(mod_weights.values()) or 1
final_weights = {k: v/total_w for k, v in mod_weights.items()}

st.sidebar.markdown("---")
st.sidebar.caption("📊 최종 적용 가중치")
st.sidebar.json(final_weights)

# --- [메인] 환경(시나리오) 설정 ---
st.header("🌍 1. 환경(시나리오 보상) 설정")
st.info("각 선택지가 4가지 윤리 프레임워크(Emotion, Social, Moral, Identity)에서 어떤 보상(-1.0 ~ 1.0)을 받는지 설정합니다.")

custom_scenarios = []
tabs = st.tabs([s.title for s in DEFAULT_SCENARIOS])

for i, (tab, default_scn) in enumerate(zip(tabs, DEFAULT_SCENARIOS)):
    with tab:
        st.markdown(f"> **상황:** {default_scn.setup}")
        col_a, col_b = st.columns(2)
        
        # Option A
        with col_a:
            st.markdown(f"### 🅰 {default_scn.options['A']}")
            r_a = default_scn.rewards["A"].copy()
            for fw in FRAMEWORKS:
                r_a[fw] = st.slider(f"[A] {fw}", -1.0, 1.0, r_a.get(fw,0.0), 0.1, key=f"s{i}a_{fw}")
        
        # Option B
        with col_b:
            st.markdown(f"### 🅱 {default_scn.options['B']}")
            r_b = default_scn.rewards["B"].copy()
            for fw in FRAMEWORKS:
                r_b[fw] = st.slider(f"[B] {fw}", -1.0, 1.0, r_b.get(fw,0.0), 0.1, key=f"s{i}b_{fw}")

        custom_scenarios.append(Scenario(
            default_scn.sid, default_scn.title, default_scn.setup, 
            default_scn.options, {"A": r_a, "B": r_b}
        ))

# --- [분석 실행] ---
st.divider()
st.header("🚀 3. 시뮬레이션 및 분석")

if st.button("시뮬레이션 시작", type="primary"):
    with st.spinner("AI 에이전트가 윤리적 가치를 학습 중입니다..."):
        df = run_simulation(selected_culture, final_weights, episodes, custom_scenarios)
    
    st.success("학습 완료!")
    
    # 1. 학습 그래프
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📈 총 보상(Reward) 변화")
        st.caption("학습이 진행될수록 AI가 얻는 보상의 총합")
        st.line_chart(df, x="episode", y="reward", color="#FF4B4B")
    with c2:
        st.subheader("🔀 행동 다양성(Diversity) 변화")
        st.caption("선택의 치우침 정도 (1.0=균형, 0.0=편향)")
        st.line_chart(df, x="episode", y="diversity", color="#1F77B4")
        
    # 2. 상관관계 분석
    st.markdown("---")
    st.subheader("🔗 다양성과 보상의 상관관계 분석")
    
    r_val, p_val = pearsonr(df["diversity"], df["reward"])
    
    col_plot, col_stat = st.columns([2, 1])
    with col_plot:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.scatter(df["diversity"], df["reward"], alpha=0.6, c='purple', edgecolors='w')
        
        # 추세선
        if len(df) > 1:
            z = np.polyfit(df["diversity"], df["reward"], 1)
            p = np.poly1d(z)
            ax.plot(df["diversity"], p(df["diversity"]), "r--", label="추세선")
            
        ax.set_xlabel("Diversity (0=Bias, 1=Fair/Balance)")
        ax.set_ylabel("Total Reward")
        ax.set_title(f"Correlation Scatter Plot (r={r_val:.2f})")
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig)
        
    with col_stat:
        st.metric("피어슨 상관계수 (r)", f"{r_val:.3f}")
        st.metric("유의확률 (P-value)", f"{p_val:.3e}")
        
        st.markdown("#### 💡 해석")
        if r_val > 0.3:
            st.success("✅ **양의 상관관계**\n\n다양한 시도를 할수록 더 높은 보상을 얻습니다.")
        elif r_val < -0.3:
            st.warning("⚠️ **음의 상관관계**\n\n특정 행동을 고수해야 보상이 높습니다.")
        else:
            st.info("⏺ **상관없음**\n\n다양성과 보상은 관계가 없습니다.")

    # 데이터 다운로드
    with st.expander("📥 학습 데이터 다운로드"):
        st.dataframe(df.head())
        st.download_button("CSV로 저장", df.to_csv(index=False), "ai_ethics_data.csv")
