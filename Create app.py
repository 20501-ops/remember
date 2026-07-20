import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# 1. 페이지 설정
st.set_page_config(page_title="Orbit Review", layout="wide")
st.title("🚀 궤도형 간격 학습 앱: Orbit Review")
st.markdown("""
본 앱은 학습 항목의 망각 상태를 **이차곡선 궤도 위의 점**으로 시각화합니다.
- **초점 (0,0):** 완전히 암기된 상태 (Memory Core)
- **궤도 위의 점 (P):** 현재 학습 항목의 상태 (시간이 흐를수록 초점에서 멀어짐)
- **임계원 (Threshold Circle):** 망각 한계선. 점이 이 원을 벗어나면 **복습 알림**이 발생합니다.
""")

# 2. 고정된 기하학적 상수 설정
K = 40.0             # 궤도 스케일 팩터
R_THRESHOLD = 6.0    # 임계원 반지름 (이 값을 넘으면 망각)
M_MAX = 10.0         # 장반경(a)의 수학적 수렴 한계치

# 3. 세션 상태를 이용한 가상 학습 데이터베이스 초기화
if 'word_db' not in st.session_state:
    st.session_state.word_db = {
        "matrix": {"a": 5.0, "e": 0.8, "n": 1, "desc": "대수학 - 행렬 (기억 변동성 큼)"},
        "eigenvalue": {"a": 7.5, "e": 0.3, "n": 4, "desc": "고유값과 고유벡터 (안정적 궤도 진입)"},
        "conic section": {"a": 4.5, "e": 0.9, "n": 0, "desc": "이차곡선 정의 (방금 학습함)"}
    }

# 4. 사이드바: 구별 검색 및 관리 기능
st.sidebar.header("🔍 학습 항목 검색 및 관리")
search_query = st.sidebar.text_input("단어 검색 (예: matrix, eigenvalue):").lower()

# 검색 처리 및 대상 단어 선정
target_word = None
if search_query in st.session_state.word_db:
    target_word = search_query
    st.sidebar.success(f"'{target_word}' 데이터 로드 완료")
else:
    if search_query:
        st.sidebar.warning("검색 결과가 없어 새 단어로 임시 등록합니다.")
        st.session_state.word_db[search_query] = {"a": 4.0, "e": 0.85, "n": 0, "desc": "새로 추가된 단어"}
        target_word = search_query
    else:
        target_word = list(st.session_state.word_db.keys())[0] # 기본값

# 현재 선택된 단어의 정보 가져오기
word_info = st.session_state.word_db[target_word]
st.sidebar.info(f"**설명:** {word_info['desc']}\n\n**현재 복습 횟수:** {word_info['n']}회")

# 5. 간격 학습 알고리즘 (귀납적 수열 기반 갱신)
st.sidebar.subheader("🔄 복습 결과 피드백 심사")
col1, col2 = st.sidebar.columns(2)

if col1.button("✅ 기억남 (Correct)"):
    # 수열의 귀납적 정의: a_{n+1} = a_n + 0.3 * (M_MAX - a_n) -> 단조증가 및 M_MAX 수렴 보장
    word_info["a"] = word_info["a"] + 0.3 * (M_MAX - word_info["a"])
    # 이심률 감소 (원에 가까워져 안정화)
    word_info["e"] = max(0.05, word_info["e"] * 0.6)
    word_info["n"] += 1
    st.rerun()

if col2.button("❌ 까먹음 (Incorrect)"):
    # 오답 시 궤도 축소 및 이심률 증가 (불안정화)
    word_info["a"] = max(3.5, word_info["a"] * 0.7)
    word_info["e"] = min(0.95, word_info["e"] * 1.3)
    st.rerun()

# 데이터 리셋 버튼
if st.sidebar.button("♻️ 데이터 초기화"):
    del st.session_state.word_db
    st.rerun()

# 6. 메인 화면 시각화 레이아웃
main_col, text_col = columns = st.columns([2, 1])

with text_col:
    st.subheader("📊 현재 궤도 파라미터")
    st.metric(label="장반경 지표 ($a_n$)", value=f"{word_info['a']:.2f}", delta="증가할수록 복습 주기 장기화")
    st.metric(label="이심률 ($e$)", value=f"{word_info['e']:.2f}", delta="0에 가까울수록 기억 안정")
    
    # 시간 경과 슬라이더 (망각 프로세스 시뮬레이션)
    # 페이즈(0 ~ pi)가 진행됨에 따라 초점에서 점이 멀어짐
    st.subheader("⏱️ 시간 경과 시뮬레이션")
    time_phase = st.slider("학습 후 경과 시간 (Phase)", min_value=0.0, max_value=np.pi, value=0.0, step=0.1)

with main_col:
    # 7. 기하학적 수식 연산 및 그래프 작도
    a = word_info["a"]
    e = word_info["e"]
    
    # 초점이 원점(0,0)에 있는 타원의 극좌표 방정식 응용 매핑
    # r(\phi) = (K / a) * (1 - e * cos(\phi)) 로 정의하면, a가 커질수록 r이 작아져(초점에 가까워져) 임계원을 늦게 벗어남
    phi = np.linspace(0, 2*np.pi, 300)
    r_orbit = (K / a) * (1 - e * np.cos(phi))
    x_orbit = r_orbit * np.cos(phi)
    y_orbit = r_orbit * np.sin(phi)
    
    # 현재 시점의 점 P 위치 계산
    r_p = (K / a) * (1 - e * np.cos(time_phase))
    x_p = r_p * np.cos(time_phase)
    y_p = r_p * np.sin(time_phase)
    
    # 그래프 그리기
    fig, ax = plt.subplots(figsize=(7, 7))
    
    # 궤도(타원) 및 초점 그리기
    ax.plot(x_orbit, y_orbit, label=f"Memory Orbit (e={e:.2f})", color="#1f77b4", linewidth=2)
    ax.scatter([0], [0], color="red", s=150, zorder=5, label="Focus (Memory Core)")
    
    # 임계원(Threshold Circle) 그리기
    theta_circle = np.linspace(0, 2*np.pi, 200)
    ax.plot(R_THRESHOLD * np.cos(theta_circle), R_THRESHOLD * np.sin(theta_circle), 
            label="Threshold Circle", color="orange", linestyle="--", linewidth=1.5)
    
    # 현재 학습 항목 점 P 표시
    ax.scatter([x_p], [y_p], color="green" if r_p <= R_THRESHOLD else "magenta", s=120, zorder=6)
    ax.plot([0, x_p], [0, y_p], color="gray", linestyle=":", alpha=0.7) # 점과 초점 사이의 거리선
    
    # 1학년 개념 연결: 위치관계에 따른 최댓값/최솟값 가이드선
    # 최솟값(d_min)은 \phi = 0 일 때, 최댓값(d_max)은 \phi = \pi 일 때 발생
    d_min = (K / a) * (1 - e)
    d_max = (K / a) * (1 + e)
    ax.scatter([-d_max, d_min], [0, 0], color="purple", s=30, alpha=0.5, zorder=4)
    
    # 데코레이션
    limit_view = 20
    ax.set_xlim(-limit_view, limit_view)
    ax.set_ylim(-limit_view, limit_view)
    ax.axhline(0, color='black', linewidth=0.5, alpha=0.3)
    ax.axvline(0, color='black', linewidth=0.5, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    ax.legend(loc="upper right")
    ax.grid(True, linestyle=':', alpha=0.5)
    
    st.pyplot(fig)

    # 8. 실시간 알림 시스템 (임계원 위치 관계 판단)
    st.subheader("🔔 실시간 기억 인출 상태 판정")
    st.write(f"현재 초점으로부터의 거리(망각도): **{r_p:.2f}** (임계 기준값: {R_THRESHOLD})")
    
    if r_p > R_THRESHOLD:
        st.error(f"⚠️ **복습 알림 발생!** 단어가 임계원 외부로 벗어났습니다. 지연 회상(장기기억) 경로가 약화되기 전에 지금 즉시 복습 원추를 가동하세요!")
    else:
        st.success("✅ **안정 상태:** 단어가 임계원 내부에 안전하게 머물고 있습니다. 단기/장기 기억 인출 경로가 균형을 이루는 중입니다.")
