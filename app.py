import streamlit as st
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# 1. 데이터 불러오기
# =========================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = (
    BASE_DIR
    / 'data'
    / 'add2_cafe_customer_experience_profile.csv'
)

df = pd.read_csv(DATA_PATH)


# =========================
# 2. 페이지 설정
# =========================

st.set_page_config(
    page_title='성내동 카페 추천',
    page_icon='☕',
    layout='wide'
)


# =========================
# 3. 제목
# =========================

st.title('☕ 성내동 카페 맞춤 추천')

st.write(
    '카페 리뷰 텍스트 분석 결과를 기반으로 '
    '나에게 맞는 카페를 추천합니다.'
)


# =========================
# 4. 사용자 선호도 입력
# =========================

st.subheader('나의 카페 선호도를 선택해주세요')


coffee = st.slider(
    '☕ 커피·음료 중요도',
    1, 5, 3
)

dessert = st.slider(
    '🥐 디저트·베이커리 중요도',
    1, 5, 3
)

space = st.slider(
    '🏠 공간·분위기 중요도',
    1, 5, 3
)


# =========================
# 5. 추천 버튼
# =========================

if st.button('카페 추천받기'):

    user_vector = np.array([
        coffee,
        dessert,
        space
    ])

    # 정규화
    user_vector = (
        user_vector / user_vector.sum()
    )

    # 카페 경험 프로필
    experience_cols = [
        '커피·음료 경험',
        '디저트·베이커리 경험',
        '공간·분위기 경험'
    ]

    cafe_vectors = df[experience_cols]

    # 유사도 계산
    similarity = cosine_similarity(
        cafe_vectors,
        user_vector.reshape(1, -1)
    )

    # 결과 저장
    df['추천 점수'] = similarity.flatten()

    # TOP 5
    recommendations = (
        df
        .sort_values(
            '추천 점수',
            ascending=False
        )
        .head(5)
    )

    st.subheader('🎉 당신을 위한 성내동 카페 TOP 5')
    
    st.write('당신의 카페 선호도와 가장 잘 어울리는 곳이에요 ☕')
    
    # 순위별 카페 이름 출력
    medals = ['🥇', '🥈', '🥉', '4️⃣', '5️⃣']
    
    for i, (_, row) in enumerate(recommendations.iterrows()):
    
        cafe_name = row['name']
    
        st.markdown(
            f"""
            <div style="
                padding: 20px;
                margin-bottom: 12px;
                border-radius: 12px;
                background-color: #F8F5F2;
                border: 1px solid #E5DED8;
                font-size: 22px;
                font-weight: bold;
            ">
                {medals[i]} {cafe_name}
            </div>
            """,
            unsafe_allow_html=True
        )
