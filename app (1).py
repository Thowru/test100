import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans

# CSV 파일 업로드
uploaded_csvfile = st.file_uploader("CSV 파일 선택", type="csv")

if uploaded_csvfile is not None:
    # CSV 파일 읽기
    df_entity = pd.read_csv(uploaded_csvfile)

    # Feature 선택
    cols_to_train = ['method_cnt', 'method_post', 'protocol_1_0', 'status_major', 'status_404', 'status_499',
                     'status_cnt', 'path_same', 'path_xmlrpc', 'ua_cnt', 'has_payload', 'bytes_avg', 'bytes_std']

    # K-means 모델 학습
    model = KMeans(n_clusters=2, random_state=42)
    model.fit(df_entity[cols_to_train])

    # K-means 결과에 따라 클러스터 할당
    df_entity['cluster_kmeans'] = model.predict(df_entity[cols_to_train])

    # Outlier 클러스터에 속한 데이터 포인트 수 확인
    outlier_count = df_entity['cluster_kmeans'].value_counts().get(0, 0)

    # 중복을 제거한 entity의 내용
    df_entity_unique = df_entity.drop_duplicates(subset=['entity'], keep='last')

    # 중복을 제거한 이상 탐지된 엔터티의 수 확인
    outlier_count_unique = df_entity_unique['cluster_kmeans'].value_counts().get(0, 0)

    # Outlier 클러스터에 속한 중복 제거된 엔터티 출력
    outlier_entities_unique = df_entity_unique[df_entity_unique['cluster_kmeans'] == 0]['entity']

    # Streamlit으로 결과 출력
    st.title('이상 탐지 결과')
    st.write(f"이상 탐지된 엔터티 수: {outlier_count}")
    st.write(f"중복을 제거한 이상 탐지된 엔터티 수: {outlier_count_unique}")
    st.write("이상 탐지된 중복 제거된 엔터티 목록:")
    st.write(outlier_entities_unique)
