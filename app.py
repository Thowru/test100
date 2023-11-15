import streamlit as st
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 그래프 한글 폰트 설정
plt.rc('font', family='NanumGothic')

def main():
    st.title('Entity 클러스터링 및 PCA 시각화')

    # 파일 업로드
    uploaded_csvfile = st.file_uploader("CSV 파일 선택", type="csv")

    if uploaded_csvfile is not None:
        # 업로드된 파일 읽기
        df_entity = pd.read_csv(uploaded_csvfile, index_col='entity')

        # Feature Scaling
        columns_to_scale = ['method_cnt', 'status_cnt', 'ua_cnt', 'bytes_avg', 'bytes_std']
        scaler = preprocessing.MinMaxScaler()
        df_entity[columns_to_scale] = scaler.fit_transform(df_entity[columns_to_scale])

        # KMeans Clustering
        cols_to_train = ['method_cnt', 'method_post', 'protocol_1_0', 'status_major', 'status_404', 'status_499', 'status_cnt', 'path_same', 'path_xmlrpc', 'ua_cnt', 'has_payload', 'bytes_avg', 'bytes_std']
        kmeans = KMeans(n_clusters=2, random_state=42)
        kmeans.fit(df_entity[cols_to_train])

        # DBSCAN Clustering
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        df_entity['cluster_dbscan'] = dbscan.fit_predict(df_entity[cols_to_train])

        # 결과 표시 - KMeans
        st.write("KMeans 클러스터별 데이터 수:")
        st.write(df_entity['cluster_kmeans'].value_counts())

        st.write("KMeans 클러스터 0에 속한 Entity:")
        st.write(df_entity[df_entity['cluster_kmeans'] == 0].index)

        # 결과 표시 - DBSCAN
        st.write("DBSCAN 클러스터별 데이터 수:")
        st.write(df_entity['cluster_dbscan'].value_counts())

        st.write("DBSCAN 클러스터 0을 제외한 Entity:")
        st.write(df_entity[df_entity['cluster_dbscan'] != 0].index)

        # PCA를 사용하여 데이터의 차원을 2로 축소
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(df_entity[cols_to_train])

        # PCA 결과를 데이터프레임에 추가
        df_entity['pca_1'] = pca_result[:, 0]
        df_entity['pca_2'] = pca_result[:, 1]

        # 2D PCA 결과를 시각화
        fig = plt.figure(figsize=(10, 6))
        plt.scatter(df_entity['pca_1'], df_entity['pca_2'], c=df_entity['cluster_kmeans'], cmap='viridis', s=60)
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        plt.title("KMeans 클러스터링된 Entity 시각화 (PCA 결과)")
        plt.colorbar(label='클러스터')

        st.pyplot(fig)

        # 2D PCA 결과를 DBSCAN 클러스터로 시각화
        fig_dbscan = plt.figure(figsize=(10, 6))
        plt.scatter(df_entity['pca_1'], df_entity['pca_2'], c=df_entity['cluster_dbscan'], cmap='viridis', s=60)
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        plt.title("DBSCAN 클러스터링된 Entity 시각화 (PCA 결과)")
        plt.colorbar(label='클러스터')

        st.pyplot(fig_dbscan)

if __name__ == '__main__':
    main()
