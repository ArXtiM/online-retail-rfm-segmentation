import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta

# Функция для предобработки данных
def preprocess_data(df):
    """
    Очищает и фильтрует исходные данные.
    - Удаляет записи без идентификатора клиента.
    - Фильтрует записи с положительным количеством и ценой.
    - Добавляет столбец общей суммы заказа.
    - Преобразует даты.
    - Удаляет выбросы по количеству и общей цене.
    """
    df = df.dropna(subset=['CustomerID'])
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    # Удаление выбросов по количеству
    q1_quantity = df['Quantity'].quantile(0.25)
    q3_quantity = df['Quantity'].quantile(0.75)
    iqr_quantity = q3_quantity - q1_quantity
    lower_bound = q1_quantity - 1.5 * iqr_quantity
    upper_bound = q3_quantity + 1.5 * iqr_quantity
    df = df[(df['Quantity'] >= lower_bound) & (df['Quantity'] <= upper_bound)]

    # Удаление выбросов по общей сумме
    q1_total = df['TotalPrice'].quantile(0.25)
    q3_total = df['TotalPrice'].quantile(0.75)
    iqr_total = q3_total - q1_total
    lower_total = q1_total - 1.5 * iqr_total
    upper_total = q3_total + 1.5 * iqr_total
    df = df[(df['TotalPrice'] >= lower_total) & (df['TotalPrice'] <= upper_total)]

    return df

# Функция для расчёта RFM-показателей
def calculate_rfm(df):
    """
    Формирует RFM таблицу:
    - Recency: сколько дней прошло с последней покупки.
    - Frequency: количество уникальных покупок.
    - Monetary: общая сумма затрат.
    """
    now = df['InvoiceDate'].max() + timedelta(days=1)
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (now - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalPrice': 'sum'
    }).reset_index()
    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
    # Фильтруем клиентов с ненулевыми затратами
    rfm = rfm[rfm['Monetary'] > 0]
    return rfm

# Функция для нормализации и кластеризации RFM-данных
def cluster_data(rfm, k=4):
    """
    Кластеризует клиентов с помощью k-means на основании логарифмированных RFM-показателей.
    Возвращает:
    - rfm: таблицу RFM с меткой кластера.
    - kmeans: обученную модель.
    - rfm_scaled: нормализованные признаки.
    """
    rfm_features = rfm[['Recency', 'Frequency', 'Monetary']].copy()
    rfm_features['Recency_log'] = np.log1p(rfm_features['Recency'])
    rfm_features['Frequency_log'] = np.log1p(rfm_features['Frequency'])
    rfm_features['Monetary_log'] = np.log1p(rfm_features['Monetary'])

    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(
        rfm_features[['Recency_log', 'Frequency_log', 'Monetary_log']]
    )

    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

    return rfm, kmeans, rfm_scaled

# Функция для построения PCA-графика кластеров
def plot_pca(rfm_scaled, clusters):
    """
    Визуализирует клиенты по двум главным компонентам (PCA) с выделением кластеров.
    """
    pca = PCA(n_components=2)
    rfm_pca = pca.fit_transform(rfm_scaled)
    fig, ax = plt.subplots()
    sns.scatterplot(
        x=rfm_pca[:, 0], y=rfm_pca[:, 1], hue=clusters, palette='viridis', ax=ax
    )
    ax.set_title('Визуализация кластеров (PCA)')
    return fig

# Основная часть Streamlit приложения

st.title('Система кластеризации клиентов Online Retail')

# Загрузка файла (Excel или CSV)
uploaded_file = "Online Retail.xlsx"
if uploaded_file:
    
    df = pd.read_excel(uploaded_file)


    st.write('Данные загружены. Размер:', df.shape)

    with st.spinner('Предобработка данных...'):
        df = preprocess_data(df)

    st.write('Данные после предобработки:', df.shape)

    with st.spinner('Расчёт RFM...'):
        rfm = calculate_rfm(df)

    st.write('RFM таблица:', rfm.head())

    k = st.slider('Выберите число кластеров (k)', 2, 10, 4)

    with st.spinner('Кластеризация...'):
        rfm_clustered, kmeans, rfm_scaled = cluster_data(rfm, k)

    profiles = rfm_clustered.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean'
    }).reset_index()

    st.write('Профили кластеров:', profiles)

    st.subheader('Визуализация PCA')
    fig_pca = plot_pca(rfm_scaled, rfm_clustered['Cluster'])
    st.pyplot(fig_pca)

    st.subheader('Рекомендации')
    # Генерация рекомендаций по каждому кластеру
    for i, row in profiles.iterrows():
        st.write(
            f'Кластер {i}: Recency={row["Recency"]:.2f}, '
            f'Frequency={row["Frequency"]:.2f}, '
            f'Monetary={row["Monetary"]:.2f}'
        )
        if row['Recency'] > 100:
            st.write('Рекомендация: Кампания по реактивации (скидки для возврата).')
        elif row['Frequency'] > 5:
            st.write('Рекомендация: Программа лояльности (бонусы для постоянных).')
        else:
            st.write('Рекомендация: Стимулирование частоты (сезонные акции).')
