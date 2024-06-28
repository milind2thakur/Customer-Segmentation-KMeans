import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pickle

# Ensure necessary directories exist
save_model_dir = 'C:/Users/ACER/Desktop/Customer Segmentation/saved_model'
save_image_dir = 'C:/Users/ACER/Desktop/Customer Segmentation/images'
os.makedirs(save_model_dir, exist_ok=True)
os.makedirs(save_image_dir, exist_ok=True)

# Function to load and preprocess data
def load_data(file_path):
    data = pd.read_csv(file_path, encoding='latin1')  # Adjust encoding if needed
    
    # Data Preprocessing
    # Remove rows with missing CustomerID
    data.dropna(subset=['CustomerID'], inplace=True)

    # Convert InvoiceDate to datetime format
    try:
        data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'], format='%d-%m-%Y %H:%M')
    except Exception as e:
        st.error(f"Error parsing dates: {e}")
        return None

    # Calculate 'Amount' for each transaction
    data['Amount'] = data['Quantity'] * data['UnitPrice']

    # Filter out negative or zero Quantity and UnitPrice
    data = data[(data['Quantity'] > 0) & (data['UnitPrice'] > 0)]

    return data

# Function to calculate RFM metrics
def calculate_rfm(data):
    # Calculate Monetary (total spending per customer)
    monetary = data.groupby('CustomerID')['Amount'].sum().reset_index()
    monetary.columns = ['CustomerID', 'Monetary']

    # Calculate Frequency (number of transactions per customer)
    frequency = data.groupby('CustomerID')['InvoiceNo'].count().reset_index()
    frequency.columns = ['CustomerID', 'Frequency']

    # Calculate Recency (days since last purchase)
    recent_date = data['InvoiceDate'].max()
    data['Recency'] = (recent_date - data['InvoiceDate']).dt.days
    recency = data.groupby('CustomerID')['Recency'].min().reset_index()

    # Merge RFM metrics into a single DataFrame
    rfm = pd.merge(monetary, frequency, on='CustomerID')
    rfm = pd.merge(rfm, recency, on='CustomerID')

    return rfm

# Function to remove outliers using IQR
def remove_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    filtered_data = data[(data[column] >= Q1 - 1.5 * IQR) & (data[column] <= Q3 + 1.5 * IQR)]
    return filtered_data

# Function to perform K-means clustering
def kmeans_clustering(data, num_clusters):
    features = ['Monetary', 'Frequency', 'Recency']
    rfm_df = data[features].copy()

    # Scale the data
    scaler = StandardScaler()
    rfm_df_scaled = scaler.fit_transform(rfm_df)

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, max_iter=300, random_state=0)
    kmeans.fit(rfm_df_scaled)

    # Add cluster labels to the DataFrame
    data['Cluster'] = kmeans.labels_

    return data, kmeans

# Function to visualize clusters and save the image
def visualize_clusters(clustered_data, save_image_dir):
    # Scatter plot for Monetary vs. Frequency
    fig, ax = plt.subplots()
    for cluster in clustered_data['Cluster'].unique():
        clustered_data_cluster = clustered_data[clustered_data['Cluster'] == cluster]
        ax.scatter(clustered_data_cluster['Frequency'], clustered_data_cluster['Monetary'], label=f'Cluster {cluster}')
    
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Monetary')
    ax.set_title('K-means Clustering (Monetary vs. Frequency)')
    ax.legend()
    image_path = os.path.join(save_image_dir, 'kmeans_clusters.png')
    plt.savefig(image_path)
    st.pyplot(fig)

    # Box plots for RFM features
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=clustered_data[['Monetary', 'Frequency', 'Recency']], orient='v', palette='Set2', saturation=1, width=0.7, ax=ax)
    ax.set_title("Outliers Variable Distribution", fontsize=14, fontweight='bold')
    ax.set_ylabel("Range", fontweight="bold")
    ax.set_xlabel("Attributes", fontweight="bold")
    image_path = os.path.join(save_image_dir, 'rfm_boxplot.png')
    plt.savefig(image_path)
    st.pyplot(fig)

    # Histogram for Recency
    fig, ax = plt.subplots()
    for cluster in clustered_data['Cluster'].unique():
        sns.histplot(clustered_data[clustered_data['Cluster'] == cluster]['Recency'], bins=20, label=f'Cluster {cluster}', ax=ax)
    
    ax.set_xlabel('Recency')
    ax.set_ylabel('Count')
    ax.set_title('Recency Distribution across Clusters')
    ax.legend()
    image_path = os.path.join(save_image_dir, 'recency_histogram.png')
    plt.savefig(image_path)
    st.pyplot(fig)

    # Bar plot for Cluster Sizes
    fig, ax = plt.subplots()
    cluster_sizes = clustered_data['Cluster'].value_counts().sort_index()
    ax.bar(cluster_sizes.index, cluster_sizes.values)
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Number of Customers')
    ax.set_title('Cluster Sizes')
    image_path = os.path.join(save_image_dir, 'cluster_sizes_barplot.png')
    plt.savefig(image_path)
    st.pyplot(fig)

    # Pair plot for RFM features
    fig = sns.pairplot(clustered_data[['Monetary', 'Frequency', 'Recency', 'Cluster']], hue='Cluster', palette='Set2', diag_kind='kde')
    fig.fig.suptitle('Pair Plot of RFM Features by Cluster', y=1.02)
    image_path = os.path.join(save_image_dir, 'rfm_pairplot.png')
    plt.savefig(image_path)
    st.pyplot(fig.fig)

# Function to visualize the Elbow curve and save the image
def visualize_elbow_curve(data, save_image_dir):
    features = ['Monetary', 'Frequency', 'Recency']
    rfm_df = data[features].copy()

    # Scale the data
    scaler = StandardScaler()
    rfm_df_scaled = scaler.fit_transform(rfm_df)

    # Calculate WCSS for different number of clusters
    wcss = []
    range_n_clusters = [2, 3, 4, 5, 6, 7, 8]
    for num_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=num_clusters, max_iter=50, n_init=10, random_state=0)
        kmeans.fit(rfm_df_scaled)
        wcss.append(kmeans.inertia_)

    # Plot the Elbow Curve
    fig, ax = plt.subplots()
    ax.plot(range_n_clusters, wcss, marker='o')
    ax.set_title('Elbow Method')
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel('WCSS')
    image_path = os.path.join(save_image_dir, 'elbow_curve.png')
    plt.savefig(image_path)
    st.pyplot(fig)

# Main function to run the Streamlit app
def main():
    st.title('Customer Segmentation with K-means Clustering')

    # Sidebar to upload file and select number of clusters
    st.sidebar.title('Upload File and Configure Clustering')
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])

    if uploaded_file is not None:
        # Load data
        data = load_data(uploaded_file)

        if data is not None:
            # Calculate RFM metrics
            rfm_data = calculate_rfm(data)

            # Remove outliers
            rfm_data = remove_outliers(rfm_data, 'Monetary')
            rfm_data = remove_outliers(rfm_data, 'Frequency')
            rfm_data = remove_outliers(rfm_data, 'Recency')

            # Select number of clusters
            num_clusters = st.sidebar.slider("Select number of clusters", min_value=2, max_value=8, value=3)

            # Perform K-means clustering
            clustered_data, kmeans_model = kmeans_clustering(rfm_data, num_clusters)

            # Visualize clusters
            st.header('Visualizing Clusters')
            visualize_clusters(clustered_data, save_image_dir)

            # Visualize Elbow Curve
            st.header('Elbow Curve to Determine Optimal Clusters')
            visualize_elbow_curve(rfm_data, save_image_dir)

            # Save model
            save_model_path = os.path.join(save_model_dir, 'kmeans_model_saved.pkl')
            with open(save_model_path, 'wb') as file:
                pickle.dump(kmeans_model, file)
            st.success(f'K-means model saved successfully to {save_model_path}')

            # Provide download link for the model
            with open(save_model_path, 'rb') as file:
                btn = st.download_button(
                    label="Download K-means model",
                    data=file,
                    file_name="kmeans_model_saved.pkl",
                    mime="application/octet-stream"
                )

if __name__ == '__main__':
    main()
