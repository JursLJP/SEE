import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from scipy.stats import gaussian_kde
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.mixture import GaussianMixture

# Load datasets
bank_data = pd.read_csv("bank-full.csv", sep=';')
online_retail_data = pd.read_csv("Online_Retail.csv")

# Convert InvoiceDate to datetime format in Online Retail dataset
online_retail_data['InvoiceDate'] = pd.to_datetime(online_retail_data['InvoiceDate'])

# Extract time intervals (event-based) for Online Retail dataset
online_retail_data = online_retail_data.sort_values(by=['CustomerID', 'InvoiceDate'])
online_retail_data['PrevInvoiceDate'] = online_retail_data.groupby('CustomerID')['InvoiceDate'].shift(1)
online_retail_data['EventInterval'] = (online_retail_data['InvoiceDate'] - online_retail_data['PrevInvoiceDate']).dt.days

# Remove NaN values
online_retail_data.dropna(subset=['EventInterval'], inplace=True)

# Extract time intervals for Bank Marketing dataset (based on campaign calls)
bank_data = bank_data.sort_values(by=['contact', 'day', 'month'])
bank_data['PrevDay'] = bank_data.groupby('contact')['day'].shift(1)
bank_data['EventInterval'] = bank_data['day'] - bank_data['PrevDay']

# Remove NaN values
bank_data.dropna(subset=['EventInterval'], inplace=True)

# Function to plot ECDF
def plot_ecdf(data, title):
    data = np.sort(data)
    y = np.arange(1, len(data) + 1) / len(data)
    plt.figure(figsize=(8, 5))
    plt.plot(data, y, marker='.', linestyle='none')
    plt.xlabel('Event Interval (Days)')
    plt.ylabel('Cumulative Probability')
    plt.title(title)
    plt.grid()
    plt.show()

plot_ecdf(online_retail_data['EventInterval'], 'ECDF of Online Retail Event Intervals')
plot_ecdf(bank_data['EventInterval'], 'ECDF of Bank Marketing Event Intervals')

# Function to compare clustering methods
def compare_clustering_methods(data, feature):
    # K-Means Clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    data['CLUSTER_KMEANS'] = kmeans.fit_predict(data[[feature]])
    
    # DBSCAN Clustering
    dbscan = DBSCAN(eps=5, min_samples=2)
    data['CLUSTER_DBSCAN'] = dbscan.fit_predict(data[[feature]])
    
    # Gaussian Mixture Model Clustering
    gmm = GaussianMixture(n_components=3, random_state=42)
    data['CLUSTER_GMM'] = gmm.fit_predict(data[[feature]])
    
    return data

online_retail_data = compare_clustering_methods(online_retail_data, 'EventInterval')
bank_data = compare_clustering_methods(bank_data, 'EventInterval')

# Visualizing Clusters
def visualize_clusters(data, feature, cluster_col, title):
    plt.figure(figsize=(10, 5))
    sns.scatterplot(x=data[feature], y=data[cluster_col], hue=data[cluster_col], palette='viridis')
    plt.xlabel(feature)
    plt.ylabel('Cluster')
    plt.title(title)
    plt.show()

visualize_clusters(online_retail_data, 'EventInterval', 'CLUSTER_KMEANS', 'K-Means Clustering - Online Retail')
visualize_clusters(online_retail_data, 'EventInterval', 'CLUSTER_DBSCAN', 'DBSCAN Clustering - Online Retail')
visualize_clusters(online_retail_data, 'EventInterval', 'CLUSTER_GMM', 'GMM Clustering - Online Retail')

visualize_clusters(bank_data, 'EventInterval', 'CLUSTER_KMEANS', 'K-Means Clustering - Bank Marketing')
visualize_clusters(bank_data, 'EventInterval', 'CLUSTER_DBSCAN', 'DBSCAN Clustering - Bank Marketing')
visualize_clusters(bank_data, 'EventInterval', 'CLUSTER_GMM', 'GMM Clustering - Bank Marketing')

# Function to summarize clustering insights
def summarize_insights(data, feature):
    summary = {
        'K-Means': data['CLUSTER_KMEANS'].value_counts().to_dict(),
        'DBSCAN': data['CLUSTER_DBSCAN'].value_counts().to_dict(),
        'GMM': data['CLUSTER_GMM'].value_counts().to_dict()
    }
    return summary

clustering_insights_retail = summarize_insights(online_retail_data, 'EventInterval')
clustering_insights_bank = summarize_insights(bank_data, 'EventInterval')

print("Clustering Insights - Online Retail:", clustering_insights_retail)
print("Clustering Insights - Bank Marketing:", clustering_insights_bank)

# Function to compare clustering results
def compare_results(data, feature):
    comparison = data[['CustomerID', feature, 'CLUSTER_KMEANS', 'CLUSTER_DBSCAN', 'CLUSTER_GMM']]
    print("Comparison of Clustering Methods:\n", comparison.head())
    return comparison

comparison_results_retail = compare_results(online_retail_data, 'EventInterval')
comparison_results_bank = compare_results(bank_data, 'EventInterval')
