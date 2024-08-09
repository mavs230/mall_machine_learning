# -*- coding: utf-8 -*-
"""
Created on Wed May 15 10:04:21 2024

@author: mavs2
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('C:/Users/mavs2/Documents/project/Mall_Customers.csv')

# Display first rows
print(data.head())

# Check for missing values
missing_values = data.isnull().sum()

# Display missing values count
print("Missing values:\n", missing_values)

# Display basic info about the dataset
print("Dataset Information")
print(data.info())

# Display summary statistic
print("\nSummary Statistic:")
print(data.describe())

# Display the first few rows of the dataset
print("First Few Rows of the Dataset:")
print(data.head())

# Visualize distributions of numerical features
sns.set(style="whitegrid")
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
sns.histplot(data['Age'], bins=20, kde=True, color='skyblue')
plt.title('Distribution of Age')

plt.subplot(1, 3, 2)
sns.histplot(data['Annual Income (k$)'], bins=20, kde=True, color='salmon')
plt.title('Distribution of Annual Income')

plt.subplot(1, 3, 3)
sns.histplot(data['Spending Score (1-100)'], bins=20, kde=True, color='green')
plt.title('Distribution of spending Score')

plt.tight_layout()
plt.show()

# Explore relationships between numerical features
sns.pairplot(data.drop('CustomerID', axis=1), hue='Gender', height=4)
plt.suptitle('Pairplot of Numerical Features', y=1.02)
plt.show()

# Select numerical features for clustering
numerical_features = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Standardize the numerical features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(numerical_features)

# Apply K-Means clustering with different numbers of clusters
k_values = [2, 3, 4, 5, 6]

for k in k_values:
    # Initialize Kmeans with with the current number of clusters
    kmeans = KMeans(n_clusters=k, random_state=42)
    
    # Fit KMeans to the scaled features
    kmeans.fit(scaled_features)
    
    # Assign cluster labels to the original dataset
    data[f'Cluster_{k}'] = kmeans.labels_

# Display the first few rows of the dataset with cluster labels
print("First Few Rows with Cluster Labels:")
print(data.head())

# Define a function to visualize clusters
def visualize_clusters(data, cluster_column, x_feature, y_feature):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=data, x=x_feature, y=y_feature, hue=cluster_column, palette='viridis')
    plt.title('Visualization of Clusters')
    plt.xlabel(x_feature)
    plt.ylabel(y_feature)
    plt.legend(title='Cluster')
    plt.grid(True)
    plt.show()

# Visualize clusters for different numbers of clusters
for k in k_values:
    cluster_column = f'Cluster_{k}'
    print(f"Visualizing Clusters for {k} Clusters")
    visualize_clusters(data, cluster_column, 'Annual Income (k$)', 'Spending Score (1-100)')

# Define a function to visualize cluster profiles
def visualize_cluster_profiles(data, cluster_column, numerical_features):
    cluster_profiles = data.groupby(cluster_column)[numerical_features].mean()
    cluster_profiles.plot(kind='bar', figsize=(10, 6))
    plt.title('Cluster Profiles')
    plt.xlabel('Cluster')
    plt.ylabel('Mean Value')
    plt.xticks(rotation=0)
    plt.legend(title='Feature')
    plt.grid(True)
    plt.show()

# Analyze cluster characteristics
for k in k_values:
    cluster_column = f'Cluster_{k}'
    cluster_stats = data.groupby(cluster_column).agg({
        'Age': ['mean', 'std'],
        'Annual Income (k$)': ['mean', 'std'],
        'Spending Score (1-100)': ['mean', 'std', 'count']
    })
    print(f"Cluster Characteristics for {k} Clusters:")
    print(cluster_stats)

# Visualize cluster profiles
visualize_cluster_profiles(data, 'Cluster_3', ['Age', 'Annual Income (k$)', 'Spending Score (1-100)'])

# Interpretation and Insights
print("\nInterpretation and Insights:")
print("Cluster 0 (Low Income, Low Spending Score): This cluster represents customers with low annual income and low spending scores. They may be more budget-conscious.")
print("Cluster 1 (High Income, High Spending Score): This cluster represents customers with high annual income and high spending scores. They are likely high-value customers and may be targeted for premium services or products.")
print("Cluster 2 (Mid Income, Mid Spending Score): This cluster represents customers with moderate annual income and spending scores. They may represent a balance between budget-conscious and high-value customers.")
