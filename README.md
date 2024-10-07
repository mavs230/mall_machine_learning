# Customer Segmentation Using K-Means Clustering

## Project Overview
This project aims to perform customer segmentation for a retail business using K-Means clustering. By analyzing customer data, we can identify distinct segments of customers, which can help in targeted marketing and personalized services.

## Motivation
Customer segmentation is crucial for businesses to tailor their marketing strategies effectively. Understanding different customer profiles enables businesses to enhance customer satisfaction and increase sales.

## Dataset
The dataset used in this project is **Mall_Customers.csv**, which contains information about customers, including their age, annual income, spending score, and gender. You can access the dataset from [here](link_to_dataset).

### Features:
- **CustomerID**: Unique identifier for each customer
- **Gender**: Gender of the customer
- **Age**: Age of the customer
- **Annual Income (k$)**: Annual income in thousands of dollars
- **Spending Score (1-100)**: Score assigned by the mall based on customer behavior and spending nature

## Installation
To run this project, you will need the following Python libraries:
```bash
pip install pandas matplotlib seaborn scikit-learn







Usage

    Clone this repository to your local machine.
    Ensure that you have the dataset in the specified path.
    Run the script using Python:
python customer_segmentation.py




Results & Insights

The K-Means clustering algorithm was applied to the standardized numerical features. The following clusters were identified:

    Cluster 0: Low Income, Low Spending Score - Budget-conscious customers.
    Cluster 1: High Income, High Spending Score - High-value customers likely to purchase premium products.
    Cluster 2: Mid Income, Mid Spending Score - Balanced customers representing both budget-conscious and high-value segments.

Visualization

Visualizations were created to display the distributions of numerical features, relationships between features, and the clustering results.
Future Work

    Experiment with different clustering algorithms (e.g., DBSCAN, Hierarchical Clustering).
    Use silhouette scores to validate the optimal number of clusters.
    Analyze additional features (e.g., geographical location) to enhance segmentation.
