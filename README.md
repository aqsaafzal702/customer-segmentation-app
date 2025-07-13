# Customer Segmentation App

An interactive **Streamlit web app** for customer segmentation using KMeans clustering on the Mall Customers dataset.  
Identify distinct customer groups based on demographics and purchasing behavior.

## 🔍 Features

-  Interactive EDA (Histograms, Box Plots, Scatter Plots)
-  Elbow Method to find optimal clusters
-  KMeans Clustering with PCA-based 2D visualization
-  Feature importance with Random Forest
-  Cluster characteristics table
-  Download clustered dataset (CSV)

## Tools & Technologies

- Python
- Streamlit
- Pandas, NumPy
- Scikit-learn (KMeans, PCA, Random Forest)
- Plotly (for beautiful interactive charts)

## Dataset

- [Mall Customer Segmentation Data – Kaggle](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)

## Run the App Locally

```bash
pip install -r requirements.txt
streamlit run app.py
