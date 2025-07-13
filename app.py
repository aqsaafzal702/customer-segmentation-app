import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

# Page setup
st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title(" Customer Segmentation")

# Description
st.markdown("""
This interactive Streamlit app performs **customer segmentation using KMeans clustering** on mall customers.  
It includes **exploratory data analysis**, **cluster visualization**, and **feature importance insights** to help understand customer behavior.
""")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("Mall_Customers.csv")

data = load_data()
if "CustomerID" in data.columns:
    data = data.drop("CustomerID", axis=1)
numeric_cols = data.select_dtypes(include='number').columns.tolist()

# ================== SIDEBAR CONTROLS ==================
st.sidebar.header("Controls")

# EDA Options
st.sidebar.subheader("EDA Options")
eda_col = st.sidebar.selectbox("Select column for analysis", numeric_cols)
chart_type = st.sidebar.radio("Chart type", ["Histogram", "Box Plot", "Scatter Plot"])

# Clustering Options
st.sidebar.subheader("Clustering Options")
features = st.sidebar.multiselect("Select features for clustering", numeric_cols, default=numeric_cols[:2])
k = st.sidebar.slider("Number of clusters", 2, 10, 5)

# ================== EDA SECTION ==================
st.header("1. Exploratory Data Analysis")

if chart_type == "Histogram":
    fig = px.histogram(
        data,
        x=eda_col,
        color="Gender" if "Gender" in data.columns else None,
        barmode="group",
        title=f"Distribution of {eda_col}"
    )
    fig.update_layout(bargap=0.25, template="plotly_white", title_x=0.5)

elif chart_type == "Box Plot":
    fig = px.box(
        data,
        y=eda_col,
        color="Gender" if "Gender" in data.columns else None,
        title=f"Box Plot of {eda_col}"
    )
    fig.update_layout(template="plotly_white", title_x=0.5)

else:
    y_col = st.selectbox("Select Y-axis", [c for c in numeric_cols if c != eda_col])
    fig = px.scatter(
        data,
        x=eda_col,
        y=y_col,
        color="Gender" if "Gender" in data.columns else None,
        title=f"{eda_col} vs {y_col}"
    )
    fig.update_layout(template="plotly_white", title_x=0.5)

st.plotly_chart(fig, use_container_width=True)
# ================== ELBOW METHOD ==================
st.header("2.1 Elbow Method ")

if len(features) < 2:
    st.info("Select at least 2 features to show the Elbow curve.")
else:
    X_elbow = data[features].dropna()
    scaler = StandardScaler()
    X_scaled_elbow = scaler.fit_transform(X_elbow)

    wcss = []
    K_range = range(1, 11)
    for i in K_range:
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(X_scaled_elbow)
        wcss.append(kmeans.inertia_)

    elbow_fig = px.line(
        x=list(K_range),
        y=wcss,
        markers=True,
        labels={'x': 'Number of Clusters (k)', 'y': 'WCSS'},
        title="Elbow Method for Optimal k"
    )
    elbow_fig.update_layout(template="plotly_white", title_x=0.5)
    st.plotly_chart(elbow_fig, use_container_width=True)
    st.markdown("""
**Interpretation**:  
The Elbow Method helps you identify the optimal number of clusters (k).  
Look for the 'elbow' point where the WCSS starts to decrease slowly — that is usually the best value for `k`.
""")

# ================== CLUSTERING SECTION ==================
st.header("2.2 Customer Clustering")

if len(features) < 2:
    st.warning("Please select at least 2 features for clustering")
else:
    # Prepare data
    X = data[features].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    data["Cluster"] = kmeans.fit_predict(X_scaled)

    # Visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    data["PCA1"] = X_pca[:, 0]
    data["PCA2"] = X_pca[:, 1]

    fig = px.scatter(
        data,
        x="PCA1",
        y="PCA2",
        color="Cluster",
        hover_data=features,
        title="Customer Clusters",
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    fig.update_layout(template="plotly_white", title_x=0.5)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
 **Interpretation**:  
Each dot represents a customer.  
Customers in the same cluster share similar characteristics based on the features you selected.  
The PCA plot reduces dimensions for visualization while preserving as much variance as possible.
""")

    # ================== EXPLAINABILITY ==================
    st.header("3. Cluster Interpretation")

    # Feature Importance
    st.subheader("Feature Importance for Clusters")
    rf = RandomForestClassifier()
    rf.fit(X_scaled, data["Cluster"])
    importance = pd.DataFrame({
        "Feature": features,
        "Importance": rf.feature_importances_
    }).sort_values("Importance", ascending=False)

    fig_imp = px.bar(
        importance,
        x="Feature",
        y="Importance",
        color="Feature",
        color_discrete_sequence=px.colors.qualitative.Pastel,
        title="Feature Importance for Clusters"
    )
    fig_imp.update_layout(template="plotly_white", title_x=0.5)
    st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("""
 **Interpretation**:  
This shows which features most influenced the clustering.  
Higher importance means that feature played a bigger role in separating customers into different groups.
""")

    # Cluster Profile Table
    st.subheader("Cluster Characteristics")
    cluster_profile = data.groupby("Cluster")[features].mean().T
    st.dataframe(cluster_profile.style.background_gradient(cmap="Blues"))

    st.markdown("""
 **Interpretation**:  
This table summarizes average feature values per cluster.  
It helps identify typical behavior or profiles of customers in each segment.
""")

    # Cluster Count Chart
    st.subheader("Cluster Sizes")
    cluster_counts = data['Cluster'].value_counts().sort_index()
    fig2 = px.bar(
        x=cluster_counts.index,
        y=cluster_counts.values,
        color=cluster_counts.index.astype(str),
        color_discrete_sequence=px.colors.qualitative.Vivid,
        labels={'x': 'Cluster', 'y': 'Number of Customers'},
        title="Number of Customers per Cluster"
    )
    fig2.update_layout(template="plotly_white", title_x=0.5)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("""
 **Interpretation**:  
This chart shows how many customers belong to each cluster.  
Clusters with too few or too many customers might indicate under- or over-segmentation.
""")

    # Download Button
    st.download_button(
        label="Download Clustered Customer Data as CSV",
        data=data.to_csv(index=False),
        file_name="customers_segmented.csv",
        mime="text/csv"
    )

# ================== FOOTER ==================
st.markdown("---")
st.markdown("**Created by Aqsa Afzal** | [GitHub](https://github.com/aqsaafzal702) | [LinkedIn](https://linkedin.com/in/aqsa-afzal-21b0a2321)")
