import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="Phishing Detection",
                   layout="wide")

st.title("🔐 Phishing Website Detection Dashboard")

# ==============================
# LOAD DATASET
# ==============================
df = pd.read_csv("PhiUSIIL_Phishing_URL_Dataset.csv")

# ==============================
# LOAD MODEL
# ==============================
model = joblib.load("phishing_detection_model.pkl")

# ==============================
# SIDEBAR MENU
# ==============================
option = st.sidebar.selectbox(
    "Choose Option",
    ["Dataset Preview",
     "Class Distribution",
     "Correlation Heatmap",
     "Feature Importance",
     "Predict Website"]
)

# ==============================
# DATASET PREVIEW
# ==============================
if option == "Dataset Preview":

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.write("Shape of dataset:", df.shape)

# ==============================
# CLASS DISTRIBUTION
# ==============================
elif option == "Class Distribution":

    st.subheader("Phishing vs Legitimate Distribution")

    fig, ax = plt.subplots()
    sns.countplot(x='label', data=df, ax=ax)

    ax.set_xlabel("Label")
    ax.set_ylabel("Count")

    st.pyplot(fig)

# ==============================
# CORRELATION HEATMAP
# ==============================
elif option == "Correlation Heatmap":

    st.subheader("Feature Correlation Heatmap")

    numeric_df = df.select_dtypes(include=['number'])

    fig, ax = plt.subplots(figsize=(6,8))
    sns.heatmap(numeric_df.corr(),
                cmap='coolwarm',
                ax=ax)

    st.pyplot(fig)

# ==============================
# FEATURE IMPORTANCE
# ==============================
elif option == "Feature Importance":

    st.subheader("Top Important Features")

    # Use ONLY numeric columns (same as training)
    X = df.drop('label', axis=1)
    X = X.select_dtypes(include=['number'])

    features = X.columns
    importances = model.feature_importances_

    fi = pd.DataFrame({
        "Feature": features,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    fig, ax = plt.subplots(figsize=(8,6))
    sns.barplot(x="Importance",
                y="Feature",
                data=fi.head(20),
                ax=ax)

    st.pyplot(fig)

# ==============================
# PREDICTION TOOL
# ==============================
else:

    st.subheader("Enter Feature Values")

    features = df.drop('label', axis=1).columns

    input_data = []

    for f in features:
        val = st.number_input(f, value=0.0)
        input_data.append(val)

    if st.button("Predict"):

        prediction = model.predict([input_data])

        if prediction[0] == 1:
            st.success("✅ Legitimate Website")
        else:
            st.error("⚠️ Phishing Website")