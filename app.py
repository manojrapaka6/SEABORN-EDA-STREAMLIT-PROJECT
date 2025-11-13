# ============================================================
# 📊 Streamlit Data Analysis App (Compact Visuals Version)
# ============================================================

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")

st.set_page_config(page_title="Data Analysis Dashboard", layout="wide")
st.title("📈 Data Analysis Dashboard")

@st.cache_data
def load_data():
    return pd.read_csv("tips_updated.csv")

df = load_data()
st.write("### 🧾 Dataset Preview")
st.dataframe(df.head())

analysis_type = st.sidebar.radio("Choose Analysis Type", ["Univariate", "Bivariate"])

# ============================================================
# 🧮 UNIVARIATE ANALYSIS
# ============================================================
if analysis_type == "Univariate":
    st.header("📊 Univariate Analysis")

    col_type = st.radio("Select Column Type", ["Numerical", "Categorical"])
    ana_type = st.radio("Select Analysis Type", ["Non-Visual", "Visual"])

    # ===============================
    # Numerical Columns
    # ===============================
    if col_type == "Numerical":
        num_cols = df.select_dtypes(include=['int', 'float']).columns
        col = st.selectbox("Select Numerical Column", num_cols)

        if ana_type == "Non-Visual":
            st.subheader("📋 Non-Visual Summary")
            st.dataframe(df[col].describe().to_frame())
        else:
            st.subheader("📈 Visual Analysis")
            chart_type = st.selectbox("Select Chart Type", ["Histogram", "Box Plot", "KDE Plot"])
            fig, ax = plt.subplots(figsize=(5, 3))
            if chart_type == "Histogram":
                sns.histplot(df[col], kde=True, ax=ax)
                plt.title(f"Histogram of {col}")
            elif chart_type == "Box Plot":
                sns.boxplot(x=df[col], ax=ax)
                plt.title(f"Box Plot of {col}")
            elif chart_type == "KDE Plot":
                sns.kdeplot(df[col], fill=True, ax=ax)
                plt.title(f"KDE Plot of {col}")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # ===============================
    # Categorical Columns
    # ===============================
    else:
        cat_cols = df.select_dtypes(exclude=['int', 'float']).columns
        col = st.selectbox("Select Categorical Column", cat_cols)

        if ana_type == "Non-Visual":
            st.subheader("📋 Non-Visual Summary")
            st.dataframe(df[col].value_counts().to_frame("Count"))
        else:
            st.subheader("📊 Visual Analysis")
            chart_type = st.selectbox("Select Chart Type", ["Count Plot", "Pie Chart"])
            fig, ax = plt.subplots(figsize=(5, 3))
            if chart_type == "Count Plot":
                sns.countplot(x=df[col], ax=ax)
                plt.xticks(rotation=45)
                plt.title(f"Count Plot of {col}")
            else:
                df[col].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
                plt.ylabel("")
                plt.title(f"Pie Chart of {col}")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

# ============================================================
# 🔗 BIVARIATE ANALYSIS
# ============================================================
else:
    st.header("🔗 Bivariate Analysis")

    st.write("### Select Relationship Type:")

    if "rel_type" not in st.session_state:
        st.session_state["rel_type"] = None

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Numerical vs Numerical"):
            st.session_state["rel_type"] = "num_num"
    with col2:
        if st.button("Numerical vs Categorical"):
            st.session_state["rel_type"] = "num_cat"
    with col3:
        if st.button("Categorical vs Categorical"):
            st.session_state["rel_type"] = "cat_cat"

    rel_type = st.session_state["rel_type"]

    if rel_type:
        ana_type = st.radio("Select Analysis Type", ["Non-Visual", "Visual"])

        # -----------------------------
        # Numerical vs Numerical
        # -----------------------------
        if rel_type == "num_num":
            num_cols = df.select_dtypes(include=['int', 'float']).columns
            colx = st.selectbox("Select X (Numerical)", num_cols)
            coly = st.selectbox("Select Y (Numerical)", num_cols)

            if ana_type == "Non-Visual":
                st.subheader("📋 Non-Visual Summary")
                if colx == coly:
                    st.warning("⚠️ Same column selected — showing univariate summary instead.")
                    st.dataframe(df[colx].describe().to_frame())
                else:
                    corr = df[[colx, coly]].corr().iloc[0, 1]
                    st.write(f"**Correlation between {colx} and {coly}:** `{corr:.3f}`")
                    st.dataframe(df[[colx, coly]].describe())
            else:
                st.subheader("📈 Visual Analysis")
                chart_type = st.selectbox("Select Chart", ["Scatter Plot", "Heatmap"])
                fig, ax = plt.subplots(figsize=(5, 3))
                if chart_type == "Scatter Plot":
                    sns.scatterplot(x=df[colx], y=df[coly], ax=ax)
                    plt.title(f"Scatter Plot: {colx} vs {coly}")
                elif chart_type == "Heatmap":
                    corr_matrix = df[[colx, coly]].corr()
                    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
                    plt.title(f"Heatmap: {colx} vs {coly}")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        # -----------------------------
        # Numerical vs Categorical
        # -----------------------------
        elif rel_type == "num_cat":
            num_cols = df.select_dtypes(include=['int', 'float']).columns
            cat_cols = df.select_dtypes(exclude=['int', 'float']).columns

            num_col = st.selectbox("Select Numerical Column", num_cols)
            cat_col = st.selectbox("Select Categorical Column", cat_cols)

            if ana_type == "Non-Visual":
                st.subheader("📋 Non-Visual Summary")
                grouped = df.groupby(cat_col)[num_col].agg(["count", "mean", "median", "std"]).reset_index()
                st.dataframe(grouped)
            else:
                st.subheader("📊 Visual Analysis")
                chart_type = st.multiselect(
                    "Select Charts",
                    ["Histogram", "Box Plot", "Violin Plot"],
                    default=["Box Plot"]
                )
                for c in chart_type:
                    fig, ax = plt.subplots(figsize=(5, 3))
                    if c == "Histogram":
                        sns.histplot(data=df, x=num_col, hue=cat_col, kde=True, ax=ax)
                        plt.title(f"Histogram of {num_col} by {cat_col}")
                    elif c == "Box Plot":
                        sns.boxplot(x=df[cat_col], y=df[num_col], ax=ax)
                        plt.xticks(rotation=45)
                        plt.title(f"Box Plot of {num_col} across {cat_col}")
                    elif c == "Violin Plot":
                        sns.violinplot(x=df[cat_col], y=df[num_col], ax=ax)
                        plt.xticks(rotation=45)
                        plt.title(f"Violin Plot of {num_col} across {cat_col}")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

        # -----------------------------
        # Categorical vs Categorical
        # -----------------------------
        elif rel_type == "cat_cat":
            cat_cols = df.select_dtypes(exclude=['int', 'float']).columns
            colx = st.selectbox("Select First Categorical Column", cat_cols)
            coly = st.selectbox("Select Second Categorical Column", cat_cols)

            if ana_type == "Non-Visual":
                st.subheader("📋 Non-Visual Summary")
                if colx == coly:
                    st.warning("⚠️ Same column selected — showing single column counts.")
                    st.dataframe(df[colx].value_counts().to_frame("Count"))
                else:
                    ct = pd.crosstab(df[colx], df[coly])
                    st.dataframe(ct)
            else:
                st.subheader("📊 Visual Analysis")
                fig, ax = plt.subplots(figsize=(5, 3))
                if colx == coly:
                    sns.countplot(x=df[colx], ax=ax)
                    plt.title(f"Count Plot of {colx}")
                else:
                    sns.countplot(x=df[colx], hue=df[coly], ax=ax)
                    plt.title(f"{colx} vs {coly}")
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
