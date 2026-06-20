
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np

st.set_page_config(page_title="Job Market Pulse", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv('salary_data_cleaned.csv')

@st.cache_resource
def load_model():
    model = joblib.load('salary_model.pkl')
    encoders = joblib.load('encoders.pkl')
    return model, encoders

df = load_data()
rf_model, encoders = load_model()

st.sidebar.title("Filters")
experience = st.sidebar.multiselect("Experience Level", options=df['experience_level'].unique().tolist(), default=df['experience_level'].unique().tolist())
remote = st.sidebar.multiselect("Work Type", options=df['remote_ratio'].unique().tolist(), default=df['remote_ratio'].unique().tolist())
company_size = st.sidebar.multiselect("Company Size", options=df['company_size'].unique().tolist(), default=df['company_size'].unique().tolist())
industry = st.sidebar.multiselect("Industry", options=sorted(df['industry'].unique().tolist()), default=df['industry'].unique().tolist())

filtered = df[
    df['experience_level'].isin(experience) &
    df['remote_ratio'].isin(remote) &
    df['company_size'].isin(company_size) &
    df['industry'].isin(industry)
]

st.title("Job Market Pulse")
st.markdown("Real-time skill demand and salary intelligence for data and AI professionals")
st.markdown("---")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Avg Salary USD", f"${filtered['salary_in_usd'].mean():,.0f}")
col2.metric("Total Records", f"{len(filtered):,}")
col3.metric("Top Job Role", filtered['job_title'].value_counts().index[0])
col4.metric("Top Country", filtered['country'].value_counts().index[0])
st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Salary Distribution")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(filtered['salary_in_usd'], bins=30, kde=True, color='steelblue', ax=ax)
    ax.set_xlabel("Salary USD")
    st.pyplot(fig)
    plt.close()

with col2:
    st.subheader("Salary by Experience Level")
    order = ['Entry','Mid','Senior','Lead']
    order = [o for o in order if o in filtered['experience_level'].unique()]
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=filtered, x='experience_level', y='salary_in_usd', order=order, palette='Blues', ax=ax)
    ax.set_xlabel("Experience")
    st.pyplot(fig)
    plt.close()

col1, col2 = st.columns(2)
with col1:
    st.subheader("Remote vs Onsite Pay")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=filtered, x='remote_ratio', y='salary_in_usd', order=['Onsite','Hybrid','Remote'], palette='Set2', ax=ax)
    ax.set_xlabel("Work Type")
    st.pyplot(fig)
    plt.close()

with col2:
    st.subheader("Skill Category Salary Lift")
    overall_avg = df['salary_in_usd'].mean()
    skill_salary = filtered.groupby('skill_category')['salary_in_usd'].mean()
    skill_lift = ((skill_salary - overall_avg) / overall_avg * 100).round(1).sort_values()
    colors = ['#e74c3c' if x < 0 else '#2ecc71' for x in skill_lift.values]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(skill_lift.index, skill_lift.values, color=colors)
    ax.axvline(x=0, color='black', linewidth=1, linestyle='--')
    ax.set_xlabel("Salary Lift %")
    st.pyplot(fig)
    plt.close()

col1, col2 = st.columns(2)
with col1:
    st.subheader("Top 10 Countries by Avg Salary")
    top_countries = filtered.groupby('country')['salary_in_usd'].mean().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(x=top_countries.values, y=top_countries.index, palette='magma', ax=ax)
    ax.set_xlabel("Avg Salary USD")
    st.pyplot(fig)
    plt.close()

with col2:
    st.subheader("Top 10 Industries by Avg Salary")
    top_industries = filtered.groupby('industry')['salary_in_usd'].mean().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(x=top_industries.values, y=top_industries.index, palette='viridis', ax=ax)
    ax.set_xlabel("Avg Salary USD")
    st.pyplot(fig)
    plt.close()

col1, col2 = st.columns(2)
with col1:
    st.subheader("Top AI Specializations")
    top_specs = filtered.groupby('ai_specialization')['salary_in_usd'].mean().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(x=top_specs.values, y=top_specs.index, palette='coolwarm', ax=ax)
    ax.set_xlabel("Avg Salary USD")
    st.pyplot(fig)
    plt.close()

with col2:
    st.subheader("Salary Trend Over Years")
    yearly = filtered.groupby('year')['salary_in_usd'].median().reset_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.lineplot(data=yearly, x='year', y='salary_in_usd', marker='o', linewidth=2.5, color='steelblue', ax=ax)
    ax.set_xlabel("Year")
    st.pyplot(fig)
    plt.close()

st.markdown("---")
st.subheader("Salary Predictor")
st.markdown("Fill in your profile to get an estimated salary:")

col1, col2, col3, col4 = st.columns(4)
with col1:
    exp_input = st.selectbox("Experience Level", ['Entry','Mid','Senior','Lead'])
with col2:
    remote_input = st.selectbox("Work Type", ['Onsite','Hybrid','Remote'])
with col3:
    size_input = st.selectbox("Company Size", ['Small','Startup','Medium','Large','Enterprise'])
with col4:
    skill_input = st.selectbox("Your Field", ['Data Analytics','Data Engineering','Data Science','Machine Learning / AI','Other'])

year_input = st.slider("Year", 2020, 2026, 2025)

if st.button("Predict My Salary", type="primary"):
    try:
        features = np.array([[
            encoders['experience_level'].transform([exp_input])[0],
            encoders['remote_ratio'].transform([remote_input])[0],
            encoders['company_size'].transform([size_input])[0],
            encoders['skill_category'].transform([skill_input])[0],
            year_input
        ]])
        prediction = rf_model.predict(features)[0]
        st.success(f"Estimated Salary: ${prediction:,.0f} USD/year")
        st.caption("Based on Random Forest model trained on 90,000 real job records - R2 = 0.54")
    except Exception as e:
        st.error(f"Prediction error: {e}")

st.markdown("---")
st.caption("Built by Althaf - Job Market Pulse - Data source: Global AI Jobs Dataset 90000 records")
