# Job Market Pulse - Real-Time Skill Demand and Salary Intelligence

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Live-red)
![ML](https://img.shields.io/badge/ML-Random%20Forest-green)
![Records](https://img.shields.io/badge/Records-90%2C000-orange)

## Live Dashboard
https://job-market-pulse-hqc6evyqecnn3fvhdbunup.streamlit.app/

## Problem Statement
Job seekers need data-backed answers to:
- Which skills are trending in the market?
- What is the salary range for a role and location?
- How does remote work affect pay?
- Which industries are hiring most?

## Project Overview
An end-to-end data analytics project analyzing 90,000 real global AI and data science job records
to provide actionable salary intelligence for job seekers and professionals.

## Key Insights
- Machine Learning and AI roles earn 5% above market average
- Data Analytics roles earn 28% below market average
- Generative AI and LLM are highest paying specializations in 2025-2026
- USA pays 30% more than most countries for data roles
- Retail and Gaming pay more than Finance and Tech
- Salary recovering strongly in 2025-2026 after 2024 dip

## Features
- Interactive filters by Experience, Work Type, Company Size, Industry
- Salary distribution analysis
- Experience level vs salary comparison
- Remote vs Onsite vs Hybrid pay gap analysis
- Skill category salary lift chart
- Top 10 countries by average salary
- Top 10 industries by average salary
- Top AI specializations by salary
- Salary trend over years (2020-2026)
- AI-powered salary predictor

## Tech Stack
- Python, Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn Random Forest
- Streamlit
- GitHub and Streamlit Cloud

## Model Performance
- Algorithm: Random Forest Regression
- Records: 90,000
- R2 Score: 0.54
- MAE: $21,769

## Project Structure
- app.py Streamlit dashboard
- salary_data_cleaned.csv Cleaned dataset
- salary_model.pkl Trained model
- encoders.pkl Label encoders
- requirements.txt Dependencies
- Job_Market_Pulse_Report.pdf Insights report

## How to Run Locally
pip install -r requirements.txt
streamlit run app.py

## Author
Shaik Althaf Basha - Aspiring Data Analyst
- GitHub: https://github.com/Althaf-725
- LinkedIn: https://linkedin.com/in/shaik-althaf-basha-690147321
- Portfolio: https://althaf-data-analyst-portfolio-rdx1.vercel.app/
