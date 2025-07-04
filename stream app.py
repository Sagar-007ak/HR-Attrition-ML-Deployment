import streamlit as st
import pandas as pd
import joblib

# Load model, scaler, and features
model = joblib.load("hr_attrition_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("features.pkl")  # Pre-saved from training script

# Label encoding maps (used during training)
business_travel_map = {"Travel_Rarely": 2, "Travel_Frequently": 1, "Non-Travel": 0}
department_map = {"Sales": 2, "Research & Development": 1, "Human Resources": 0}
education_field_map = {
    "Life Sciences": 1, "Medical": 4, "Marketing": 2,
    "Technical Degree": 5, "Human Resources": 0, "Other": 3
}
gender_map = {"Male": 0, "Female": 1}
job_role_map = {
    "Sales Executive": 7, "Research Scientist": 6, "Laboratory Technician": 4,
    "Manufacturing Director": 1, "Healthcare Representative": 2, "Manager": 0,
    "Sales Representative": 8, "Research Director": 5, "Human Resources": 3
}
marital_status_map = {"Single": 2, "Married": 1, "Divorced": 0}
overtime_map = {"Yes": 1, "No": 0}

# App title
st.title("🔍 HR Attrition Prediction App")

# Input widgets
age = st.slider("Age", 18, 60)
business_travel = st.selectbox("Business Travel", list(business_travel_map.keys()))
daily_rate = st.number_input("Daily Rate", 100, 1500)
department = st.selectbox("Department", list(department_map.keys()))
distance = st.slider("Distance From Home", 1, 30)
education = st.slider("Education (1-5)", 1, 5)
education_field = st.selectbox("Education Field", list(education_field_map.keys()))
env_satisfaction = st.slider("Environment Satisfaction", 1, 4)
gender = st.selectbox("Gender", list(gender_map.keys()))
hourly_rate = st.number_input("Hourly Rate", 30, 100)
job_involvement = st.slider("Job Involvement", 1, 4)
job_level = st.slider("Job Level", 1, 5)
job_role = st.selectbox("Job Role", list(job_role_map.keys()))
job_satisfaction = st.slider("Job Satisfaction", 1, 4)
marital_status = st.selectbox("Marital Status", list(marital_status_map.keys()))
monthly_income = st.number_input("Monthly Income", 1000, 20000)
monthly_rate = st.number_input("Monthly Rate", 2000, 27000)
num_companies_worked = st.slider("Number of Companies Worked", 0, 10)
overtime = st.selectbox("OverTime", list(overtime_map.keys()))
percent_salary_hike = st.slider("Percent Salary Hike", 10, 25)
relationship_satisfaction = st.slider("Relationship Satisfaction", 1, 4)
stock_option_level = st.slider("Stock Option Level", 0, 3)
total_working_years = st.slider("Total Working Years", 0, 40)
training_times = st.slider("Training Times Last Year", 0, 6)
work_life_balance = st.slider("Work Life Balance", 1, 4)
years_at_company = st.slider("Years at Company", 0, 40)
years_in_current_role = st.slider("Years in Current Role", 0, 18)
years_since_last_promo = st.slider("Years Since Last Promotion", 0, 15)
years_with_curr_manager = st.slider("Years with Current Manager", 0, 17)

# When Predict is clicked
if st.button("Predict"):
    input_data_dict = {
        "Age": age,
        "BusinessTravel": business_travel_map[business_travel],
        "DailyRate": daily_rate,
        "Department": department_map[department],
        "DistanceFromHome": distance,
        "Education": education,
        "EducationField": education_field_map[education_field],
        "EnvironmentSatisfaction": env_satisfaction,
        "Gender": gender_map[gender],
        "HourlyRate": hourly_rate,
        "JobInvolvement": job_involvement,
        "JobLevel": job_level,
        "JobRole": job_role_map[job_role],
        "JobSatisfaction": job_satisfaction,
        "MaritalStatus": marital_status_map[marital_status],
        "MonthlyIncome": monthly_income,
        "MonthlyRate": monthly_rate,
        "NumCompaniesWorked": num_companies_worked,
        "OverTime": overtime_map[overtime],
        "PercentSalaryHike": percent_salary_hike,
        "RelationshipSatisfaction": relationship_satisfaction,
        "StockOptionLevel": stock_option_level,
        "TotalWorkingYears": total_working_years,
        "TrainingTimesLastYear": training_times,
        "WorkLifeBalance": work_life_balance,
        "YearsAtCompany": years_at_company,
        "YearsInCurrentRole": years_in_current_role,
        "YearsSinceLastPromotion": years_since_last_promo,
        "YearsWithCurrManager": years_with_curr_manager
    }

    # Arrange input features in correct order
    input_df = pd.DataFrame([[input_data_dict[col] for col in feature_columns]], columns=feature_columns)

    # Scale input and predict
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)

    st.success(f"🎯 Attrition Prediction: {'Yes' if prediction[0] == 1 else 'No'}")
