import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "best_tourism_package_prediction_model_v1.joblib")
model = joblib.load(model_path)

st.title("Tourism Package Prediction App")
st.write("""
This application predicts the likelihood of a customer purchasing a package based on its operational parameters.
Enter the data below to get a prediction.
""")

# Numerical Values
Age= st.number_input("Age",18,90,25,1)
CityTier = st.selectbox("City Tier", [1, 2, 3])
DurationOfPitch = st.number_input("Duration of Pitch",1,120,5,1)
NumberOfPersonVisiting = st.number_input("Number of Persons Visiting",1, 5, 1,1)
NumberOfFollowups = st.number_input("Number of Follow-ups",1, 6, 1,1)
PreferredPropertyStar = st.selectbox("Preferred Property Star", [ 3, 4, 5])
NumberOfTrips = st.number_input("Number of Trips (per year)",1, 30, 1,1)
Passport = st.selectbox("Has Passport?", [0,1])
PitchSatisfactionScore = st.selectbox("Pitch Satisfaction Score",[ 1,2,3, 4, 5])
OwnCar = st.selectbox("Own a Car?", [0,1])
NumberOfChildrenVisiting = st.number_input("Number of Children Visiting",0, 3, 0,1)
MonthlyIncome = st.number_input("Monthly Income",1000, 99999, 50000)

# Categorical values
TypeofContact = st.selectbox("TypeofContact", ["Self Enquiry", "Company Invited"])
Occupation = st.selectbox("Occupation", ['Salaried', 'Free Lancer', 'Small Business', 'Large Business'])
Gender = st.selectbox("Gender", ["Male", "Female"])
ProductPitched = st.selectbox("ProductPitched", ['Deluxe', 'Basic', 'Standard', 'Super Deluxe', 'King'])
MaritalStatus = st.selectbox("MaritalStatus", ["Single", "Married", "Divorced"])
Designation = st.selectbox("Designation",['Manager', 'Executive', 'Senior Manager', 'AVP', 'VP'])

input_data = pd.DataFrame([{
    "Age": Age,
    "CityTier": CityTier,
    "DurationOfPitch": DurationOfPitch,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "NumberOfFollowups": NumberOfFollowups,
    "PreferredPropertyStar": PreferredPropertyStar,
       "NumberOfTrips": NumberOfTrips,
    "Passport": Passport,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "OwnCar": OwnCar,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "MonthlyIncome": MonthlyIncome,
      "TypeofContact": TypeofContact,
    "Occupation": Occupation,
    "Gender": Gender,
    "ProductPitched": ProductPitched,
    "MaritalStatus": MaritalStatus,
    "Designation": Designation,
}])

if st.button("Predict Tourism Package Selection"):
    prediction = model.predict(input_data)[0]
    result = "Tourism Package Selected" if prediction == 1 else "Tourism Package Not Selected"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
