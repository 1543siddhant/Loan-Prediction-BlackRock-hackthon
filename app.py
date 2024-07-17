import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the models
price_model = joblib.load('price_model.pkl')  # Linear Regression model for price prediction
approval_model = joblib.load('approval_model.pkl')  # Logistic Regression model for loan approval prediction

# Streamlit app
st.title("Agricultural Financial Inclusion System")

st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "Price and Loan Prediction"])

# Home Page
if app_mode == "Home":
    st.header("Welcome to the Agricultural Financial Inclusion System")
    image_path = "home.jpeg"
    st.image(image_path, use_column_width=True)
    st.markdown("""
    ## Features:
    1. Price prediction based on land and farmer characteristics.
    2. Loan approval prediction based on farmer's income and loan request.
    """)

# Price and Loan Prediction Page
elif app_mode == "Price and Loan Prediction":
    st.header("Price and Loan Prediction")

    area = st.number_input("Enter area of the land (in sq ft):", min_value=1000, max_value=3000, step=100)
    land_contour = st.selectbox("Enter land contour:", ["flat", "sloping", "hilly"])
    distance_from_road = st.number_input("Enter distance from the road (in meters):", min_value=50, max_value=300, step=10)
    soil_type = st.selectbox("Enter soil type:", ["loamy", "clay", "sandy", "silty"])
    income = st.number_input("Enter farmer's annual income (in INR):", min_value=20000, max_value=100000, step=1000)
    loan_request = st.number_input("Enter loan request amount (in INR):", min_value=10000, max_value=1000000, step=10000)

    if st.button("Predict"):
        # Create input data for price prediction
        input_data_price = pd.DataFrame({
            'area': [area],
            'distance_from_road': [distance_from_road],
            'income': [income],
            'land_contour_hilly': [1 if land_contour == 'hilly' else 0],
            'land_contour_sloping': [1 if land_contour == 'sloping' else 0],
            'soil_type_clay': [1 if soil_type == 'clay' else 0],
            'soil_type_sandy': [1 if soil_type == 'sandy' else 0],
            'soil_type_silty': [1 if soil_type == 'silty' else 0]
        })

        # Add missing columns with value 0
        for column in price_model.feature_names_in_:
            if column not in input_data_price.columns:
                input_data_price[column] = 0

        # Ensure the order of columns matches the training data
        input_data_price = input_data_price[price_model.feature_names_in_]

        # Predict the farm price
        predicted_price = price_model.predict(input_data_price)[0]

        # Determine the loan value
        loan_value = predicted_price if predicted_price <= 500000 else predicted_price * 0.85

        # Calculate loan approval probability based on loan request
        if loan_request <= loan_value:
            approval_probability = 1.0  # 100% probability of loan approval
        else:
            diff_ratio = (loan_request - loan_value) / loan_value  # Calculate relative difference
            approval_probability = np.exp(-5 * diff_ratio)  # Exponential decrease in probability

        # Create input data for loan approval prediction
        input_data_approval = input_data_price.copy()
        input_data_approval['price'] = predicted_price

        # Add missing columns with value 0
        for column in approval_model.feature_names_in_:
            if column not in input_data_approval.columns:
                input_data_approval[column] = 0

        # Ensure the order of columns matches the training data
        input_data_approval = input_data_approval[approval_model.feature_names_in_]

        # Predict the loan approval
        approval_prediction = approval_model.predict(input_data_approval)[0]
        approval_result = "Approved" if approval_prediction == 1 else "Not Approved"

        # Display results
        st.success(f"Estimated land price: INR {predicted_price:,.2f}")
        st.success(f"Estimated loan value: INR {loan_value:,.2f}")
        st.success(f"Estimated probability of loan approval: {approval_probability * 100:.2f}%")
        st.success(f"Loan approval status: {approval_result}")

# Run the Streamlit app
if __name__ == '__main__':
    st.run()
