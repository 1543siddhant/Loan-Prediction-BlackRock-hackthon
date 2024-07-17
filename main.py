import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
import numpy as np
import joblib

# Generate a larger synthetic dataset
np.random.seed(0)

areas = np.random.randint(1000, 3000, size=50)
land_contours = np.random.choice(['flat', 'sloping', 'hilly'], size=50)
distances_from_road = np.random.randint(50, 300, size=50)
soil_types = np.random.choice(['loamy', 'clay', 'sandy', 'silty'], size=50)
prices = (areas * 120) + (distances_from_road * -100) + \
         (np.random.choice([0, 10000, 20000], size=50)) + \
         (np.random.choice([5000, 10000, 15000], size=50))
incomes = np.random.randint(20000, 100000, size=50)
loan_approval = (incomes / 10 > prices / 20).astype(int)  # Simple rule for loan approval

data = {
    'area': areas,
    'land_contour': land_contours,
    'distance_from_road': distances_from_road,
    'soil_type': soil_types,
    'price': prices,
    'income': incomes,
    'loan_approval': loan_approval
}
df = pd.DataFrame(data)

# Encode categorical features 'land_contour' and 'soil_type'
df = pd.get_dummies(df, columns=['land_contour', 'soil_type'], drop_first=True)

# Define features (X) and target variables (y)
X_price = df.drop(['price', 'loan_approval'], axis=1)
y_price = df['price']
X_approval = df.drop('loan_approval', axis=1)
y_approval = df['loan_approval']

# Train the Linear Regression model for price prediction
price_model = LinearRegression()
price_model.fit(X_price, y_price)

# Train the Logistic Regression model for loan approval prediction
approval_model = LogisticRegression()
approval_model.fit(X_approval, y_approval)

# Save the models
joblib.dump(price_model, 'price_model.pkl')
joblib.dump(approval_model, 'approval_model.pkl')

# Streamlit app
st.title("Farm Loan Prediction")

area = st.number_input("Enter area of the land:", min_value=0.0, step=1.0)
land_contour = st.selectbox("Enter land contour:", ['flat', 'sloping', 'hilly'])
distance_from_road = st.number_input("Enter distance from the road:", min_value=0.0, step=1.0)
soil_type = st.selectbox("Enter soil type:", ['loamy', 'clay', 'sandy', 'silty'])
income = st.number_input("Enter farmer's annual income:", min_value=0.0, step=1.0)
loan_request = st.number_input("Enter loan request amount:", min_value=0.0, step=1.0)

if st.button("Predict"):
    # Load the models
    price_model = joblib.load('price_model.pkl')
    approval_model = joblib.load('approval_model.pkl')

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
    for column in X_price.columns:
        if column not in input_data_price.columns:
            input_data_price[column] = 0

    # Ensure the order of columns matches the training data
    input_data_price = input_data_price[X_price.columns]

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
    for column in X_approval.columns:
        if column not in input_data_approval.columns:
            input_data_approval[column] = 0

    # Ensure the order of columns matches the training data
    input_data_approval = input_data_approval[X_approval.columns]

    # Print the results
    st.write("Estimated land price:", predicted_price)
    st.write("Estimated loan value:", loan_value)
    st.write("Estimated probability of loan approval: {:.2f}%".format(approval_probability * 100))
