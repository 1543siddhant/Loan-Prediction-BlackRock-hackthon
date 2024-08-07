# Loan-Prediction-BlackRock-hackthon
Loan prediction system for Bankers to determine farmer's Loan approval. Ref-Black Rock (American multinational investment company) Hackathon. 

# Agricultural Financial Inclusion System

Welcome to the Agricultural Financial Inclusion System, a machine learning-based web application developed using Streamlit. This system predicts land prices and loan approval status for farmers based on various inputs such as land area, soil type, income, and more.

## Features

1. **Price Prediction**: Estimate the price of land based on its characteristics.
2. **Loan Approval Prediction**: Determine the likelihood of loan approval based on the farmer's income and loan request amount.

## Installation

To run this application locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/agricultural-financial-inclusion-system.git
    cd agricultural-financial-inclusion-system
    ```

2. Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Download the model files and place them in the root directory of the project:
    - `price_model.pkl`
    - `approval_model.pkl`

## Usage

To run the Streamlit application, use the following command:
```bash
streamlit run app.py
```
## Price and Loan Prediction
On this page, you can input the following details to get predictions:

Area of the land (in sq ft)

Land contour (flat, sloping, hilly)

Distance from the road (in meters)

Soil type (loamy, clay, sandy, silty)

Farmer's annual income (in INR)

Loan request amount (in INR)

Click the "Predict" button to get the following predictions:

Estimated land price
Estimated loan value
Estimated probability of loan approval
Loan approval status
Model Details
Price Prediction Model

## A Linear Regression model is used to predict the price of the land based on its characteristics. The input features for the model are:

Area
Distance from the road
Income
Land contour (encoded as dummy variables)
Soil type (encoded as dummy variables)
Loan Approval Model



