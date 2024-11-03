import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
import utils as ut
from openai import OpenAI

# Initializes OpenAI client
try:
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.environ.get('GROQ_API_KEY')
    )
except Exception as e:
    st.error("Failed to initialize OpenAI client. Please check the API key and base URL.")
    st.stop()

def load_model(filename):
    """Loads model from a pickle file and handles exceptions."""
    try:
        with open(filename, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        st.error(f"Model file '{filename}' not found. Please ensure the file is in the correct directory.")
        st.stop()
    except pickle.UnpicklingError:
        st.error(f"Failed to load model from '{filename}'. The file may be corrupted or in an incorrect format.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading the model: {str(e)}")
        st.stop()

# Loads models at the beginning
models = {
    'XGBoost': load_model('xgb_model.pkl'),
    'Random Forest': load_model('rf_model.pkl'),
    'K-Nearest Neighbors': load_model('knn_model.pkl'),
    'Voting Classifier': load_model('voting_Clf.pkl'),
    'SVM': load_model('svm_model.pkl'),
    'Decision Tree': load_model('bt_model.pkl'),
    'Naive Bayes': load_model('nb_model.pkl'),
    'XGBoost with SMOTE': load_model('xgboost_featureEngineered.pkl')
}

def prepare_input(credit_score, location, gender, age, tenure, balance,
                  num_of_products, has_credit_history, is_active_member, estimated_salary):
    """Prepares input data for prediction."""
    input_dict = {
        'credit_score': credit_score,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_of_products,
        'HasCrCard': int(has_credit_history),
        'IsActiveMember': int(is_active_member),
        'EstimatedSalary': estimated_salary,
        'Geography_France': int(location == 'France'),
        'Geography_Germany': int(location == 'Germany'),
        'Geography_Spain': int(location == 'Spain'),
        'Gender_Male': int(gender == 'Male'),
        'Gender_Female': int(gender == 'Female'),
    }
    return pd.DataFrame([input_dict]), input_dict

def calculate_customer_percentiles(df, selected_customer):
    """Calculates percentiles of selected customer's attributes."""
    try:
        metrics = ['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary']
        percentiles = {metric: round(np.percentile(df[metric], (df[metric] < selected_customer[metric]).mean() * 100), 2)
                       for metric in metrics}
        return percentiles
    except KeyError as e:
        st.error(f"Column {e} missing in data. Please check the dataset.")
        st.stop()
    except Exception as e:
        st.error(f"Error calculating percentiles: {str(e)}")
        st.stop()

def make_prediction(input_df):
    """Makes predictions using multiple models and displays results."""
    try:
        probabilities = {
            model_name: model.predict_proba(input_df)[0][1] for model_name, model in models.items()
        }
        avg_probability = np.mean(list(probabilities.values()))

        col1, col2 = st.columns(2)
        with col1:
            fig_probs = ut.create_gauge_chart(avg_probability)
            st.plotly_chart(fig_probs, use_container_width=True)
            st.write(f"The customer has a **{avg_probability:.2%}** probability of churning.")
        with col2:
            fig_probs = ut.create_model_probability_chart(probabilities)
            st.plotly_chart(fig_probs, use_container_width=True)

        return avg_probability
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None

def explain_prediction(probability, input_dict, surname):
    """Generates explanation for prediction using OpenAI API."""
    try:
        prompt = f"""
        You are an expert data scientist at a bank. Please provide an explanation for the churn prediction based on the following input:
        {input_dict} with a predicted probability of churn: {probability:.2%}.
        """
        raw_response = client.chat.completions.create(
            model="llama-3.2-3b-preview",
            messages=[{"role": "user", "content": prompt}]
        )
        return raw_response.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating explanation: {str(e)}")
        return "Explanation could not be generated."

def generate_email(probability, input_dict, explanation, surname):
    """Generates personalized email for customer using OpenAI API."""
    try:
        prompt = f"""
        As a manager at HS Bank, please generate a personalized email for the customer based on the following details:
        Probability of churn: {probability:.2%}
        Explanation: {explanation}
        Customer Input: {input_dict}
        """
        raw_response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}]
        )
        return raw_response.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating email: {str(e)}")
        return "Email could not be generated."

st.title("📊 Customer Churn Prediction")
st.markdown("<hr style='border:1px solid #ddd;'>", unsafe_allow_html=True)

# Load customer data
try:
    df = pd.read_csv("churn.csv")
except FileNotFoundError:
    st.error("Data file 'churn.csv' not found. Please upload the file and try again.")
    st.stop()
except pd.errors.EmptyDataError:
    st.error("Data file 'churn.csv' is empty. Please provide a valid dataset.")
    st.stop()
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.stop()

customers = [f"{row['CustomerId']} - {row['Surname']}" for _, row in df.iterrows()]
selected_customer_option = st.selectbox("Select a Customer", customers)

if selected_customer_option:
    selected_customer_id = int(selected_customer_option.split(" - ")[0])
    selected_customer = df.loc[df["CustomerId"] == selected_customer_id].iloc[0]

    # Collects customer details and prepares input
    input_df, input_dict = prepare_input(
        selected_customer['CreditScore'],
        selected_customer['Geography'],
        selected_customer['Gender'],
        selected_customer['Age'],
        selected_customer['Tenure'],
        selected_customer['Balance'],
        selected_customer['NumOfProducts'],
        selected_customer['HasCrCard'],
        selected_customer['IsActiveMember'],
        selected_customer['EstimatedSalary']
    )

    if st.button("Predict Churn"):
        avg_probability = make_prediction(input_df)
        if avg_probability is not None:
            percentiles = calculate_customer_percentiles(df, selected_customer)
            fig_percentiles = ut.create_percentile_bar_chart(percentiles)
            st.plotly_chart(fig_percentiles, use_container_width=True)

            explanation = explain_prediction(avg_probability, input_dict, selected_customer["Surname"])
            st.subheader("📝 Explanation of Predictions")
            st.markdown(explanation)

            email = generate_email(avg_probability, input_dict, explanation, selected_customer["Surname"])
            st.subheader("📧 Personalized Email")
            st.markdown(email)

