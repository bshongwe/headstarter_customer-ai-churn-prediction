import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import main  # Assuming your main file is named main.py


class TestChurnPredictionApp(unittest.TestCase):

    @patch('main.pd.read_csv')
    def test_load_data_success(self, mock_read_csv):
        # Mocking read_csv to return a sample DataFrame
        mock_read_csv.return_value = pd.DataFrame({
            'CustomerId': [1, 2],
            'Surname': ['Smith', 'Doe'],
            'CreditScore': [600, 700],
            'Age': [40, 50],
            'Tenure': [5, 10],
            'Balance': [50000, 60000],
            'EstimatedSalary': [40000, 45000],
            'NumOfProducts': [1, 2],
            'HasCrCard': [1, 1],
            'IsActiveMember': [1, 0],
            'Geography': ['France', 'Germany'],
            'Gender': ['Male', 'Female']
        })

        df = main.load_data()
        self.assertEqual(len(df), 2)

    @patch('main.pd.read_csv', side_effect=FileNotFoundError)
    def test_load_data_file_not_found(self, mock_read_csv):
        with self.assertRaises(SystemExit):
            main.load_data()

    @patch('main.pickle.load')
    @patch('builtins.open', new_callable=MagicMock)
    def test_load_model_success(self, mock_open, mock_pickle_load):
        mock_open.return_value.__enter__.return_value = MagicMock()
        mock_pickle_load.return_value = MagicMock()  # Mock the model

        model = main.load_model('model.pkl')
        self.assertIsNotNone(model)

    @patch('main.pickle.load', side_effect=FileNotFoundError)
    def test_load_model_file_not_found(self, mock_pickle_load):
        with self.assertRaises(SystemExit):
            main.load_model('model.pkl')

    @patch('main.np.percentile')
    def test_calculate_customer_percentiles_success(self, mock_percentile):
        df = pd.DataFrame({
            'CreditScore': [600, 700],
            'Age': [40, 50],
            'Tenure': [5, 10],
            'Balance': [50000, 60000],
            'EstimatedSalary': [40000, 45000],
        })
        selected_customer = {'CreditScore': 650, 'Age': 45, 'Tenure': 7, 'Balance': 55000, 'EstimatedSalary': 42000}

        mock_percentile.return_value = 50
        percentiles = main.calculate_customer_percentiles(df, selected_customer)

        self.assertEqual(percentiles['CreditScore'], 50)
        self.assertEqual(percentiles['Age'], 50)

    @patch('main.np.mean')
    @patch('main.xgboost_model.predict_proba', return_value=np.array([[0, 1]]))
    @patch('main.random_forest_model.predict_proba', return_value=np.array([[0, 1]]))
    @patch('main.knn_model.predict_proba', return_value=np.array([[0, 1]]))
    def test_make_prediction_success(self, mock_knn, mock_rf, mock_xgb, mock_mean):
        input_df = pd.DataFrame({'CreditScore': [600], 'Age': [40]})
        input_dict = {}
        
        mock_mean.return_value = 0.8
        avg_probability = main.make_prediction(input_df, input_dict)

        self.assertAlmostEqual(avg_probability, 0.8)

    @patch('main.client.chat.completions.create')
    def test_explain_prediction_success(self, mock_create):
        mock_create.return_value.choices[0].message.content = "Customer is likely to churn due to reasons..."
        
        explanation = main.explain_prediction(0.85, {}, "Doe")
        self.assertIn("Customer is likely to churn", explanation)

    @patch('main.client.chat.completions.create')
    def test_generate_email_success(self, mock_create):
        mock_create.return_value.choices[0].message.content = "Dear customer, we noticed..."
        
        email = main.generate_email(0.85, {}, "Explanation", "Doe")
        self.assertIn("Dear customer", email)


if __name__ == '__main__':
    unittest.main()
