# headstarter_customer-ai-churn-prediction
Headstarter AI Software Engineering Accelerator - Project 1

---

# 🚀 📊 Customer Churn Prediction with Machine Learning

This project leverages advanced machine learning techniques to predict which customers are likely to stop using a service, also known as customer churn. The solution analyzes various customer behavioral and demographic data points—such as age, account balance, and activity history—to identify patterns associated with churn. By training multiple models, including **XGBoost, Random Forest, and SVM**, the project selects the best-performing model for deployment. The final model is deployed through an interactive **web app**, providing real-time predictions and actionable insights. In addition to predictions, the app offers personalized explanations of the factors driving each customer’s risk of churning. Businesses can use these insights to implement targeted retention strategies, boosting customer satisfaction and loyalty through automated engagement, such as personalized email campaigns.

---

## 🛠️ Features

| Feature                                | Description                                      |
|----------------------------------------|--------------------------------------------------|
| Data Preprocessing                     | Handle missing values, encode categorical data, and normalize numerical features. |
| EDA (Exploratory Data Analysis)       | Visualize data distributions and analyze correlations to gain insights. |
| Model Training                         | Train **5 different machine learning models** for churn prediction. |
| Hyperparameter Tuning                  | Use **grid search** to optimize the models. |
| Web App Deployment                     | Serve the best-performing model through a web application for real-time predictions. |
| Churn Explanation                      | Provide feature importance insights and personalized recommendations. |
| Email Generation                       | Generate engaging emails to encourage at-risk customers to stay. |

---

## 🚀 Tech Stack

| Technology          | Description                                          |
|---------------------|------------------------------------------------------|
| Python              | Programming language for data processing and model building. |
| Pandas & NumPy      | Libraries for data manipulation and numerical operations. |
| Scikit-Learn & XGBoost | Tools for machine learning and model building. |
| Seaborn & Matplotlib | Visualization libraries for EDA.                  |
| Streamlit           | For deploying the web app and interacting with predictions. |
| OpenAI API          | For generating explanations and personalized customer emails. |
| Groq API            | For interacting with models via Groq's accelerated AI API services. |
| Jupyter Notebook     | For iterative development and experimentation.      |
| Pickle              | For saving and loading trained machine learning models. |

---

## 🏆 Results

The models were evaluated using the following metrics:

| Model                       | Accuracy | Precision (Class 1 - Churn) | Recall (Class 1 - Churn) | F1-score (Class 1 - Churn) |
|-----------------------------|----------|-----------------------------|---------------------------|-----------------------------|
| **XGBoost Classifier**      | 85.4%    | 67%                         | 50%                       | 0.58                        |
| **Random Forest Classifier**| 86.4%    | 75%                         | 46%                       | 0.57                        |
| **Decision Tree Classifier**| 78.5%    | 46%                         | 51%                       | 0.48                        |
| **K-Nearest Neighbors (KNN)**| 82.4%   | 59%                         | 36%                       | 0.44                        |
| **Support Vector Classifier (SVM)**| 85.6% | 77%                     | 38%                       | 0.51                        |
| **Gaussian Naive Bayes**   | 81.8%    | 56%                         | 38%                       | 0.45                        |
| **Voting Classifier**       | 85.3%    | 63%                         | 59%                       | 0.61                        |

The **Random Forest Classifier** and **XGBoost Classifier** achieved the highest accuracies, with XGBoost also providing better precision for churn cases. Ultimately, the **XGBoost model** was selected for deployment due to its balance of accuracy and interpretability.

---

## 🧪 Tests

| Test Type            | Description                                          |
|----------------------|----------------------------------------------------|
| Unit Tests           | Test individual functions for expected outputs.    |
| Integration Tests    | Ensure that the entire application works together as expected. |
| Performance Tests    | Evaluate the responsiveness of the web application under load. |

---

## 📦 Deployment Specifications

| Specification        | Details                                             |
|----------------------|----------------------------------------------------|
| Environment          | Python 3.8 or higher                               |
| Dependencies         | Flask, Streamlit, scikit-learn, XGBoost, etc.     |
| Deployment Method    | Deploy via Streamlit Sharing or Heroku.           |
| Access               | URL will be provided upon deployment.             |

---

## 📋 Installation

Provide instructions on how to install and run your project locally.

```bash
# Clone the repository
git clone https://github.com/bshongwe/headstarter_customer-ai-churn-prediction.git

# Navigate to the project directory
cd headstarter_customer-ai-churn-prediction

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

---

## 📝 Usage

Describe how to use your application, including how to access the web app and any required input data formats.

---

## 🗝️ License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
