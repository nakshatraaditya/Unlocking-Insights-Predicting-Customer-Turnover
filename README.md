# Customer Churn Prediction

This project aims to predict customer churn for a business, helping to identify customers who are likely to leave or discontinue their service. By analyzing customer data and using machine learning models, this project provides insights to improve customer retention strategies.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

Customer churn is a critical issue for businesses as retaining existing customers is often more cost-effective than acquiring new ones. This project uses historical customer data to build predictive models that estimate the likelihood of a customer leaving. The goal is to provide actionable insights for improving customer retention and optimizing marketing strategies.

## Dataset

The dataset used in this project contains various features related to customer behavior and service usage. It includes:
- **Customer ID**: Unique identifier for each customer
- **Demographic Information**: Age, gender, location
- **Service Usage**: Number of service requests, duration of service
- **Churn Label**: Whether the customer has churned (Yes/No)


## Installation

To set up the project, follow these steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/nakshatraaditya/customer-churn-prediction.git
    cd customer-churn-prediction
    ```

2. Create a virtual environment:
    ```sh
    python -m venv venv
    ```

3. Activate the virtual environment:
    - On Windows:
        ```sh
        venv\Scripts\activate
        ```
    - On macOS/Linux:
        ```sh
        source venv/bin/activate
        ```

4. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. **Load and Preprocess Data**:
    ```python
   import pandas as pd
   import numpy as np
   import plotly.express as px
   import matplotlib.pyplot as plt

   data_df = pd.read_csv('/content/churn (2).csv')
   data_df.head()

    ```

2. **Train and Evaluate Models**:
    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import accuracy_score, precision_score, f1_score
   from sklearn.model_selection import train_test_split

  
   X = data_df.drop('Churn', axis=1)
   y = data_df['Churn']
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)

   def modeling(alg, alg_name, params={}):
    model = alg(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    def print_scores(alg_name, y_true, y_pred):
        acc_score = accuracy_score(y_true, y_pred)
        pre_score = precision_score(y_true, y_pred)
        f_score = f1_score(y_true, y_pred, average='weighted')

        print(f"{alg_name}")
        print(f"accuracy: {acc_score:.10f}")
        print(f"precision: {pre_score:.10f}")
        print(f"f1_score: {f_score:.10f}")

    print_scores(alg_name, y_test, y_pred)
    return model


   model = modeling(GradientBoostingClassifier, "Gradient Boosting Classifier", params=gbm_params)

    ```

3. **Results and Analysis**:
    - Review the model's accuracy and classification report to understand its performance.
    - Analyze the features contributing to customer churn and potential areas for improvement.

## Model Details

- **Model Used**: Logistic Regression, GradientDescentClassifier, RandomForestClassifier, SVC
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1 Score

## Results

The model(GradientDescentClassifier) achieves an accuracy of approximately **91.1%** on the test data. The classification report provides additional metrics like precision, recall, and F1-score, which offer insights into the model's performance.

## Contributing

Contributions are welcome! If you have suggestions for improvements or additional features, please feel free to open an issue or submit a pull request.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/yourFeature`)
3. Commit your changes (`git commit -m 'Add new feature or fix bug'`)
4. Push to the branch (`git push origin feature/yourFeature`)
5. Open a pull request


