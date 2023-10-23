# Overview

This data science project focuses on building a binary classifier to predict customer churn for Syria Telecom, a telecommunications company. The primary objective is to reduce revenue loss by identifying customers who are likely to discontinue their services. The project will explore patterns within the available data to make proactive business decisions.

# Business Understanding

* Stakeholders: The primary stakeholders include Syria Telecom's management, customer service teams, and marketing departments. They will be directly affected by the outcomes of this project.
* Business Problem: The project aims to solve the problem of customer churn by identifying potential churners in advance, allowing the company to take targeted retention actions.
* Scope: This project will focus on building a predictive model for customer churn. It is within the scope of the project to gather, preprocess, and analyze relevant data to construct this model.

# Data Understanding

* Data Sources: *The data is obtained from Syria Telecom's and accessed through Kegel.*
* Data Access: *The data is publicly available for use by interested parties and will therefore require no authorizations.*
* Target Variable: *The target variable is "churn," which is binary (1 for churn, 0 for non-churn).*
* Predictors: *The predictors include customer account length, international plan status, voice mail plan status, number of voice mail messages, usage statistics for day, evening, and night, international usage statistics, and the number of customer service calls made.*
* Data Types: *The data includes numeric and categorical features.*
* Data Distribution: *The distribution of the data for each variable will be explored during data analysis.*
* Data Volume: *The dataset contains 21 features and 3333 records, sufficient for building a predictive model.*
* Data Quality: *Data quality will be verified during the preprocessing phase, and measures will be taken to address any potential issues.*

## Stakeholder audience choice
The primary stakeholders in this project are the management, customer service teams, and marketing departments within Syria Telecom. Their collaboration and input are crucial for the project's success.

## Dataset choice
The dataset used for this project is sourced from Syria Telecom through Keggle. It contains historical customer data with relevant features for the churn prediction.

# Modeling

The modeling phase of this project will involve the development of three different models, each with varying complexity and sophistication. These models will help us effectively predict customer churn for Syria Telecom.



## Simple Baseline Model (Logistic Regression):

In order to establish a baseline for our predictive modeling, we will begin with a simple yet interpretable model. Logistic regression is a logical choice for this. It will help us understand the basic relationships between predictors and customer churn.
We will assess the model's performance using standard evaluation metrics such as accuracy, precision, recall, and the ROC-AUC score.

## More-Complex Model (Random Forest):

To capture more complex patterns and interactions within the data, we will implement a random forest classifier. Random forests are an ensemble method that can handle both categorical and numeric features efficiently.
This model will be evaluated using the same metrics as the baseline model to compare performance.

## Tuned Hyperparameter Model (Random Forest with Tuned Hyperparameters):

The third model will be a refined version of the random forest model. We will use hyperparameter tuning techniques to optimize its performance. This will involve adjusting parameters such as the number of trees, maximum depth, and minimum samples required to split nodes.

Cross-validation and grid search will be used to identify the best hyperparameters.
We will compare the performance of the tuned model with the baseline and initial random forest models to assess the impact of hyperparameter tuning.

Throughout the modeling phase, model interpretability will be considered. Interpretable models are valuable for understanding which features have the most influence on predicting customer churn. This information can be used by stakeholders to make informed decisions on customer retention strategies.

The final model selected for deployment will be the one that demonstrates the best performance in terms of accurately predicting customer churn. Its predictive power, along with its interpretability, will assist Syria Telecom in identifying at-risk customers and implementing proactive measures to reduce churn and improve customer retention.

We start off with performing a Train Test Split
'''python 
# Import the relevant function
from sklearn.model_selection import train_test_split

# Split df into X and y
X = df.drop("Cover_Type", axis=1)
y = df["Cover_Type"]

# Perform train-test split with random_state=42 and stratify=y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
'''

we thereafter build and evaluate a baseline model

'''python
# Import relevant classes and functions
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import numpy as np

# Instantiate a LogisticRegression with random_state=42
baseline_model = LogisticRegression(random_state=42)

# Use cross_val_score with scoring="neg_log_loss" to evaluate the model on X_train and y_train
baseline_neg_log_loss_cv = cross_val_score(baseline_model, X_train, y_train, cv=5, scoring="neg_log_loss")

# Calculate the mean log loss by negating the mean of baseline_neg_log_loss_cv
baseline_log_loss = -(baseline_neg_log_loss_cv.mean())
baseline_log_loss
'''

We write a custom cross validation function


'''python
baseline_model = LogisticRegression(random_state=42)
baseline_neg_log_loss_cv = cross_val_score(baseline_model, X_train, y_train, scoring="neg_log_loss")
baseline_log_loss = -(baseline_neg_log_loss_cv.mean())
baseline_log_loss
'''
We thn use stratifiedKFold to provide the information we need to make seperate train test splits inside X_train

'''python
# Run this cell without changes
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

# Negative log loss doesn't exist as something we can import,
# but we can create it
neg_log_loss = make_scorer(log_loss, greater_is_better=False, needs_proba=True)

# Instantiate the model (same as previous example)
baseline_model = LogisticRegression(random_state=42)

# Create a list to hold the score from each fold
kfold_scores = np.ndarray(5)

# Instantiate a splitter object and loop over its result
kfold = StratifiedKFold()
for fold, (train_index, val_index) in enumerate(kfold.split(X_train, y_train)):
    # Extract train and validation subsets using the provided indices
    X_t, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
    y_t, y_val = y_train.iloc[train_index], y_train.iloc[val_index]
    
    # Clone the provided model and fit it on the train subset
    temp_model = clone(baseline_model)
    temp_model.fit(X_t, y_t)
    
    # Evaluate the provided model on the validation subset
    neg_log_loss_score = neg_log_loss(temp_model, X_val, y_val)
    kfold_scores[fold] = neg_log_loss_score
    
-(kfold_scores.mean())
'''

Using the custom cross validation function with stratifiedKFold

'''python
# Import relevant sklearn and imblearn classes
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def custom_cross_val_score(estimator, X, y):
    # Create a list to hold the scores from each fold
    kfold_train_scores = np.ndarray(5)
    kfold_val_scores = np.ndarray(5)

    # Instantiate a splitter object and loop over its result
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # Add shuffle and random_state
    for fold, (train_index, val_index) in enumerate(kfold.split(X, y)):
        # Extract train and validation subsets using the provided indices
        X_t, X_val = X.iloc[train_index], X.iloc[val_index]
        y_t, y_val = y.iloc[train_index], y.iloc[val_index]
        
        # Instantiate StandardScaler
        scaler = StandardScaler()  # Instantiate StandardScaler
        # Fit and transform X_t
        X_t_scaled = scaler.fit_transform(X_t)  # Fit and transform X_t
        # Transform X_val
        X_val_scaled = scaler.transform(X_val)  # Transform X_val
        
        # Instantiate SMOTE with random_state=42 and sampling_strategy=0.28
        sm = SMOTE(sampling_strategy=0.28, random_state=42)  # Instantiate SMOTE
        # Fit and transform X_t_scaled and y_t using sm
        X_t_oversampled, y_t_oversampled = sm.fit_resample(X_t_scaled, y_t)  # Fit and transform using SMOTE
        
        # Clone the provided model and fit it on the train subset
        temp_model = clone(estimator)
        temp_model.fit(X_t_oversampled, y_t_oversampled)
        
        # Evaluate the provided model on the train and validation subsets
        neg_log_loss_score_train = -log_loss(y_t_oversampled, temp_model.predict_proba(X_t_oversampled))
        neg_log_loss_score_val = -log_loss(y_val, temp_model.predict_proba(X_val_scaled))
        kfold_train_scores[fold] = neg_log_loss_score_train
        kfold_val_scores[fold] = neg_log_loss_score_val
        
    return kfold_train_scores, kfold_val_scores

model_with_preprocessing = LogisticRegression(random_state=42, class_weight={1: 0.28})
preprocessed_train_scores, preprocessed_neg_log_loss_cv = custom_cross_val_score(model_with_preprocessing, X_train, y_train)
-(preprocessed_neg_log_loss_cv.mean())
'''

comparing with the baseline log loss

'''python
# Run this cell without changes
print(-baseline_neg_log_loss_cv.mean())
print(-preprocessed_neg_log_loss_cv.mean())
'''

Evaluating our model using the custom cross val score
'''python
# Replace None with appropriate code
less_regularization_train_scores, less_regularization_val_scores = None

print("Previous Model")
print("Train average:     ", -preprocessed_train_scores.mean())
print("Validation average:", -preprocessed_neg_log_loss_cv.mean())
print("Current Model")
print("Train average:     ", -less_regularization_train_scores.mean())
print("Validation average:", -less_regularization_val_scores.mean())
'''

Chosing and evaluating a final model
'''python
# Run this cell without changes
final_model = model_less_regularization

# Instantiate StandardScaler
scaler = StandardScaler()
# Fit and transform X_train
X_train_scaled = scaler.fit_transform(X_train)
# Transform X_test
X_test_scaled = scaler.transform(X_test)

# Instantiate SMOTE with random_state=42 and sampling_strategy=0.28
sm = SMOTE(sampling_strategy=0.28, random_state=42)
# Fit and transform X_train_scaled and y_train using sm
X_train_oversampled, y_train_oversampled = sm.fit_resample(X_train_scaled, y_train)

'''

We fit the model on the full training data
'''python
# Run this cell without changes
final_model.fit(X_train_oversampled, y_train_oversampled)
'''
Evaluating the model on the test data
'''python
# Run this cell without changes
log_loss(y_test, final_model.predict_proba(X_test_scaled))
'''

# Evaluation
The dataset used for this project is sourced from Syria Telecom's internal database, which contains historical customer data with relevant features for the churn prediction task.

# Conclusion
This project aims to provide Syria Telecom with a predictive tool to identify customers at risk of churning. By addressing this issue proactively, the company can implement strategies to retain customers and reduce financial losses. Successful implementation of the predictive model will empower Syria Telecom to make data-driven decisions, ultimately improving customer retention and business profitability.