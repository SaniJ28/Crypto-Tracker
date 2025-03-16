# Delivery Time Prediction Model Using Delivery and Pickup Data
Both delivery and pickup datasets are crucial for optimizing delivery time predictions.
This project focuses on analyzing historical delivery and pickup data to develop an accurate predictive model for estimating delivery durations.


## Understanding the Dataset
The dataset we are working on is a combination of Delivery Dataset Containing delivery order details with timestamps, geolocation, and durations and Pickup Dataset Containing pickup order details with similar attributes.
Both datasets were preprocessed for consistency.


# Dataset Preprocessing
We cleaned the dataset by removing irrelevant columns like order_id, region_id, and courier_id. Missing values were handled by dropping rows with extensive nulls or filling them using statistical methods. Date columns were converted to datetime format for better time-based analysis.
# Convert time columns to datetime
df['accept_time'] = pd.to_datetime(df['accept_time'])
df['delivery_time'] = pd.to_datetime(df['delivery_time'])

**Feature Engineering**We created new features, such as accept_delivery_distance, calculated from geospatial coordinates, to capture the distance between pickup and delivery points.

**Handling Outliers and Correlation**Outliers were treated using the Interquartile Range (IQR), and features with a correlation above 0.85 were removed to prevent redundancy.
# Drop highly correlated features
high_corr_features = {col for i, col in enumerate(df.corr().columns) for j in range(i) if abs(df.corr().iloc[i, j]) > 0.85}
df.drop(columns=high_corr_features, inplace=True)

**Dimensionality Reduction**We applied Principal Component Analysis (PCA) to reduce data complexity while preserving key information.
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
df_pca = pd.DataFrame(pca.fit_transform(df.select_dtypes('number')))


## EDA
**Introduction:**
The dataset contains delivery and pickup data from multiple sources, comprising geospatial coordinates, time-based information, and derived features like delivery distance. Key variables include accept_gps_lat, accept_gps_lng, delivery_gps_lat, delivery_gps_lng, accept_time, delivery_time, and calculated metrics like accept_delivery_distance.

**Information of Dataset:**
Using value counts on the target variable, we found that the dataset is balanced, as both delivery and pickup categories have comparable instances. This eliminates the need for data balancing techniques.

**Univariate Analysis:**
Histograms and box plots were used to analyze the distribution of numeric features. Most features exhibited skewed distributions, particularly the geospatial distances, which showed long tails due to extreme delivery distances. Outliers were detected using the Interquartile Range (IQR) method and handled by capping them within acceptable limits.

**Descriptive Statistics:**
Using the describe() function, we observed the following key statistics:

Feature                	|Mean       |Std	    |Min	    |25%	    |50%	    |75%	    |Max
accept_delivery_distance|2.53 km	|1.82 km	|0.12 km	|1.10 km	|2.32 km	|3.45 km	    |12.40 km
delivery_time_diff	    |45.23 min	|22.45 min	|12.00 min	|32.00 min	|43.00 min	|56.00 min	|120.00 min

**Correlation Plot of Numerical Variables:**
A correlation matrix was generated to identify relationships between numerical features. We found strong positive correlations between geospatial coordinates and delivery distance, while other variables like delivery_time_diff showed low correlation with spatial features. Features with a correlation coefficient above 0.85 were dropped.

**Visualization of Variables:**
-Delivery Distance: Skewed distribution with a few long-distance outliers.
-Time Difference: Consistent delivery time across most orders with a few extreme values.
-Outliers: Outliers were observed in distance and time features and were treated using capping techniques.
-Geospatial Patterns: Delivery points are generally clustered within a specific region, with few distant deliveries.
These insights guided further preprocessing by refining outlier handling, reducing multicollinearity, and enhancing feature engineering.
 
 
## Preprocessing Again

**Datetime Conversion (Repeated)**accept_time and delivery_time were repeatedly converted to datetime format for consistency in time-based calculations and dataset merging.
**Merging Pickup and Delivery Datasets** the pickup dataset and delivery dataset were merged using order_id to create a unified dataset for comprehensive analysis.
**Handling Correlation (Repeated)** After merging, highly correlated features (correlation > 0.85) were identified and removed to avoid redundancy.
**Dimensionality Reduction (Repeated PCA)** PCA was reapplied after merging to reduce the complexity of the dataset while preserving the newly added information.


## Model Building

#### Metrics considered for Model Evaluation
**Accuracy, Precision, Recall, F1 Score, and ROC-AUC**
-Accuracy: Measures the proportion of correctly classified instances out of the total instances.
-Precision: Evaluates the proportion of true positive predictions out of all positive predictions, indicating the model’s ability to avoid false positives.
-Recall: Measures the proportion of true positive cases correctly identified by the model.
-F1 Score: Harmonic mean of Precision and Recall, providing a balanced measure between the two.
-ROC-AUC: Measures the model's ability to distinguish between classes across different thresholds.

#### Logistic Regression
- Logistic Regression models the probability of a binary outcome using the logistic function:
P(y) = 1 / (1 + e^-(A + Bx)), where A is the intercept and B is the coefficient.
-It outputs a probability score, which can be thresholded to make binary classifications.
-Used as a baseline model to compare performance against other algorithms.

#### Random Forest Classifier
- An ensemble method that builds multiple decision trees and combines their outputs to improve accuracy and reduce overfitting.
-Bagging and Feature Randomness: Bagging reduces variance by combining multiple models trained on random subsets, while feature randomness ensures each tree considers a random subset of features, increasing model diversity.
-It provides feature importance, which is useful for feature selection.
#### Linear Discriminant Analysis
- Linear Discriminant Analysis, or LDA, uses the information from both(selection and target) features to create a new axis and projects the data on to the new axis in such a way as to **minimizes the variance and maximizes the distance between the means of the two classes.**
-LDA projects the data onto a new axis that minimizes intra-class variance and maximizes inter-class distance.
-Unlike PCA (which is unsupervised), LDA is a supervised technique that uses class labels for dimensionality reduction.
Works best on large datasets where class separability is a key factor.


### Choosing the features
Feature selection was performed using different methods across the pipeline:
**Correlation Analysis:**
We identified and removed features with a high correlation coefficient (above 0.85) to reduce redundancy and multicollinearity.
correlation_matrix = df.corr()
high_corr_features = {col for i, col in enumerate(correlation_matrix.columns) for j in range(i) if abs(correlation_matrix.iloc[i, j]) > 0.85}
df.drop(columns=high_corr_features, inplace=True)

**Feature Importance from Random Forest:**
We used the Random Forest Classifier to extract feature importance and select the most influential variables for the model.
importances = rf_model.feature_importances_
important_features = [feature for feature, importance in zip(X_train.columns, importances) if importance > 0.01]
#### 1. Applying LDA on Selected Features
After feature selection, we used LDA to reduce dimensionality further and focus on features that provide the best class separation.
lda_transformed = lda.fit_transform(X_train[important_features], y_train)
#### 2. Applying XGBoost Classifier on Selected Features
XGBoost was implemented for its ability to handle large datasets efficiently and its robustness in capturing complex patterns through gradient boosting.

from xgboost import XGBClassifier
xgb_model = XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=10)
xgb_model.fit(X_train[important_features], y_train)
y_pred_xgb = xgb_model.predict(X_test[important_features])
print(classification_report(y_test, y_pred_xgb))
**Purpose:**
-Leverage gradient boosting for better model accuracy.
-Handle missing data and large-scale problems effectively.
### PCA Transformation
Principal Component Analysis (PCA) was applied after feature engineering to reduce dimensionality while preserving most of the dataset’s variance. This helped to mitigate the curse of dimensionality and improve model efficiency.

from sklearn.decomposition import PCA
pca = PCA(n_components=5)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
**Purpose:**
-Reduce computational complexity while retaining important patterns.
-Mitigate multicollinearity issues by creating uncorrelated principal components.


## Deployment
you can access our app by following this link [stock-price-application-streamlit](https://stock-price-2.herokuapp.com/) or by click [stock-price-application-flask](https://stock-price-flask.herokuapp.com/)
### Streamlit
- It is a tool that lets you creating applications for your machine learning model by using simple python code.
- We write a python code for our app using Streamlit; the app asks the user to enter the following data (**news data**, **Open**, **Close**).
- The output of our app will be 0 or 1 ; 0 indicates that stock price will decrease while 1 means increasing of stock price.
- The app runs on local host.
- To deploy it on the internt we have to deploy it to Heroku.

### Heroku
We deploy our Streamlit app to [ Heroku.com](https://www.heroku.com/). In this way, we can share our app on the internet with others. 
We prepared the needed files to deploy our app sucessfully:
- Procfile: contains run statements for app file and setup.sh.
- setup.sh: contains setup information.
- requirements.txt: contains the libraries must be downloaded by Heroku to run app file (stock_price_App_V1.py)  successfully 
- stock_price_App_V1.py: contains the python code of a Streamlit web app.
- stock_price_xg.pkl : contains our XGBClassifier model that built by modeling part.
- X_train2.npy: contains the train data of modeling part that will be used to apply PCA trnsformation to the input data of the app.

### Flask 
We also create our app   by using flask , then deployed it to Heroku . The files of this part are located into (Flask_deployment) folder. You can access the app by following this link : [stock-price-application-flask](https://stock-price-flask.herokuapp.com/)

