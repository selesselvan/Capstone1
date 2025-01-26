# -*- coding: utf-8 -*-




# load data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

data = pd.read_csv('global air pollution dataset.csv')

print(data.head(10))

data.dtypes

"""# Exploratory data analysis"""

data.shape

data.head().T

data.duplicated().sum()

data.columns

for col in data.columns:
  print(col)
  print(data[col].unique()[:10])
  print(data[col].nunique())
  print()

data.describe()

data.isnull().sum()

# Drop rows with missing values
data = data.dropna()

# Shape after dropping missing values
print("Shape after dropping missing values:", data.shape)

# Verify
print(data.isnull().sum())

# Categorical columns
data.describe(include=['object'])



data['AQI Category'].value_counts()

# Distribution of AQI Categories
data['AQI Category'].value_counts().plot.bar()
plt.title('Distribution of AQI Category')
plt.show()

# Scatter plot
data = data.sort_values('AQI Value')

plt.figure(figsize=(14, 8))
for category in data['AQI Category'].unique():
    subset = data[data['AQI Category'] == category]
    plt.scatter(subset['AQI Value'], [category] * len(subset), label=category)

plt.xlabel('AQI Value')
plt.ylabel('AQI Category')
plt.title('AQI Categories vs AQI Values')
plt.legend()
plt.grid(True)
plt.show()

categories = data['AQI Category'].unique()
for category in categories:
    min_value = data[data['AQI Category'] == category]['AQI Value'].min()
    max_value = data[data['AQI Category'] == category]['AQI Value'].max()
    print(f"{category}: Min AQI = {min_value}, Max AQI = {max_value}")

"""**In summary:**

Good Air Quality: An AQI value of 50 or below indicates good air quality.

Unhealthy Air Quality: When the AQI exceeds 100, the air quality becomes unhealthy, posing a greater health risk.

Breakdown of cities by air quality:


*   Good Air Quality: 9,688 cities
*   Moderate Air Quality: 9,087 cities
*   Unhealthy Air Quality: 2,215 cities
*   Very Unhealthy Air Quality: 286 cities
*   Hazardous Air Quality: 191 cities







"""

maximum=data[data['AQI Value']==data['AQI Value'].max()]
maximum

maximum['Country'].value_counts()

maximum['Country'].value_counts().plot.bar()

Good_AQI=data[data['AQI Value'] <=50]
Good_AQI

Good_AQI['Country'].value_counts()

Good_AQI['Country'].value_counts().head(25).plot.bar()
plt.title('Country with Good AQI')
plt.ylabel('No.of cities')
plt.show()

"""**To sum up:**

* Cities with Maximum AQI (500): 103 cities have the maximum AQI value of 500.

* Of these, 95 cities are in India, 5 in Pakistan, and 1 each in the United States, South Africa, and the Russian Federation.

**Cities with Good Air Quality:**

* Brazil leads with 1125 cities having good air quality.
* Russian Federation ranks second.
* The United States and Germany are in third and fourth place, respectively.
"""

# Correlation heatmap for numerical features
numeric_columns = ['AQI Value', 'CO AQI Value', 'Ozone AQI Value', 'NO2 AQI Value', 'PM2.5 AQI Value']
correlation_matrix = data[numeric_columns].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of AQI Values')
plt.tight_layout()
plt.show()

"""1. PM2.5 is the strongest contributor to AQI (correlation: 0.98).
2. CO has a moderate impact on AQI (correlation: 0.43).
3. Ozone and NO2 have weaker correlations with AQI (0.41 and 0.23, respectively).
4. Ozone and NO2 show an inverse relationship (correlation: -0.18).
5. PM2.5 dominates AQI, while other pollutants have a smaller influence.

#Data Preprocessing
"""

data.info()

data.head(10)

data.shape




le_country = LabelEncoder()
le_city = LabelEncoder()
le_co_aqi_category = LabelEncoder()
le_ozone_aqi_category = LabelEncoder()
le_no2_aqi_category = LabelEncoder()
le_pm25_aqi_category = LabelEncoder()

data['Country'] = le_country.fit_transform(data['Country'])
data['City'] = le_city.fit_transform(data['City'])
data['CO AQI Category'] = le_co_aqi_category.fit_transform(data['CO AQI Category'])
data['Ozone AQI Category'] = le_ozone_aqi_category.fit_transform(data['Ozone AQI Category'])
data['NO2 AQI Category'] = le_no2_aqi_category.fit_transform(data['NO2 AQI Category'])
data['PM2.5 AQI Category'] = le_pm25_aqi_category.fit_transform(data['PM2.5 AQI Category'])


# target variable 'AQI Category'
le_category = LabelEncoder()
data['AQI Category'] = le_category.fit_transform(data['AQI Category'])

data.info()

data["AQI Category"].value_counts()

# Saving mapping for future reference
# country_mapping = dict(zip(le_country.classes_, le_country.transform(le_country.classes_)))
# city_mapping = dict(zip(le_city.classes_, le_city.transform(le_city.classes_)))
# category_mapping = dict(zip(le_category.classes_, le_category.transform(le_category.classes_)))
# co_aqi_category_mapping = dict(zip(le_co_aqi_category.classes_, le_co_aqi_category.transform(le_co_aqi_category.classes_)))
# ozone_aqi_category_mapping = dict(zip(le_ozone_aqi_category.classes_, le_ozone_aqi_category.transform(le_ozone_aqi_category.classes_)))
# no2_aqi_category_mapping = dict(zip(le_no2_aqi_category.classes_, le_no2_aqi_category.transform(le_no2_aqi_category.classes_)))
# pm25_aqi_category_mapping = dict(zip(le_pm25_aqi_category.classes_, le_pm25_aqi_category.transform(le_pm25_aqi_category.classes_)))


# print("AQI Category Mapping:", category_mapping)

"""# Feature importance analysis"""

correlation_matrix = data.corr(method='pearson')

plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()

"""**Results:**

* PM2.5 AQI is the dominant contributor to overall AQI, with a near-perfect correlation- 0.98.
* CO and ozone AQI values also moderately influence AQI, while NO2 plays a smaller role.
* AQI categories strongly correlate with their respective pollutant values, particularly for PM2.5 and ozone.






"""

# Mutual Information measures the mutual dependence between two variables( between each feature and the target variable.)






# Separate features and target
X = data.drop('AQI Category', axis=1)
y = data['AQI Category']

# Mutual information scores
mi_scores = mutual_info_classif(X, y)

# Dataframe of features and their MI scores
mi_scores_df = pd.DataFrame({'Feature': X.columns, 'MI Score': mi_scores})
mi_scores_df = mi_scores_df.sort_values('MI Score', ascending=False).reset_index(drop=True)

# Plot
plt.figure(figsize=(12, 8))
plt.bar(mi_scores_df['Feature'], mi_scores_df['MI Score'])
plt.xticks(rotation=90)
plt.title('Mutual Information Scores')
plt.xlabel('Features')
plt.ylabel('MI Score')
plt.tight_layout()
plt.show()

# Print MI scores
print(mi_scores_df)

"""**Results:**

* AQI Value, PM2.5 AQI Value, and PM2.5 AQI Category have the highest MI scores,- they are the most informative features for predicting the target variable.
* Country and CO AQI Value show moderate importance.
* Features like NO2 AQI Category and CO AQI Category have very low scores(they provide less to no information about the target variable).
"""

print(X.columns)

print(X.shape)

print(y)

"""# Model selection and parameter tuning"""

# LogisticRegression




# Split into train, validation and test sets
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=44)

# Split train+val into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, stratify=y_train_val, random_state=44)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression with Grid Search
lr_params = {'C': [0.001, 0.01, 0.1, 1, 2, 5, 10, 100], 'max_iter': [1000, 2000]}
lr_grid = GridSearchCV(LogisticRegression(), lr_params, cv=5, n_jobs=-1, verbose=1)
lr_grid.fit(X_train_scaled, y_train)

print("Best Logistic Regression parameters:", lr_grid.best_params_)

# Evaluation on validation set
val_pred = lr_grid.predict(X_val_scaled)
print("\nValidation Set Results:")
print(classification_report(y_val, val_pred))
print(f"Validation Accuracy: {accuracy_score(y_val, val_pred)}")

# Evaluation on test set
test_pred = lr_grid.predict(X_test_scaled)
print("\nTest Set Results:")
print(classification_report(y_test, test_pred))
print(f"Test Accuracy: {accuracy_score(y_test, test_pred)}")

# RandomForest



# # Random Forest with Grid Search
# rf_params = {
#     'n_estimators': [100, 150, 200, 300, 400],
#     'max_depth': [None, 10, 20, 25, 30],
#     'min_samples_split': [2, 5, 10, 15],
#     'min_samples_leaf': [1, 2, 4, 5]
# }

# rf_grid = GridSearchCV(RandomForestClassifier(random_state=44), rf_params, cv=5, n_jobs=-1, verbose=1)
# rf_grid.fit(X_train_scaled, y_train)

# print("Best Random Forest parameters:", rf_grid.best_params_)

# # Evaluation on validation set
# val_pred = rf_grid.predict(X_val_scaled)
# print("\nValidation Set Results:")
# print(classification_report(y_val, val_pred))
# print(f"Validation Accuracy: {accuracy_score(y_val, val_pred)}")

# # Evaluation on test set
# test_pred = rf_grid.predict(X_test_scaled)
# print("\nTest Set Results:")
# print(classification_report(y_test, test_pred))
# print(f"Test Accuracy: {accuracy_score(y_test, test_pred)}")

# # Feature Importance
# feature_importance = pd.DataFrame({
#     'feature': X.columns,
#     'importance': rf_grid.best_estimator_.feature_importances_
# }).sort_values('importance', ascending=False)

# print("\nFeature Importance:")
# print(feature_importance)

# # CNN




# # Reshaping data for CNN (assuming 1D convolution)
# X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
# X_val_reshaped = X_val_scaled.reshape(X_val_scaled.shape[0], X_val_scaled.shape[1], 1)
# X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

# # Defining the CNN model
# def create_cnn_model(input_shape, num_classes):
#     model = Sequential([
#         Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
#         MaxPooling1D(pool_size=2),
#         Conv1D(filters=128, kernel_size=3, activation='relu'),
#         MaxPooling1D(pool_size=2),
#         Flatten(),
#         Dense(128, activation='relu'),
#         Dropout(0.3),
#         Dense(num_classes, activation='softmax')
#     ])
#     return model

# # Create and compile the model
# num_classes = len(np.unique(y_train))
# model = create_cnn_model((X_train_scaled.shape[1], 1), num_classes)
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# # Model training
# history = model.fit(X_train_reshaped, y_train, epochs=50, batch_size=32,
#                     validation_data=(X_val_reshaped, y_val), verbose=1)

# # Evaluation on validation set
# val_pred = model.predict(X_val_reshaped)
# val_pred_classes = np.argmax(val_pred, axis=1)
# print("\nValidation Set Results:")
# print(classification_report(y_val, val_pred_classes))
# print(f"Validation Accuracy: {accuracy_score(y_val, val_pred_classes)}")

# # Evaluation on test set
# test_pred = model.predict(X_test_reshaped)
# test_pred_classes = np.argmax(test_pred, axis=1)
# print("\nTest Set Results:")
# print(classification_report(y_test, test_pred_classes))
# print(f"Test Accuracy: {accuracy_score(y_test, test_pred_classes)}")

# # Training history
# plt.figure(figsize=(12, 4))
# plt.subplot(1, 2, 1)
# plt.plot(history.history['accuracy'], label='Training Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.title('Model Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()

# plt.subplot(1, 2, 2)
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.title('Model Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()

# plt.tight_layout()
# plt.show()

"""**To summarzie:**

* The Random Forest model emerges as the best-performing model with perfect accuracy (1.0) on both validation and test sets, showing no signs of overfitting.

* Logistic Regression achieves 99.95% validation accuracy and 99.87% test accuracy, with minimal overfitting, while the CNN shows slight overfitting, with validation accuracy (99.10%) higher than test accuracy (98.87%).

* Random Forest's feature importance analysis highlights AQI Value, PM2.5 AQI Value, and PM2.5 AQI Category as the most influential features.

# Save the model
"""

# Create and train the Random Forest model with best parameters.
# Best Random Forest parameters: {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}





rf_model = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_leaf=1, min_samples_split=2, random_state=42)
rf_model.fit(X_train, y_train)

# Save the model 

filename = 'random_forest_model.pkl'
pickle.dump(rf_model, open(filename, 'wb'))
print(f"Model saved as {filename}")