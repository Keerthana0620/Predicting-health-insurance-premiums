
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
#import rfit 

#%%

df_input = pd.read_csv("Documents/insurance_dataset.csv")
df_input.head()

#%%

# Summary of world1 and world2 dataset
df_input.head()
df_input.describe()
df_input.shape
df_input.info()

# Data Cleaning
df_input.isna().values.any()

df_input.drop_duplicates(inplace=True)

#%%

print(df_input.dtypes)
print(df_input.isnull().sum())
#%%

# null is present in medical_history and family_medical_history.
df = df_input
df['medical_history'].fillna('None', inplace=True)
df['family_medical_history'].fillna('None', inplace=True)
df.isnull().sum()

#%%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

scaler = StandardScaler()
# df[['age', 'bmi', 'children']] = scaler.fit_transform(df[['age', 'bmi', 'children']])
df = pd.get_dummies(df,columns=['gender', 'smoker', 'region', 'medical_history', 'family_medical_history', 'exercise_frequency', 'occupation', 'coverage_level'])
boolean_columns = df.select_dtypes(include='bool').columns

for i in boolean_columns:
    df[i] = df[i].astype(int)
X = df.drop(['charges'], axis=1)
y = df['charges']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# %%

model = RandomForestRegressor(n_estimators=20, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# %%
plt.figure(figsize= (10, 10))
plt.scatter (y_test, y_pred, color = 'red', label='Comparison of Prediction between Actual & Prediction data')
plt.legend()
plt.grid()
plt.title('Random Forest Regression')
plt.xlabel("Prediction data")
plt.ylabel('Actual data')
plt.show()

# a = a.apply(lambda x: x.astype(int) if x.dtype == bool else x)
# a

# %%
from sklearn.svm import SVR

svr = SVR(kernel='linear',C=3)

svr.fit(X_train[:200000], y_train[:200000])
y_pred = svr.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')
# %%
errors = abs(y_pred - y_test)

print('Metrics for Gradient boosting model after tuning')
print("Mean squared error: ", mean_squared_error(y_test, y_pred))
print('Average absolute error:', round(np.mean(errors), 2))
# Calculate mean absolute percentage error (MAPE)
mape = 100 * np.mean((errors / y_test))
print('mean absolute percentage error (MAPE):', mape)
# Calculate and display accuracy
accuracy = 100 - mape
print('Accuracy:', round(accuracy, 2), '%.')
# %%
df
# %%
