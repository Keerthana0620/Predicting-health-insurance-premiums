
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

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
# %%
def pedict_data(model, X_test):
    return model.predict(X_test)
# %%
random_forest_r2 = []
random_forest_mae = []
random_forest_mse = []
random_forest_mape = []
random_forest_acc = []
def random_forest_model(n_estimators, test_size=0.2):
    global rf_model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    rf_model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    rf_model.fit(X_train, y_train)

    y_pred = pedict_data(rf_model, X_test)

    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    errors = abs(y_pred - y_test)
    mape = 100 * np.mean((errors / y_test))
    accuracy = round((100 - mape),2)
    random_forest_r2.append(r2)
    random_forest_mae.append(mae)
    random_forest_mse.append(mse)
    random_forest_mape.append(mape)
    random_forest_acc.append(accuracy)
    print(f'\nIstimators size: {n_estimators}')
    print(f'Mean Absolute Error: {mae}')
    print(f'Mean Squared Error: {mse}')
    print(f'R-squared: {r2}')
    print(f'rf_mape: {mape}')
    print(f'rf_acc: {accuracy}')

istimators = [20]

for i in istimators:
    random_forest_model(i)

# %%
from sklearn import tree
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=800)
tree.plot_tree(rf_model.estimators_[0],
               feature_names=['smoker','gender'],
               filled = True)

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_pred = pedict_data(rf_model, X_train)
r2 = r2_score(y_train, y_pred)
r2
# %%
# fig = plt.figure()
# ax1 = fig.add_subplot(221)
# ax2 = fig.add_subplot(222)
# ax3 = fig.add_subplot(223)
# ax4 = fig.add_subplot(224)
# ax1.title.set_text('R2')
# ax2.title.set_text('MAE')
# ax3.title.set_text('MSE')
# ax4.title.set_text('Accuracy')
# plt.tight_layout()
# plt.show()
figure, axis = plt.subplots(2, 2) 
axis[0,0].plot(istimators, random_forest_r2)
axis[0,0].set_title('R2')

axis[0,1].plot(istimators, random_forest_mae)
axis[0,1].set_title('MAE')

axis[1,0].plot(istimators, random_forest_mse)
axis[1,0].set_title('MSE')

axis[1,1].plot(istimators, random_forest_acc)
axis[1,1].set_title('Accuracy')
plt.tight_layout()

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.001, random_state=42)
y_pred = pedict_data(rf_model, X_test)
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
final_dict = {}

def svr_model(C_value, kernel='linear'):
    global svr_model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    svr_model = SVR(kernel=kernel,C=C_value)

    svr_model.fit(X_train[:100000], y_train[:100000])
    y_pred = pedict_data(svr_model, X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    errors = abs(y_pred - y_test)
    mape = 100 * np.mean((errors / y_test))
    accuracy = round((100 - mape),2)
    final_dict[kernel]['svr_r2'].append(r2)
    final_dict[kernel]['svr_mae'].append(mae)
    final_dict[kernel]['svr_mse'].append(mse)
    final_dict[kernel]['svr_mape'].append(mape)
    final_dict[kernel]['svr_acc'].append(accuracy)
    print(f'\nC value: {C_value}')
    print(f'Mean Absolute Error: {mae}')
    print(f'Mean Squared Error: {mse}')
    print(f'R-squared: {r2}')
    print(f'svr_mape: {mape}')
    print(f'svr_acc: {accuracy}')

kernal_values = ['linear']
c_values = [3]
for j in kernal_values:
    final_dict[j] = {'svr_r2':[], 'svr_mae':[], 'svr_mse':[], 'svr_mape':[], 'svr_acc':[]}
    for i in c_values:
        svr_model(i, j)
# %%
X_train, X_test, y_train, y_test = train_test_split(X[:100000], y[:100000], test_size=0.2, random_state=42)
# y_pred = pedict_data(svr_model, X_train)
# r2 = r2_score(y_train, y_pred)
# r2
# %%
y_pred = pedict_data(svr_model, X_train)
r2 = r2_score(y_train, y_pred)
r2

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
final_dict
# %%
final_dict = {"linear": {"svr_r2": [0.8354721613586709,
   0.9879258124387017,
   0.9931792867476782,
   0.9942962864140545,
   0.9948018688313414],
  "svr_mae": [1442.2623408083427,
   391.2504441642245,
   299.5320934368227,
   278.00288381488053,
   268.1446534442885],
  "svr_mse": [3201830.5257494566,
   234972.40726335152,
   132736.00430791758,
   110998.38434894962,
   101159.3854882148],
  "svr_mape": [9.672150351184063,
   2.59304613731504,
   1.9727324199822969,
   1.817776992393024,
   1.7483061931064119],
  "svr_acc": [90.33, 97.41, 98.03, 98.18, 98.25]},
 "poly": {"svr_r2": [0.007932739226342078,
   0.011385999582347983,
   0.014207020936198522,
   0.016520832621645387,
   0.01859741973292328],
  "svr_mae": [3565.221190807272,
   3559.0334276721014,
   3554.1359239114745,
   3549.998184474194,
   3546.2166331439444],
  "svr_mse": [19306345.14725723,
   19239142.20754486,
   19184243.09527827,
   19139214.649353992,
   19098802.76491441],
  "svr_mape": [24.260121467323057,
   24.2016630657169,
   24.17085798832001,
   24.140449274704174,
   24.109062836615713],
  "svr_acc": [75.74, 75.8, 75.83, 75.86, 75.89]},
 "rbf": {"svr_r2": [0.0022650716042862395,
   0.004570346988091556,
   0.0065070164807902175,
   0.008165466462295323,
   0.009813568035992315],
  "svr_mae": [3574.817399445121,
   3570.918563971914,
   3567.505071696521,
   3564.6082563979708,
   3561.7159986844126],
  "svr_mse": [19416642.051124427,
   19371779.72779313,
   19334090.741228808,
   19301816.1072212,
   19269742.860700883],
  "svr_mape": [24.32929229148853,
   24.306716057151124,
   24.27765841197998,
   24.25655110809166,
   24.23475498495296],
  "svr_acc": [75.67, 75.69, 75.72, 75.74, 75.77]},
 "sigmoid": {"svr_r2": [-0.0004442770395358675,
   -0.0004491362271024091,
   -0.0004486417703595258,
   -0.0004648603904751081,
   -0.0005032492503000974],
  "svr_mae": [3579.514094748113,
   3579.473260845434,
   3579.430778719957,
   3579.4306070033213,
   3579.4193985304482],
  "svr_mse": [19469367.931828417,
   19469462.495126694,
   19469452.872641493,
   19469768.498698458,
   19470515.573626637],
  "svr_mape": [24.370911025040787,
   24.369049623315643,
   24.36777388055875,
   24.367856041094868,
   24.364102366446176],
  "svr_acc": [75.63, 75.63, 75.63, 75.63, 75.64]}}
c_values = [1,2,3,4,5]

figure, axis = plt.subplots(2, 2) 
axis[0,0].plot(c_values, final_dict['linear']['svr_r2'])
axis[0,0].plot(c_values, final_dict['poly']['svr_r2'])
axis[0,0].plot(c_values, final_dict['rbf']['svr_r2'])
axis[0,0].plot(c_values, final_dict['sigmoid']['svr_r2'])
axis[0,0].set_title('R2')

axis[0,1].plot(c_values, final_dict['linear']['svr_mae'])
axis[0,1].plot(c_values, final_dict['poly']['svr_mae'])
axis[0,1].plot(c_values, final_dict['rbf']['svr_mae'])
axis[0,1].plot(c_values, final_dict['sigmoid']['svr_mae'])
axis[0,1].set_title('MAE')

axis[1,0].plot(c_values, final_dict['linear']['svr_mse'])
axis[1,0].plot(c_values, final_dict['poly']['svr_mse'])
axis[1,0].plot(c_values, final_dict['rbf']['svr_mse'])
axis[1,0].plot(c_values, final_dict['sigmoid']['svr_mse'])
axis[1,0].set_title('MSE')

axis[1,1].plot(c_values, final_dict['linear']['svr_acc'])
axis[1,1].plot(c_values, final_dict['poly']['svr_acc'])
axis[1,1].plot(c_values, final_dict['rbf']['svr_acc'])
axis[1,1].plot(c_values, final_dict['sigmoid']['svr_acc'])
axis[1,1].set_title('Accuracy')
plt.tight_layout()
plt.legend(['linear', 'poly', 'rbf', 'sigmoid'],bbox_to_anchor=(1, 2.4), loc='upper left')
# %%
X