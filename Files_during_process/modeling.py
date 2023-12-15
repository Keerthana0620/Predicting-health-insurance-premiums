
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
