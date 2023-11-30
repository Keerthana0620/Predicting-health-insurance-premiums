#%%#%%
import csv
import seaborn as sns
import pandas as pd
import numpy as np

#%%
df=pd.read_csv("insurance_dataset.csv")
df.info()
# %%
#Data_Preprocessing

#Checking for duplicates

num_duplicates = df.duplicated().sum()
print(f"Number of duplicate rows: {num_duplicates}")

df = df.drop_duplicates()

# Checking for NA's
missing_values = df.isnull().sum()

# Handling missing values with mean
if missing_values.any():
    df.fillna(df.mean(), inplace=True)
    print("Missing values filled with mean.")
else:
    print("No NA values found.")

# %%
# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns


age_bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]

plt.figure(figsize=(10, 6))
sns.histplot(df['age'], bins=age_bins, kde=False, color='skyblue')
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Count')
plt.xticks(range(0, 101, 5))

plt.show()

# %%
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 6))
ax = sns.countplot(x='gender', data=df, palette='pastel', order=['male', 'female'])

for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 6), textcoords='offset points', fontsize=12, color='black')

plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')

plt.show()



# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Create a count plot for medical history distribution
plt.figure(figsize=(10, 6))
ax = sns.countplot(x='medical_history', data=df, palette='pastel', order=['Diabetes', 'Heart disease', 'High blood pressure','None'])
plt.title('Medical History Distribution')
plt.xlabel('Medical History')
plt.ylabel('Count')

# Annotate the count on each bar
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 6), textcoords='offset points', fontsize=12, color='black')

plt.show()

#%%%
import pandas as pd


smoker_medical_history_percentages = pd.crosstab(df['smoker'], df['medical_history'], normalize='index') * 100

res = smoker_medical_history_percentages.style.format('{:.2f}%')

res

# %%
import matplotlib.pyplot as plt
import seaborn as sns


bmi_bins = [0, 18.5, 24.9, 29.9, float('inf')]
bmi_labels = ['Underweight', 'Normal weight', 'Overweight', 'Obesity']

df['bmi_category'] = pd.cut(df['bmi'], bins=bmi_bins, labels=bmi_labels, right=False)

plt.figure(figsize=(10, 6))
sns.countplot(x='bmi_category', data=df, palette='deep', order=bmi_labels)
plt.title('BMI Distribution')
plt.xlabel('BMI Category')
plt.ylabel('Count')

plt.show()

df.drop('bmi_category', axis=1, inplace=True)

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Define BMI categories
bmi_bins = [0, 18.5, 24.9, 29.9, float('inf')]
bmi_labels = ['Underweight', 'Normal weight', 'Overweight', 'Obesity']

df['bmi_category'] = pd.cut(df['bmi'], bins=bmi_bins, labels=bmi_labels, right=False)

plt.figure(figsize=(12, 6))
sns.countplot(x='bmi_category', hue='occupation', data=df, palette='pastel', order=bmi_labels)
plt.title('BMI Distribution by Occupation')
plt.xlabel('BMI Category')
plt.ylabel('Count')
plt.show()


df.drop('bmi_category', axis=1, inplace=True)

bmi_category = pd.cut(df['bmi'], bins=bmi_bins, labels=bmi_labels, right=False)

bmi_occupation_counts = pd.crosstab(df['occupation'], bmi_category)

bmi_occupation_percentages = bmi_occupation_counts.div(bmi_occupation_counts.sum(axis=1), axis=0) * 100

print("BMI Distribution by Occupation:")
print(bmi_occupation_percentages.applymap(lambda x: f'{x:.2f}%'))

#%%
import pandas as pd
import statsmodels.api as sm


bmi_bins = [0, 18.5, 24.9, 29.9, float('inf')]
bmi_labels = ['Underweight', 'Normal weight', 'Overweight', 'Obesity']

df['bmi_category'] = pd.cut(df['bmi'], bins=bmi_bins, labels=bmi_labels, right=False)
gender_dummy = pd.get_dummies(df['gender'], drop_first=True)
smoker_dummy = pd.get_dummies(df['smoker'], prefix='smoke', drop_first=True)
medical_history_dummies = pd.get_dummies(df['medical_history'], drop_first=True)
family_medical_history_dummies = pd.get_dummies(df['family_medical_history'], prefix='family_medical_history', drop_first=True)
exercise_frequency_dummies = pd.get_dummies(df['exercise_frequency'], drop_first=True)
occupation_dummies = pd.get_dummies(df['occupation'], drop_first=True)
coverage_level_dummies = pd.get_dummies(df['coverage_level'], drop_first=True)


formula = 'charges ~ age + bmi_category + smoker +medical_history+family_medical_history+exercise_frequency+occupation+coverage_level'

model = sm.OLS.from_formula(formula, data=df).fit()

print(model.summary())


# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

X = df[['age','bmi', 'smoker', 'medical_history', 'family_medical_history', 'occupation', 'coverage_level']]
y = df['charges']

X_encoded = pd.get_dummies(X, columns=['bmi','smoker', 'medical_history', 'family_medical_history', 'occupation', 'coverage_level'], drop_first=True)

xtrain, xtest, ytrain, ytest = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

lrmodel = LinearRegression()
lrmodel.fit(xtrain, ytrain)

y_pred = lrmodel.predict(xtest)

r2 = r2_score(ytest, y_pred)

errors = abs(y_pred - ytest)

mape = 100 * np.mean((errors / ytest))

accuracy = 100 - mape


print(f'R-squared on the training: {lrmodel.score(xtrain, ytrain)}')
print(f'R-squared on the test: {lrmodel.score(xtest, ytest)}')
print("Mean squared error: ", mean_squared_error(ytest, y_pred))
print('Average absolute error:', round(np.mean(errors), 2))
print('mean absolute percentage error (MAPE):', mape)
print('Accuracy:', round(accuracy, 2), '%.')
print(f'Cross-validated R-squared: {cross_val_score(lrmodel, X_encoded, y, cv=5).mean()}')


# %%
