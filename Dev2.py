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

#%%

import seaborn as sns
import matplotlib.pyplot as plt

# Box plot

sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.barplot(x='region', y='charges', data=df)
plt.title('Insurance Charges Across Regions')
plt.xlabel('Region')
plt.ylabel('Charges')
plt.show()

#%%
from scipy.stats import f_oneway

charges_by_region = [df['charges'][df['region'] == region] for region in df['region'].unique()]

f_statistic, p_value = f_oneway(*charges_by_region)

print(f"F-statistic: {f_statistic}")
print(f"P-value: {p_value}")

'''
Null Hypothesis : There is no significant difference in the mean insurance charges across the four regions.

Alternative Hypothesis : There is a significant difference in the mean insurance charges across the four regions.

F-statistic: 1644.2059645503782
P-value: 0.0

The ANOVA test revealed significantly different mean insurance charges 
among at least two of the four regions (p < 0.001). 
Subsequent Tukey HSD analysis indicated specific pairwise 
differences in mean charges, offering insights into the 
direction and significance of regional variations.
'''

from statsmodels.stats.multicomp import pairwise_tukeyhsd

tukey_result = pairwise_tukeyhsd(df['charges'], df['region'])
print(tukey_result)
#%%

sns.set(style="whitegrid")
sns.barplot(x='smoker', y='charges', data=df)
plt.title('Average Insurance Charges for Smokers and Non-Smokers')
plt.xlabel('Smoker')
plt.ylabel('Charges')
plt.show()


#%%

from scipy.stats import ttest_ind

smoker_charges = df[df['smoker'] == 'yes']['charges']
non_smoker_charges = df[df['smoker'] == 'no']['charges']

t_stat, p_value = ttest_ind(smoker_charges, non_smoker_charges)
print(f'T-statistic: {t_stat}, p-value: {p_value}')

'''
T-statistic: 686.9351780779127, p-value: 0.0
'''



#%%

from scipy.stats import pearsonr

# Calculate Pearson's correlation coefficient and p-value
correlation_coefficient, p_value = pearsonr(df['age'], df['charges'])

# Print the results
print(f"Pearson's correlation coefficient: {correlation_coefficient}")
print(f"P-value: {p_value}")

#%%

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Set the plot style to whitegrid
sns.set(style="whitegrid")

# Define a function to remove outliers based on IQR
def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

# Remove outliers for each gender
df_no_outliers = pd.concat([remove_outliers_iqr(df[df['gender'] == 'male'], 'charges'),
                            remove_outliers_iqr(df[df['gender'] == 'female'], 'charges')])

# Create the box plot without outliers
sns.boxplot(x='gender', y='charges', data=df_no_outliers)
plt.title('Insurance Premiums by Gender')
plt.xlabel('Gender')
plt.ylabel('Charges')
plt.show()


male_charges = df[df['gender'] == 'male']['charges']
female_charges = df[df['gender'] == 'female']['charges']

t_stat, p_value = ttest_ind(male_charges, female_charges)
print(f'Test Statistic: {t_stat}, p-value: {p_value}')
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
from scipy.stats import chi2_contingency

contingency_table = pd.crosstab(df['gender'], df['medical_history'])

chi2, p, dof, expected = chi2_contingency(contingency_table)

print("Chi-square test for independence between gender and medical history:")
print(f"Chi2 value: {chi2}")
print(f"P-value: {p}")

'''
Chi-square test for independence between gender and medical history:
Chi2 value: 3.426201023953485
P-value: 0.33046048711039894
There's no Significant assosciation between Gender and Medical history of an individual
'''

#%%
from scipy.stats import f_oneway


grouped_data = [df[df['occupation'] == occupation]['charges'] for occupation in df['occupation'].unique()]


f_statistic, p_value = f_oneway(*grouped_data)

print(f"F-statistic: {f_statistic}")
print(f"P-value: {p_value}")

'''
F-statistic: 10960.740716743454
P-value: 0.0
There is a significant difference in charges among different occupations.
'''


#%%
import seaborn as sns
import matplotlib.pyplot as plt

# Scatter plot of children vs charges
sns.scatterplot(x='children', y='charges', data=df)
plt.title('Insurance Charges by Number of Children')
plt.xlabel('Number of Children')
plt.ylabel('Charges')
plt.show()

#%%
import pandas as pd
from scipy.stats import f_oneway

exercise_levels = df['exercise_frequency'].unique()

charges_by_group = {level: df[df['exercise_frequency'] == level]['charges'] for level in exercise_levels}

f_statistic, p_value = f_oneway(*charges_by_group.values())

print(f'F-statistic: {f_statistic}')
print(f'P-value: {p_value}')

'''
F-statistic: 9594.186867640998
P-value: 0.0
'''

#%%
import seaborn as sns
import matplotlib.pyplot as plt

# Interaction plot
sns.catplot(x='smoker', y='charges', hue='exercise_frequency', kind='point', data=df)
plt.title('Effect of Smoking and Exercise Frequency on charges')
plt.xlabel('Smoker')
plt.ylabel('Charges')
plt.show()


#%%
import statsmodels.stats.multicomp as mc

# Create a dataframe for the Tukey HSD test
tukey_data = df[['smoker', 'exercise_frequency', 'charges']]

# Perform Tukey HSD test
tukey_result = mc.MultiComparison(tukey_data['charges'], tukey_data[['smoker', 'exercise_frequency']])
tukey_table = tukey_result.tukeyhsd()

# Print the Tukey HSD results
print(tukey_table)



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
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error                   

X = df[['age','gender' , 'bmi','smoker', 'medical_history', 'family_medical_history', 'occupation','coverage_level']]
y = df['charges']

X_encoded = pd.get_dummies(X, columns=['gender','smoker', 'medical_history', 'family_medical_history', 'occupation','coverage_level'], drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

lrmodel = LinearRegression()
lrmodel.fit(X_train, y_train)


y_pred_train = lrmodel.predict(X_train)
y_pred_test = lrmodel.predict(X_test)


r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)
errors = abs(y_pred_test - y_test)
mape = 100 * np.mean((errors / y_test))
accuracy = 100 - mape


print("Mean squared error: ", mean_squared_error(y_test, y_pred_test))
print('Average absolute error:', round(np.mean(errors), 2))
print('Mean absolute percentage error (MAPE):', mape)
print('Accuracy:', round(accuracy, 2), '%.')

print(f'R-squared on the training set: {r2_train}')
print(f'R-squared on the test set: {r2_test}')


# %%
import matplotlib.pyplot as plt
import seaborn as sns

results_train = pd.DataFrame({'Actual': y_train, 'Predicted': y_pred_train})
results_test = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_test})

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.scatterplot(data=results_train.head(100), x='Actual', y='Predicted')
plt.plot(results_train.head(100)['Actual'], results_train.head(100)['Actual'], color='red', linestyle='--')
plt.title('Training Set: Actual vs Predicted Charges')
plt.xlabel('Actual Charges')
plt.ylabel('Predicted Charges')

plt.subplot(1, 2, 2)
sns.scatterplot(data=results_test.head(100), x='Actual', y='Predicted')
plt.plot(results_test.head(100)['Actual'], results_test.head(100)['Actual'], color='red', linestyle='--')
plt.title('Testing Set: Actual vs Predicted Charges')
plt.xlabel('Actual Charges')
plt.ylabel('Predicted Charges')

plt.tight_layout()
plt.show()
# %%
from scipy.stats import f_oneway

# Example assuming 'medical_history' is the categorical variable and 'charges' is the continuous variable
medical_history_levels = df['medical_history'].unique()

# Create a list of charges for each level of medical history
charges_by_history = [df[df['medical_history'] == history]['charges'] for history in medical_history_levels]

# Perform ANOVA
f_statistic, p_value = f_oneway(*charges_by_history)

# Print the results
print(f'F-statistic: {f_statistic}')
print(f'P-value: {p_value}')

# %%
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm

model_FMH= ols('charges ~ C(family_medical_history)',data=df).fit()
result_FMH = sm.stats.anova_lm(model_FMH, type=1)
  
# Print the result
print(result_FMH, "\n")

print(" There is significant difference among the means of family_medical_history groups of charges")

tukey_FMH = pairwise_tukeyhsd(endog=df['charges'], groups=df['family_medical_history'], alpha=0.05)
print(tukey_FMH)

# %%
