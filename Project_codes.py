# To add a new cell, type ''
# To add a new markdown cell, type '#%%[markdown]'

#%%[markdown]
#
# # Predicting Health Insurance Premiums in the US : Data Mining Project
# ## By: Keerthana Aravindhan
# ### Date: Nov 07 2023
#
# 
# We have the Insurance dataset here.  
# The variables in the dataset are:  
# * `age`: Age in years
# * `gender`: Male / female
# * `bmi`: bmi count
# * `children`: number of children
# * `smoker`: Yes/no
# * `region`: northeast, northwest, southeast, southwest
# * `medical_history`: Diabetes/High blood pressure/Heart disease/None
# * `family_medical_history`: Diabetes/High blood pressure/Heart disease/None
# * `exercie`: frequently, occasionaly, Rarely, Never
# * `occupation`: White collar, blue collar, student, unemployed
# * `coverage_level`: Basic, premium, standard
# * `charges`: premium insurance charges amount. (target)


#%%

#############
## Imports ##
#############

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn import preprocessing
# from IPython.display import display
#import rfit 

from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor
import scipy.stats as stats
import statsmodels.api as sm
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from sklearn.ensemble import GradientBoostingRegressor
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor 
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

##################################################
#<<<<<<<<<<<<<<<< End of Section >>>>>>>>>>>>>>>>#

#%%

##########################
## Set Display Settings ##
##########################

#Possible Graph Color Schemes
#color_list=['BuPu', 'PRGn', 'Pastel1', 'Pastel2', 'Set2', 'binary', 'bone', 'bwr',
#                 'bwr_r', 'coolwarm', 'cubehelix', 'gist_earth', 'gist_gray', 'icefire',]

#Diagram Settings
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.colheader_justify', 'center')
pd.set_option('display.precision', 3)
#pd.set_option('display.float_format', lambda x: '%.2f' % x)

#Select Color Palette
sns.set_palette('Set2')

#RGB values of pallette
#print(sns.color_palette('Set2').as_hex())
#col_pallette=['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494', '#b3b3b3']

##################################################
#<<<<<<<<<<<<<<<< End of Section >>>>>>>>>>>>>>>>#

#%%

###############
## Functions ##
###############

# encoding categorical variables.
def Encoder(x):
    """
    Encoding Categorical variables in the dataset.
    """
    columnsToEncode = list(x.select_dtypes(include=['object']))
    le = preprocessing.LabelEncoder()
    for feature in columnsToEncode:
        try:
           x[feature] = le.fit_transform(x[feature])
        except:
            print('Error encoding '+feature)
    return df

##################################################
#<<<<<<<<<<<<<<<< End of Section >>>>>>>>>>>>>>>>#


#%%

###############
## Load Data ##
###############

df_input = pd.read_csv("insurance_dataset.csv")
df_input.head()
#display(df_input.head().style.set_sticky(axis="index"))

#%%

# Details about dataset:
print(f"Number of Observations in dataset: {len(df_input)} \n")
print(f"Shape of dataset: {df_input.shape} \n")
print(f"Dataset info: {df_input.info()} \n")
print(f"Dataset describe: \n {df_input.describe()}")

##################################################
#<<<<<<<<<<<<<<<< End of Section >>>>>>>>>>>>>>>>#

#%%

################
## Clean Data ##
################

# Remove missing values
print(df_input.isnull().sum())
df_input.drop_duplicates(inplace=True)

# null is present in medical_history and family_medical_history.
df = df_input.copy()
df['medical_history'].fillna('None', inplace=True)
df['family_medical_history'].fillna('None', inplace=True)

# After removing
print("\nAfter Removing missing values:\n")
print(df.isnull().sum())

#%%

#Convert columns to desired data types
print(df.dtypes)

# Converting categorical variables into int

df = Encoder(df)
print("\n")
print (df.head())

""" 
male : 1, female : 0 
smoker : 1, non smoker : 0
medical_history: Heart disease- 1,High blood pressure- 2,Diabetes- 0, None- 3
region: southeast-2 , northwest-1 ,southwest- 3, northeast- 0
family_medical_history: Heart disease- 1,High blood pressure- 2,Diabetes- 0, None- 3
exercise_frequency: Frequently-0 ,never-1 ,occasionaly-2 ,rarely-3 
occupation : blue collar- 0, student- 1,unemployed- 2,whitecollar- 3
coverage_level : Basic- 0,Premium- 1,Standard- 2  
"""

#%%
print("Data Types after conversion \n")
print(df.dtypes)

##################################################
#<<<<<<<<<<<<<<<< End of Section >>>>>>>>>>>>>>>>#

#%%

###############################
## Exploratory data analysis ##
###############################

print('\nGraphical Exploration of numerical variables:\n')

# Charge Distribution :

print('\nCharges Distribution')
sns.histplot(data=df, x='charges', bins=30, color='m')
plt.xlabel ('Charges ', size=12)
plt.ylabel('Count', size=12)
plt.title('Distribution of Charges', size=14)
plt.show()

# The dataset exhibits a notable absence of outliers, and 
# the distribution of charges aligns closely with a normal distribution.

#%%

# Distribution of age:

print('\nAge Distribution')
sns.kdeplot(df['age'], color = "m", alpha = 0.5)
plt.xlabel ('Age ', size=12)
plt.ylabel('Density', size=12)
plt.title('Distribution of Age', size=14)
plt.show()

# all age groups are present in almost equal density.

#%%

# Scatter plot of age vs charges:

age = pd.pivot_table(df, values='charges', columns='age', aggfunc='mean')
melt_age = pd.melt(age, var_name='age', value_name='average_charges')

sns.scatterplot(x='age', y='average_charges', data=melt_age, alpha=0.5, color='blue')

plt.title('Scatter Plot between Charges and age')
plt.xlabel('age')
plt.ylabel('Average Charges')
plt.show()

# age and charges are almost linear to each other.
# Thus, when age increase charges increase.

#%%

# Distribution of bmi:

print('\nBMI Distribution')
sns.kdeplot(df['bmi'], color = "m", alpha = 0.5)
plt.xlabel ('bmi ', size=12)
plt.ylabel('density', size=12)
plt.title('Distribution of BMI', size=14)
plt.show()

# all bmi values are also present in almost equal level.

#%%

# Scatter plot of BMI vs charges:

bmi = pd.pivot_table(df, values='charges', columns='bmi', aggfunc='mean')
melt_bmi = pd.melt(bmi, var_name='bmi', value_name='average_charges')

sns.scatterplot(x='bmi', y='average_charges', data=melt_bmi, alpha=0.5, color='orange')

plt.title('Scatter Plot between Charges and bmi')
plt.xlabel('bmi')
plt.ylabel('Average Charges')
plt.show()

# bmi and charges are almost linear to each other.

#%%

print('\nGraphical Exploration of categorical variables:\n')

# Coverage level vs Charges:

df['coverage_level'].value_counts()
# Coverage level count is more in Basic and least in Premium

plt.figure(figsize=(10, 6))

# Create a box plot between charges and coverage_level.
df.boxplot(column='charges', by='coverage_level', vert=False, flierprops={'marker': 'o', 'markerfacecolor': 'red', 'markersize': 5})
plt.suptitle('')  # Remove the default title
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.title('Box Plot of Charges by coverage_level')
plt.xlabel('Charges')
plt.ylabel('coverage_level')
plt.tight_layout()
plt.show()

# The analysis exactly reveals that Premium customers incur the highest charges, 
# trailed by Standard and Basic customers in descending order.


#%%

# KDE plot of charges vs smoker

print("Smoker counts: ", df['smoker'].value_counts())

sns.set(style="whitegrid")
# Create a KDE plot
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df, x='charges', hue='smoker', fill=True, common_norm=False, palette='viridis')
# Set plot labels and title
plt.xlabel('Charges')
plt.ylabel('Density')
plt.title('KDE Plot of Charges vs. Smoker')
# Show the plot
plt.show()

# The KDE plot clearly illustrates that more of Smokers tend to incur higher charges 
# compared to non-smokers.


#%%

# Gender vs Charges:

print("Gender counts: ", df['gender'].value_counts())
# male      500107
# female    499893

plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='gender', y='charges', palette='pastel')
# Set plot labels and title
plt.xlabel('Gender')
plt.ylabel('Charges')
plt.title('Barplot of Charges vs. Gender')
# Show the plot
plt.show()

# On average, males tend to give higher charges than females. 
# Additionally, the maximum charge in the dataset is paid by a male, while the 
# minimum charges are associated with females.


#%%

# Charges vs Medical History:

medical = pd.pivot_table(df, values='charges', columns=['medical_history'], aggfunc='mean')
melt_medical = pd.melt(medical, var_name='medical_history', value_name='average_charges')

# Create a bar chart with 'hue' for gender
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")
sns.barplot(data=melt_medical, x='medical_history', y='average_charges', color='g')
plt.xlabel('medical_history')
plt.ylabel('Average Charges')
plt.title('Average Charges by Medical_history')
plt.xticks([0,1,2,3], ['Diabets', 'Heart disease', 'High blood pressue', 'None'])
plt.tight_layout()
plt.show()

# The average charges are notably higher for individuals with heart disease, followed by those with diabetes in medical_history.

#%%

# Charges vs Family Medical History:

family_medical = pd.pivot_table(df, values='charges', columns=['family_medical_history'], aggfunc='mean')
# Melt the pivot table to a long format
melt_family_medical = pd.melt(family_medical, var_name='family_medical_history', value_name='average_charges')

# Create a bar chart with 'hue' for gender
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")
sns.barplot(data=melt_family_medical, x='family_medical_history', y='average_charges')
plt.xlabel('family_medical_history')
plt.ylabel('Average Charges')
plt.title('Average Charges by family_medical_history')
plt.xticks([0,1,2,3], ['Diabets', 'Heart disease', 'High blood pressue', 'None'])
plt.tight_layout()
plt.show()

# The average charges are notably higher for individuals with heart disease, followed by those with diabetes in family_medical_history.

#%%

# Charges vs Occupation:

print("Occupation counts: ", df['occupation'].value_counts())

occupation = pd.pivot_table(df, values='charges', columns=['occupation'], aggfunc='mean')
# Melt the pivot table to a long format
melt_occupation = pd.melt(occupation, var_name='occupation', value_name='average_charges')

# Create a bar chart with 'hue' for gender
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")
sns.barplot(data=melt_occupation, x='occupation', y='average_charges', color='pink')
plt.xlabel('occupation')
plt.ylabel('Average Charges')
plt.title('Average Charges by occupation')
plt.xticks([0,1,2,3], ['bluecollar', 'student', 'unemployed', 'whitecollar'])
plt.tight_layout()
plt.show()

#%%

# Occupation count:

occupation_counts = df['occupation'].value_counts().reset_index()
occupation_counts = occupation_counts.sort_values(by='occupation')

plt.figure(figsize=(12, 6))
plt.plot(occupation_counts['occupation'], occupation_counts['count'], marker='o', linestyle='-', color='b')
# Set plot labels and title
plt.xlabel('Occupation')
plt.ylabel('Count')
plt.title('Count of Occupations')
# Rotate x-axis labels for better visibility
plt.xticks(rotation=45, ha='right')
# Show the plot
plt.show()

# The average charges are notably higher for white collar occupation, followed by those
#  in blue collar and student.
# Unemployed pay less charges. Also, according to dataset, Unemployed are in more count.

#%%

# Exercise Freq count:

exercise_counts = df['exercise_frequency'].value_counts().reset_index()
exercise_counts = exercise_counts.sort_values(by='exercise_frequency')

plt.figure(figsize=(12, 6))
plt.plot(exercise_counts['exercise_frequency'], exercise_counts['count'], marker='o', linestyle='-', color='black')
# Set plot labels and title
plt.xlabel('exercise_frequency')
plt.ylabel('Count')
plt.title('Count of exercise_frequency')
# Rotate x-axis labels for better visibility
plt.xticks(rotation=45, ha='right')
# Show the plot
plt.show()

#%%

# Exercise_freq vs medical_history

plt.figure(figsize=(11,7))
sns.countplot(x='exercise_frequency',hue='medical_history',data=df,palette='Set3')

#%%

# Exercise_freq vs medical_history vs charges

sns.set(style="whitegrid")
# Create a grouped bar chart
plt.figure(figsize=(12, 8))
sns.barplot(data=df, x='exercise_frequency', y='charges', hue='medical_history', palette='muted')

# Set plot labels and title
plt.xlabel('Exercise Frequency')
plt.ylabel('Charges')
plt.title('Exercise Frequency vs. Medical History vs. Charges')
plt.xticks([0,1,2,3], ['Frequently', 'never', 'occasionaly', 'rarely'])
# Show the legend
plt.legend(title='Medical History')
# Show the plot
plt.show()

# Individuals who engage in frequent exercise and have a history of heart disease tend 
# to incur higher charges. Following this, those who exercise occasionally and have a history 
# of heart disease exhibit the highest charges.
 
# Conversely, the lowest charges are observed among individuals with no exercise regimen and no 
# reported medical history.


#%%

# Exercise_freq vs smoker vs charges

sns.set(style="whitegrid")
# Create a grouped bar chart
plt.figure(figsize=(12, 8))
sns.barplot(data=df, x='exercise_frequency', y='charges', hue='smoker', palette='PRGn')

# Set plot labels and title
plt.xlabel('Exercise Frequency')
plt.ylabel('Charges')
plt.title('Exercise Frequency vs. Smokers vs. Charges')
plt.xticks([0,1,2,3], ['Frequently', 'never', 'occasionaly', 'rarely'])
# Show the legend
plt.legend(title='Smokers')
# Show the plot
plt.show()

# Individuals who engage in frequent exercise and have a smoking habit tend 
# to incur higher charges. Following this, those who exercise occasionally and rarely and have smoking habit
#  exhibit the highest charges.
 
# Conversely, the lowest charges are observed among individuals with no exercise regimen and no 
# smoking habit.

#%%

##### Bar plot of categorical variables vs charges:

categorical_features = ['gender', 'smoker', 'region', 'children', 'medical_history', 'family_medical_history', 'exercise_frequency', 'occupation', 'coverage_level' ]
 
plt.subplots(figsize=(20, 10))
for i, col in enumerate(categorical_features):
    plt.subplot(3, 3, i + 1)
    df.groupby(col).mean()['charges'].plot.bar()
plt.show()

##################################################
#<<<<<<<<<<<<<<<< End of Section >>>>>>>>>>>>>>>>#

#%%

#######################
## Statistical Tests ##
#######################

#### T-Test between 2 sample independent and Quantitative dependent variable 

# Charges vs Gender:

df.boxplot(column='charges', by='gender', vert=False, flierprops={'marker': 'o', 'markerfacecolor': 'red', 'markersize': 5})
plt.suptitle('')  # Remove the default title
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.title('Box Plot of Charges by gender')
plt.xlabel('Charges')
plt.ylabel('gender')
plt.tight_layout()
plt.show()

charges = df['charges']
charges_male = charges[df['gender']==1]
charges_female = charges[df['gender']==0]

# T test between Income of male and female.
t_stat, p_value = stats.ttest_ind(charges_female, charges_male)

print(f"T test Result: \n P value: {p_value}")
alpha = 0.05
# Check the p-value
if p_value < alpha:  # You can choose your significance level (e.g., 0.05)
    print(" Reject the null hypothesis; There is a significant difference in charges between gender")
else:
    print(" There is no significant difference in charges between gender")


#%%

# Charges vs Smoker:

df.boxplot(column='charges', by='smoker', vert=False, flierprops={'marker': 'o', 'markerfacecolor': 'red', 'markersize': 5})
plt.suptitle('')  # Remove the default title
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.title('Box Plot of Charges by smoker')
plt.xlabel('Charges')
plt.ylabel('smoker')
plt.tight_layout()
plt.show()


charges_smoker = charges[df['smoker']==1]
charges_nonsmoker = charges[df['smoker']==0]

# T test between Income of smoker and non-smoker.
t_stat, p_value = stats.ttest_ind(charges_smoker, charges_nonsmoker)

print(f"T test Result: \n P value: {p_value}")
# Check the p-value
if p_value < alpha: 
    print(" Reject the null hypothesis; There is a significant difference in charges between smoker and nonsmoker")
else:
    print(" There is no significant difference in charges between smoker and non-smoker")

#%%

#### One way Anovas

#Boxplot charges vs medical_history

g1 = sns.boxplot(data = df, y = 'charges', x = 'medical_history', palette='Set3', showmeans = True, meanprops = {"marker": "o", "markerfacecolor":"red", "markeredgecolor":"black", "markersize":"6"}, showfliers = False)
g1.set(title ='Boxplot charges vs medical_history')
plt.xlabel('medical_history')
plt.ylabel('charges') 
plt.xticks([0,1,2,3], ['Diabets', 'Heart disease', 'High blood pressue', 'None'])
plt.grid()
plt.show()

#%%

model_MH= ols('charges ~ C(medical_history)',data=df).fit()
result_MH = sm.stats.anova_lm(model_MH, type=1)
  
# Print the result
print(result_MH, "\n")

tukey_MH = pairwise_tukeyhsd(endog=df['charges'], groups=df['medical_history'], alpha=0.05)
print(tukey_MH)

#%%

#Boxplot charges vs family_medical_history

g1 = sns.boxplot(data = df, y = 'charges', x = 'family_medical_history', palette='Set3', showmeans = True, meanprops = {"marker": "o", "markerfacecolor":"red", "markeredgecolor":"black", "markersize":"6"}, showfliers = False)
g1.set(title ='Boxplot charges vs family_medical_history')
plt.xlabel('family_medical_history')
plt.ylabel('charges') 
plt.xticks([0,1,2,3], ['Diabets', 'Heart disease', 'High blood pressue', 'None'])
plt.grid()
plt.show()

#%%

model_FMH= ols('charges ~ C(family_medical_history)',data=df).fit()
result_FMH = sm.stats.anova_lm(model_FMH, type=1)
  
# Print the result
print(result_FMH, "\n")

tukey_FMH = pairwise_tukeyhsd(endog=df['charges'], groups=df['family_medical_history'], alpha=0.05)
print(tukey_FMH)

#%%

#Boxplot charges vs occupation

g1 = sns.boxplot(data = df, y = 'charges', x = 'occupation', palette='Set3', showmeans = True, meanprops = {"marker": "o", "markerfacecolor":"red", "markeredgecolor":"black", "markersize":"6"}, showfliers = False)
g1.set(title ='Boxplot charges vs occupation')
plt.xlabel('occupation')
plt.ylabel('charges') 
plt.grid()
plt.show()

#%%

model_OC= ols('charges ~ C(occupation)',
            data=df).fit()
result_OC = sm.stats.anova_lm(model_OC, type=1)
  
# Print the result
print(result_OC, "\n")

tukey_OC = pairwise_tukeyhsd(endog=df['charges'], groups=df['occupation'], alpha=0.05)
print(tukey_OC)

#%%

#Boxplot charges vs exercise_frequency

g1 = sns.boxplot(data = df, y = 'charges', x = 'exercise_frequency', palette='Set3', showmeans = True, meanprops = {"marker": "o", "markerfacecolor":"red", "markeredgecolor":"black", "markersize":"6"}, showfliers = False)
g1.set(xticklabels= charges)
g1.set(title ='Boxplot charges vs exercise_frequency')
plt.xlabel('exercise_frequency')
plt.ylabel('charges') 
plt.grid()
plt.show()

#%%

model_EF= ols('charges ~ C(exercise_frequency)',
            data=df).fit()
result_EF = sm.stats.anova_lm(model_EF, type=1)
  
# Print the result
print(result_EF, "\n")

tukey_EF = pairwise_tukeyhsd(endog=df['charges'], groups=df['exercise_frequency'], alpha=0.05)
print(tukey_EF)

#%%

### TWO-WAY ANOVAs

# Charges vs smoker vs gender 
twoway_model = ols('charges ~ C(smoker) + C(age) + C(smoker):C(age)',
            data=df).fit()
result = sm.stats.anova_lm(twoway_model, type=2)
print(result, "\n")

combination = df.smoker + df.age

tukey_TWM2 = pairwise_tukeyhsd(endog=df['charges'], groups=combination , alpha=0.05)

print(tukey_TWM2)

##################################################
#<<<<<<<<<<<<<<<< End of Section >>>>>>>>>>>>>>>>#


#%%

#################
## Correlation ##
#################

sns.pairplot(df)
plt.show()

#%%

sns.distplot(df['charges'])

#%%

sns.set(rc={'figure.figsize':(14,8)})
correlation_matrix = df.corr()
#print(correlation_matrix)
sns.heatmap(correlation_matrix, annot=True, cmap='YlGnBu')
plt.show()

# Scaling of numerical variables (ie: age and bmi) not needed because they are in almost same range.

##################################################
#<<<<<<<<<<<<<<<< End of Section >>>>>>>>>>>>>>>>#

#%%

#####################
## Split Data ##
#####################

X = df.drop('charges', axis=1)
y = df['charges']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("X_Train set shape: ", X_train.shape)
print("y_Train set shape: ", y_train.shape)
print("X_test set shape: ", X_test.shape)
print("y_test set shape: ", y_test.shape)

#%%

######################
# VIF of Features ####
######################

X = df.drop('charges', axis=1)
y = df['charges']
vif_data = pd.DataFrame()
vif_data["Variable"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print("VIF Data: \n", vif_data)

#%%

#####################
## Model Selection ##
#####################

########## Linear Regression:
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

#%%
########## Linear Regression Model ##########

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

#%%
#### Interpretation of Data:

########### LR Results Plot ##################
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


#%%
########## Support Vector Regression:

########## Random Forest Regression:

#%%

########## Gradient Boost Regression:

#### Baseline model:

cv = KFold(n_splits=10, shuffle=True, random_state=1)
for depth in range(1,10):
    tree_regressor = tree.DecisionTreeRegressor(max_depth=depth, random_state=1)
    if tree_regressor.fit(X_train,y_train).tree_.max_depth < depth:
        break
    score_depth = np.mean(cross_val_score(tree_regressor, X, y, scoring='neg_mean_squared_error', cv=cv))
    print(depth,score_depth)

# depth 9 have less score compared to other depth values

# Results:
# score = np.mean(cross_val_score(tree_regressor, X, y, scoring='neg_mean_squared_error', cv=cv))
# 1 -13247943.420627367
# 2 -10510672.377096811
# 3 -8250142.040627992
# 4 -5999409.474873242
# 5 -3909488.221551772
# 6 -2426197.886180152
# 7 -1738341.1798021798
# 8 -1398207.0646736831
# 9 -1106418.8964559727

#%%

GBR1 = GradientBoostingRegressor()
GBR1.fit(X_train,y_train)
y_pred = GBR1.predict(X_test)
y_pred_train1=GBR1.predict(X_train)
score = np.mean(cross_val_score(GBR1, X, y, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1))

baseline_errors = abs(y_pred - y_test)
baseline_mape = 100 * np.mean((baseline_errors / y_test))
baseline_accuracy = 100 - baseline_mape

print('Metrics for Gradient boosting model for baseline')
print("cross_val_score: ", round(score, 2))
print("Mean squared error: ", mean_squared_error(y_test, y_pred))
print('Average absolute error:', round(np.mean(baseline_errors), 2))

print('mean absolute percentage error (MAPE):', baseline_mape)
print('Accuracy:', round(baseline_accuracy, 2), '%.')

print("Train R2 score: ", round(r2_score(y_train,y_pred_train1),4))
print("Test R2 score: ", round(r2_score(y_test,y_pred),4))

# Results: 
# Metrics for Gradient boosting model for baseline
# cross_val_score:  -354295.11
# Mean squared error:  356467.4813048484
# Average absolute error: 476.51
# mean absolute percentage error (MAPE): 3.085583236997528
# Accuracy: 96.91 %.

# Train R2 score:  0.9818
# Test R2 score:  0.9817


# NMSE - lower, AAE - lower, MAPE - lower, R2 - higher.

#%%
#### Hyperparameter tuning :

# Running these codes for parameter tuning takes longer time:

# GBR = GradientBoostingRegressor()
# search_grid = {'n_estimators': [19,25,50], 'learning_rate': [0.2,0.5,0.8], 'max_depth': [9,10,12], 'random_state': [1]}
# search = GridSearchCV(estimator=GBR, param_grid=search_grid, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1, cv=5)
# search.fit(X_train,y_train)
# print(search.best_params_)
# print(search.best_score_)
# print(search.best_estimator_)

# Results:
# GradientBoostingRegressor(learning_rate=0.8, max_depth=9, n_estimators=25,random_state=1)

#%%

#### modeling gradient boosting after tuning:

GBR2 = GradientBoostingRegressor(n_estimators=25, learning_rate=0.8, subsample= 0.9, max_depth=9, random_state=1)
GBR2.fit(X_train,y_train)
y_pred = GBR2.predict(X_test)
y_pred_train2=GBR2.predict(X_train)
score = np.mean(cross_val_score(GBR2, X, y, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1))

errors = abs(y_pred - y_test)
mape = 100 * np.mean((errors / y_test))
accuracy = 100 - mape

print('Metrics for Gradient boosting model after tuning')
print("cross_val_score: ", round(score, 2))
print("Mean squared error: ", mean_squared_error(y_test, y_pred))
print('Average absolute error:', round(np.mean(errors), 2))

print('mean absolute percentage error (MAPE):', mape)
print('Accuracy:', round(accuracy, 2), '%.')

print("Train R2 score: ", round(r2_score(y_train,y_pred_train2),4))
print("Test R2 score: ", round(r2_score(y_test,y_pred),4))

# Results:
# Metrics for Gradient boosting model after tuning
# cross_val_score:  -122656.38
# Mean squared error:  126511.28304750347
# Average absolute error: 292.57
# mean absolute percentage error (MAPE): 1.899894828242299
# Accuracy: 98.1 %.

# Train R2 score:  0.9941
# Test R2 score:  0.9935


#%%

#### Variable importance:

feature_list = list(X.columns)
importances = list(GBR1.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

plt.style.use('fivethirtyeight')
x_values = list(range(len(importances)))
plt.bar(x_values, importances, orientation = 'vertical', color = 'r', edgecolor = 'k', linewidth = 1.2)
plt.xticks(x_values, feature_list, rotation='vertical')
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances')

# Results:

# Variable: smoker               Importance: 0.33
# Variable: coverage_level       Importance: 0.22
# Variable: medical_history      Importance: 0.18
# Variable: family_medical_history Importance: 0.18
# Variable: exercise_frequency   Importance: 0.03
# Variable: occupation           Importance: 0.03
# Variable: gender               Importance: 0.01
# Variable: bmi                  Importance: 0.01
# Variable: children             Importance: 0.01
# Variable: age                  Importance: 0.0
# Variable: region               Importance: 0.0

#  smoker, medical_history, family_medical_history, coverage_level - higher related features.


#%%

########## XG Boost Regression:

#### Baseline model:

XG1 = XGBRegressor()
XG1.fit(X_train,y_train)
y_pred = XG1.predict(X_test)
y_pred_train3 = XG1.predict(X_train)

score = np.mean(cross_val_score(XG1, X, y, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1))

baseline_errors = abs(y_pred - y_test)
baseline_mape = 100 * np.mean((baseline_errors / y_test))
baseline_accuracy = 100 - baseline_mape

print('Metrics for XG Boost model for baseline')
print("cross_val_score: ", round(score, 2))
print("Mean squared error: ", mean_squared_error(y_test, y_pred))
print('Average absolute error:', round(np.mean(baseline_errors), 2))

print('mean absolute percentage error (MAPE):', baseline_mape)
print('Accuracy:', round(baseline_accuracy, 2), '%.')

print("Train R2 score: ", round(r2_score(y_train,y_pred_train3),4))
print("Test R2 score: ", round(r2_score(y_test,y_pred),4))

# Results:
# Metrics for XG Boost model for baseline
# cross_val_score:  -119241.14
# Mean squared error:  120503.16185165636
# Average absolute error: 286.93
# mean absolute percentage error (MAPE): 1.8464141812974073
# Accuracy: 98.15 %.
# Train R2 score:  0.9939
# Test R2 score:  0.9938

# NMSE - lower, AAE - lower, MAPE - lower, R2 - higher.

#%%
# Hyperparameter tuning :

# Running these codes for parameter tuning takes longer time:

# XG = XGBRegressor()
# search_grid = {'n_estimators': [15,20,200], 'gamma':[0,0.15,0.3,0.5,1], 'max_depth': [4,5,9], 'random_state': [1]}
# search = GridSearchCV(estimator=XG, param_grid=search_grid, scoring='neg_mean_squared_error', n_jobs=-1, cv=cv)
# search.fit(X_train,y_train)
# print(search.best_params_)
# print(search.best_score_)
# print(search.best_estimator_)

# Results:
# XGBRegressor(n_estimators=200, gamma=0, max_depth=4, random_state=1)

#%%

# modeling gradient boosting after tuning:

XG2 = XGBRegressor(n_estimators=200, gama=0.3, max_depth=4, random_state=1)
XG2.fit(X_train,y_train)
y_pred = XG2.predict(X_test)
y_pred_train4=XG2.predict(X_train)
score = np.mean(cross_val_score(XG2, X, y, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1))

# -225094.9911112765

errors = abs(y_pred - y_test)
mape = 100 * np.mean((errors / y_test))
accuracy = 100 - mape

print('Metrics for XG Boost model after tuning')
print("cross_val_score: ", round(score, 2))
print("Mean squared error: ", mean_squared_error(y_test, y_pred))
print('Average absolute error:', round(np.mean(errors), 2))

print('mean absolute percentage error (MAPE):', mape)
print('Accuracy:', round(accuracy, 2), '%.')

print("Train R2 score: ", round(r2_score(y_train,y_pred_train4),4))
print("Test R2 score: ", round(r2_score(y_test,y_pred),4))

# Results:
# Metrics for XG Boost model after tuning
# cross_val_score:  -103237.37
# Mean squared error:  103509.93648385104
# Average absolute error: 270.27
# mean absolute percentage error (MAPE): 1.7459202766297914
# Accuracy: 98.25 %.

# Train R2 score:  0.9941
# Test R2 score:  0.9947

#%%

#### Final Model with important features:

features=pd.DataFrame(data=XG2.feature_importances_,index=X.columns,columns=['Importance'])
important_features=features[features['Importance']>0.04]
print("Important Features: \n", important_features.sort_values(by="Importance"))

Xf= df[['smoker', 'coverage_level', 'medical_history', 'family_medical_history', 'exercise_frequency']]
xtrain,xtest,ytrain,ytest=train_test_split(Xf,y,test_size=0.33,random_state=42)
finalmodel=XGBRegressor(n_estimators=200, gamma=0.3, max_depth=4, random_state=1)
finalmodel.fit(xtrain,ytrain)
ypred_train_final=finalmodel.predict(xtrain)
ypred_test_final=finalmodel.predict(xtest)

errors = abs(y_pred - y_test)
mape = 100 * np.mean((errors / y_test))
accuracy = 100 - mape

print('Metrics for Final model of important features')
print("Mean squared error: ", mean_squared_error(y_test, y_pred))
print('Average absolute error:', round(np.mean(errors), 2))

print('mean absolute percentage error (MAPE):', mape)
print('Accuracy:', round(accuracy, 2), '%.')

print("Train R2 score: ", round(r2_score(y_train,y_pred_train4),4))
print("Test R2 score: ", round(r2_score(y_test,y_pred),4))

# Results:
# Metrics for Final model of important features
# Mean squared error:  103509.93648385104
# Average absolute error: 270.27
# mean absolute percentage error (MAPE): 1.7459202766297914
# Accuracy: 98.25 %.

# Train R2 score:  0.9947
# Test R2 score:  0.9947

###################################################################################

# %%
