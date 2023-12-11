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


from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import mutual_info_regression
import scipy.stats as stats
import statsmodels.api as sm
from scipy.stats import f_oneway
from scipy.stats import chi2_contingency
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error  
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor 
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

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
print(df_input.duplicated().sum())

# null is present in medical_history and family_medical_history.
df = df_input.copy()
df.drop_duplicates(inplace=True)
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
medical_history: Diabetes- 0, Heart disease- 1, High blood pressure- 2, None- 3
region: northeast- 0, northwest-1 , southeast-2 , southwest- 3, 
family_medical_history: Heart disease- 1,High blood pressure- 2,Diabetes- 0, None- 3
exercise_frequency: Frequently-0 ,never-1 ,occasionaly-2 ,rarely-3 
occupation : blue collar- 0, student- 1,unemployed- 2,whitecollar- 3
coverage_level : Basic- 0,Premium- 1,Standard- 2  
"""

#%%

# df[['age', 'bmi', 'children']] = scaler.fit_transform(df[['age', 'bmi', 'children']])
df = pd.get_dummies(df,columns=['gender', 'smoker', 'region', 'medical_history', 'family_medical_history', 'exercise_frequency', 'occupation', 'coverage_level'])
boolean_columns = df.select_dtypes(include='bool').columns

for i in boolean_columns:
    df[i] = df[i].astype(int)

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

# Scatter plot of age vs charges:

age = pd.pivot_table(df, values='charges', columns='age', aggfunc='mean')
melt_age = pd.melt(age, var_name='age', value_name='average_charges')

sns.scatterplot(x='age', y='average_charges', data=melt_age, alpha=0.5, color='blue')

plt.title('Scatter Plot of Average Charges for Each Age')
plt.xlabel('Age')
plt.ylabel('Average Charges')
plt.show()

# age and charges are almost linear to each other.
# Thus, when age increase charges increase.


#%%

print('\nGraphical Exploration of categorical variables:\n')

# Coverage level vs Charges:

plt.figure(figsize=(10, 6))

# Create a box plot between charges and coverage_level.
df.boxplot(column='charges', by='coverage_level', vert=False, flierprops={'marker': 'o', 'markerfacecolor': 'red', 'markersize': 5})
plt.suptitle('')  # Remove the default title
plt.xticks(rotation=45, ha='right') 
plt.title('Box Plot of Charges by coverage_level')
plt.xlabel('Charges')
plt.ylabel('coverage_level')
plt.tight_layout()
plt.show()

# The analysis exactly reveals that Premium customers incur the highest charges, 
# trailed by Standard and Basic customers in descending order.


#%%

# Smoker vs Charges :

print("Smoker counts: ", df['smoker'].value_counts())
# Smoker count is more 

sns.set(style="whitegrid")
# Create a KDE plot
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df, x='charges', hue='smoker', fill=True, common_norm=False, palette='viridis')
plt.xlabel('Charges')
plt.ylabel('Density')
plt.title('KDE Plot of Charges vs. Smoker')
plt.show()

# The KDE plot clearly illustrates that more of Smokers tend to incur higher charges 
# compared to non-smokers.

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

# Gender vs Charges:

print("Gender counts: ", df['gender'].value_counts())
# male      500107
# female    499893

plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='gender', y='charges', palette='pastel')
plt.xlabel('Gender')
plt.ylabel('Charges')
plt.title('Barplot of Charges vs. Gender')
plt.show()

# On average, males tend to give higher charges than females. 
# Additionally, the maximum charge in the dataset is paid by a male, while the 
# minimum charges are associated with females.

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

# exercise_frequency: Frequently-0 ,never-1 ,occasionaly-2 ,rarely-3 

# people with rarely exercise freq is high in count and the least is people with never exercise freq

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

# medical_freq vs smoker vs charges

sns.set(style="whitegrid")
# Create a grouped bar chart
plt.figure(figsize=(12, 8))
sns.barplot(data=df, x='medical_history', y='charges', hue='smoker', palette='muted')

# Set plot labels and title
plt.xlabel('medical history')
plt.ylabel('Charges')
plt.title('Smoker vs. Medical History vs. Charges')
plt.xticks([0,1,2,3], ['Diabets', 'Heart disease', 'High blood pressue', 'None'])
# Show the legend
plt.legend(title='Smoker')
# Show the plot
plt.show()

# Individuals who have heart disease and have a smoking habit tend to incur higher charges. 
# Conversely, the lowest charges are observed among individuals with no medical history and no smoking habit.


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

# T test between charges of male and female.
t_stat, p_value = stats.ttest_ind(charges_female, charges_male)

print(f"T test Result: \n P value: {p_value}")
alpha = 0.05
# Check the p-value
if p_value < alpha: 
    print(" Reject the null hypothesis; There is a significant difference in charges between gender")
else:
    print(" There is no significant difference in charges between gender")

# Based on both the graphical representation and the results of the t-test, it is 
# evident that there is variation among the mean values of charges across male and female.

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

# T test between charges of smoker and non-smoker.
t_stat, p_value = stats.ttest_ind(charges_smoker, charges_nonsmoker)

print(f"T test Result: \n P value: {p_value}")
# Check the p-value
if p_value < alpha: 
    print(" Reject the null hypothesis; There is a significant difference in charges between smoker and nonsmoker")
else:
    print(" There is no significant difference in charges between smoker and non-smoker")

# Based on both the graphical representation and the results of the t-test, it is 
# evident that there is variation among the mean values of charges across smokers and non smokers.

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

print(" There is significant difference among the means of medical_history groups of charges")

tukey_MH = pairwise_tukeyhsd(endog=df['charges'], groups=df['medical_history'], alpha=0.05)
print(tukey_MH)

# Based on both the graphical representation and the results of the one-way ANOVA test, it is 
# evident that substantial variations exist among the mean values of charges across 
# different medical history groups.

# The Tukey HSD test further confirms that all pairwise differences are statistically significant.

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

print(" There is significant difference among the means of family_medical_history groups of charges")

tukey_FMH = pairwise_tukeyhsd(endog=df['charges'], groups=df['family_medical_history'], alpha=0.05)
print(tukey_FMH)

# Based on both the bar chart and the results of the one-way ANOVA test, it is 
# evident that substantial variations exist among the mean values of charges across 
# different family medical history groups.

# The Tukey HSD test further confirms that all pairwise differences are statistically significant.

#%%

#Boxplot charges vs occupation

g1 = sns.boxplot(data = df, y = 'charges', x = 'occupation', palette='Set3', showmeans = True, meanprops = {"marker": "o", "markerfacecolor":"red", "markeredgecolor":"black", "markersize":"6"}, showfliers = False)
g1.set(title ='Boxplot charges vs occupation')
plt.xlabel('occupation')
plt.ylabel('charges') 
plt.xticks([0,1,2,3], ['blue collar', 'student', 'unemployed', 'whitecollar'])
plt.grid()
plt.show()

#%%

model_OC= ols('charges ~ C(occupation)',data=df).fit()
result_OC = sm.stats.anova_lm(model_OC, type=1)
  
# Print the result
print(result_OC, "\n")

print(" There is significant difference among the means of occupation groups of charges")

tukey_OC = pairwise_tukeyhsd(endog=df['charges'], groups=df['occupation'], alpha=0.05)
print(tukey_OC)

# The Tukey HSD test further confirms that all pairwise differences are statistically significant.
# Least difference : The mean charges for Group 1 (student) are, on average, $495.5484 less than the mean charges for Group 2 (unemployed)

#%%

#Boxplot charges vs exercise_frequency

g1 = sns.boxplot(data = df, y = 'charges', x = 'exercise_frequency', palette='Set3', showmeans = True, meanprops = {"marker": "o", "markerfacecolor":"red", "markeredgecolor":"black", "markersize":"6"}, showfliers = False)
g1.set(xticklabels= charges)
g1.set(title ='Boxplot charges vs exercise_frequency')
plt.xlabel('exercise_frequency')
plt.ylabel('charges') 
plt.xticks([0,1,2,3], ['Frequently', 'never', 'occasionaly', 'rarely'])
plt.grid()
plt.show()

#%%

model_EF= ols('charges ~ C(exercise_frequency)',data=df).fit()
result_EF = sm.stats.anova_lm(model_EF, type=1)
  
# Print the result
print(result_EF, "\n")

print(" There is significant difference among the means of occupation groups of charges")

tukey_EF = pairwise_tukeyhsd(endog=df['charges'], groups=df['exercise_frequency'], alpha=0.05)
print(tukey_EF)

# The Tukey HSD test further confirms that all pairwise differences are statistically significant.
# # Least difference : The mean charges for Group 1 (never) are, on average, $481.7174 more than the mean charges for Group 3 (rarely)

#%%

# charges vs region

charges_by_region = [df['charges'][df['region'] == region] for region in df['region'].unique()]

f_statistic, p_value = f_oneway(*charges_by_region)

print(f"F-statistic: {f_statistic}")
print(f"P-value: {p_value}")

'''
There is a significant difference in charges between different regions.
'''
tukey_region = pairwise_tukeyhsd(endog=df['charges'], groups=df['region'], alpha=0.05)
print(tukey_region)

# The ANOVA test revealed significantly different mean insurance charges among four regions 
# Subsequent Tukey HSD analysis indicated specific pairwise differences in mean charges.

#%%

### 2 way ANOVA

twoway_model1 = ols('charges ~ C(medical_history) + C(exercise_frequency) + C(medical_history):C(exercise_frequency)',
            data=df).fit()
result = sm.stats.anova_lm(twoway_model1, type=2)
print(result, "\n")

# The two-way ANOVA results suggest that both 'smoker' and 'exercise_frequency' have a significant impact on the charges. 
# However, the interaction effect between these two factors is not statistically significant.

#%%

twoway_model2 = ols('charges ~ C(exercise_frequency) + C(smoker) + C(smoker):C(exercise_frequency)',
            data=df).fit()
result = sm.stats.anova_lm(twoway_model2, type=2)
print(result, "\n")

# The two-way ANOVA results suggest that both 'smoker' and 'medical_history' have a significant impact on the charges. 
# the interaction effect between these two factors is also statistically significant.

#%%

### Chi sq test 

# gender vs medical history

contingency_table = pd.crosstab(df['gender'], df['medical_history'])

chi2, p, dof, expected = chi2_contingency(contingency_table)

print("Chi-square test for independence between gender and medical history:")
print(f"Chi2 value: {chi2}")
print(f"P-value: {p}")

'''
There's no Significant assosciation between Gender and Medical history of an individual
'''

#%%

# smoker vs medical history

contingency_table = pd.crosstab(df['smoker'], df['medical_history'])

chi2, p, dof, expected = chi2_contingency(contingency_table)

print("Chi-square test for independence between smoker and medical history:")
print(f"Chi2 value: {chi2}")
print(f"P-value: {p}")

'''
There is no significant assosciation between smoker and Medical history of an individual
'''

#%%

# smoker vs exercise frequency

contingency_table = pd.crosstab(df['smoker'], df['exercise_frequency'])

chi2, p, dof, expected = chi2_contingency(contingency_table)

print("Chi-square test for independence between smoker and exercise_frequency:")
print(f"Chi2 value: {chi2}")
print(f"P-value: {p}")

'''
There is significant assosciation between smoker and exercise_frequency of an individual
'''

##################################################
#<<<<<<<<<<<<<<<< End of Section >>>>>>>>>>>>>>>>#

#%%

#####################
# Inference 
#####################

# Smart question 2 : In comparison to non-smokers, how much does being a "smoker" add to the rise in
# insurance costs?
# Smoker vs charges

model1 = sm.OLS.from_formula('charges ~ C(smoker)', data=df).fit()
print(model1.summary())

# From the results, we could see that smoker = 1
# impacts more on charges than smoker = 0, same is viwed from EDA as well

# smoker add 5000.5740 unit to rise in charges than non smokers.

#%%

# Smart question 3 : How much does age impact insurance premiums, and is this impact consistent across
# different regions?

# age : region

model2 = sm.OLS.from_formula('charges ~ age + age : C(region)', data=df).fit()
print(model2.summary())

# The impact of age on charges is larger (31.1384) compared to the interaction terms of age with different levels of 'region.'
# But when compared within the regions. age with region = 0 (northeast) provide impact more on charges than other regions.

#%%

# Smoker : medical history vs charges

model4 = sm.OLS.from_formula('charges ~ C(smoker) : C(medical_history)', data=df).fit()
print("LR model summary: ")
print(model4.summary())

# among the groups of medical history, we could see that medical history = 1 (heart disease) 
# impacts more on charges, same is viwed from EDA as well

# But overall, it is smoker + heart dieases which impacts the increase in charge unit.

#%%

# charges vs coveragelevel

model6 = sm.OLS.from_formula('charges ~ C(coverage_level)', data=df).fit()
print(model6.summary())

# from EDA and inference, we could see coverage level = 1 (Premium) impacts more on charges


#%%

#################
## Correlation ##
#################

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

X = df.drop(['charges'], axis=1)
y = df['charges']

#%%
#####################
# Feature Selection 
#####################

### mutual_info_regression feature selection

# Calculate scores
importance = mutual_info_regression(X,y)
# plotting ranks
feat_importances = pd.Series(importance, df.columns[0:len(df.columns)-1])
feat_importances.plot(kind='barh', color ='teal')
plt.show()

# Top features influencing charges
# Smoker
# covergae_level
# medical_history
# family_medical_history

#%%

from sklearn.preprocessing import StandardScaler, LabelEncoder

scaler = StandardScaler()
X[['age', 'bmi']] = scaler.fit_transform(X[['age', 'bmi']])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("X_Train set shape: ", X_train.shape)
print("y_Train set shape: ", y_train.shape)
print("X_test set shape: ", X_test.shape)
print("y_test set shape: ", y_test.shape)


#%%

#####################
## Model Selection ##
#####################

########## Linear Regression Model ##########

X = df[['age','gender', 'bmi','smoker', 'medical_history', 'family_medical_history', 'occupation','coverage_level']]
y = df['charges']

X_encoded = pd.get_dummies(X, columns=['gender','smoker', 'medical_history', 'family_medical_history', 'occupation','coverage_level'], drop_first=True)
from sklearn.preprocessing import StandardScaler, LabelEncoder
scaler = StandardScaler()
X_encoded[['age', 'bmi']] = scaler.fit_transform(X_encoded[['age', 'bmi']])

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

##LR Results Plot

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

###### Linear regression assumptions.

# Linearity, Outliers and Influential Points

residuals = y_test - y_pred_test
# Standardized residuals
standardized_residuals = (residuals - np.mean(residuals)) / np.std(residuals)
# Create a QQ plot
sm.qqplot(standardized_residuals, line='45', fit=True)
plt.title('QQ Plot of Observed vs. Predicted Values')
plt.show()

#%%

# linear relationship between age 

correlation_coefficient = np.corrcoef(df['age'], df['charges'])[0, 1]
print(f'Correlation Coefficient: {correlation_coefficient}')

correlation_coefficient = np.corrcoef(df['bmi'], df['charges'])[0, 1]
print(f'Correlation Coefficient: {correlation_coefficient}')

# Correlation Coefficient: 0.06339041373316799
# Correlation Coefficient: 0.10442935155994461

#%%

# Normality of charges :

# Check normality using a normal probability plot (Q-Q plot)
plt.figure(figsize=(8, 8))
sm.qqplot(df['charges'], line='45', fit=True)
plt.title('Normal Probability Plot of charges')
plt.show()

#%%

# Normality of residuals :

residuals = y_test - y_pred_test
# Check normality using a histogram of residuals
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True, bins=15)
plt.title('Histogram of Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()

# Check normality using a normal probability plot (Q-Q plot)
plt.figure(figsize=(8, 8))
sm.qqplot(residuals, line='45', fit=True)
plt.title('Normal Probability Plot of Residuals')
plt.show()

#%%
########## Support Vector Regression model ##########

svr = SVR(kernel='linear',C=3)

svr.fit(X_train[:200000], y_train[:200000])
y_pred = svr.predict(X_test)

# %%

errors = abs(y_pred - y_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)   

print('Metrics for SVM model :')
print("Mean squared error: ", mean_squared_error(y_test, y_pred))
print('Average absolute error:', round(np.mean(errors), 2))
# Calculate mean absolute percentage error (MAPE)
mape = 100 * np.mean((errors / y_test))
print('mean absolute percentage error (MAPE):', mape)
# Calculate and display accuracy
accuracy = 100 - mape
print('Accuracy:', round(accuracy, 2), '%.')
print(f'Mean Absolute Error: {mae}')
print(f'R-squared: {r2}')

# Metrics for SVM model :
# Mean squared error:  9194809.948672732
# Average absolute error: 2470.98
# mean absolute percentage error (MAPE): 15.777191941162611
# Accuracy: 84.22 %.
# Mean Absolute Error: 2470.9781478290224
# R-squared: 0.5275195874963398

# with second option
# Metrics for SVM model :
# Mean squared error:  83623.51013089206
# Average absolute error: 250.49
# mean absolute percentage error (MAPE): 1.6204962885585796
# Accuracy: 98.38 %.
# Mean Absolute Error: 250.48841222402845
# R-squared: 0.9957029595193154

# %%
########## Random Forest Regression model ##########

rfm = RandomForestRegressor(n_estimators=20, random_state=42)
rfm.fit(X_train, y_train)

y_pred = rfm.predict(X_test)

# Evaluate the model
errors = abs(y_pred - y_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('Metrics for Random forest regression model :')
print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')
mape = 100 * np.mean((errors / y_test))
print('mean absolute percentage error (MAPE):', mape)
# Calculate and display accuracy
accuracy = 100 - mape
print('Accuracy:', round(accuracy, 2), '%.')
print(f'Mean Absolute Error: {mae}')
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

#%%

########## Gradient Boost Regression ##########

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

# 1 -13247943.42
# 2  -9689109.89
# 3  -6687578.30
# 4  -3676228.84
# 5  -2837936.04
# 6  -2357374.35
# 7  -1922768.85
# 8  -1562168.49
# 9  -1252018.68

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

# second option (svr option)
# Metrics for Gradient boosting model for baseline
# Mean squared error:  240289.17438117307
# Average absolute error: 394.81
# mean absolute percentage error (MAPE): 2.625008797629864
# Accuracy: 97.37 %.
# Train R2 score:  0.9878
# Test R2 score:  0.9877

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

# Results: label encoding.
#GBR2 = GradientBoostingRegressor(n_estimators=25, learning_rate=0.8, subsample= 0.9, max_depth=9, random_state=1)
# Metrics for Gradient boosting model after tuning
# cross_val_score:  -122656.38
# Mean squared error:  126511.28304750347
# Average absolute error: 292.57
# mean absolute percentage error (MAPE): 1.899894828242299
# Accuracy: 98.1 %.

# Train R2 score:  0.9941
# Test R2 score:  0.9935

# Metrics for Gradient boosting model after tuning
# Mean squared error:  142320.90262198035
# Average absolute error: 308.0
# mean absolute percentage error (MAPE): 1.993095393673771
# Accuracy: 98.01 %.
# Train R2 score:  0.9934
# Test R2 score:  0.9927

# second choice svr check
# GBR2 = GradientBoostingRegressor(n_estimators=200, learning_rate=0.6, subsample= 0.9, max_depth=10, random_state=1)
# Metrics for Gradient boosting model after tuning
# Mean squared error:  133939.26538929652
# Average absolute error: 299.77
# mean absolute percentage error (MAPE): 1.927994322062208
# Accuracy: 98.07 %.
# Train R2 score:  0.9969
# Test R2 score:  0.9931

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

########## XG Boost Regression model ##########

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

XG2 = XGBRegressor(n_estimators=200, gamma=0.3, max_depth=4, random_state=1)
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

# Results:  label encoding
# Metrics for XG Boost model after tuning
# cross_val_score:  -103237.37
# Mean squared error:  103509.93648385104
# Average absolute error: 270.27
# mean absolute percentage error (MAPE): 1.7459202766297914
# Accuracy: 98.25 %.

# Train R2 score:  0.9941
# Test R2 score:  0.9947


XG2 = XGBRegressor(n_estimators=800, gamma=0.5, max_depth=3, random_state=1)
XG2.fit(X_train,y_train)
y_pred = XG2.predict(X_test)
y_pred_train4=XG2.predict(X_train)

# second choice svr  / label encoding.
XG2 = XGBRegressor(n_estimators=2000, gamma=0.5, max_depth=4, random_state=1)
# Metrics for XG Boost model after tuning
# Mean squared error:  85635.9558945334
# Average absolute error: 252.5
# mean absolute percentage error (MAPE): 1.6322867577811113
# Accuracy: 98.37 %.
# Train R2 score:  0.9958
# Test R2 score:  0.9956


#%%

models = ['Linear', 'Random Forest', 'SVM', 'Gradient Boosting', 'XG Boost']
mse_values = [839240.2362, 139478.7883, 84801.66551, 126511.283, 103509.9365]
accuracy_values = [95.15, None, None, 98.10, 98.25]  # None for models without accuracy values

# Plotting the Mean Squared Error
plt.figure(figsize=(10, 6))
plt.plot(models, mse_values, marker='o', label='M.S.E')
plt.title('Mean Squared Error (MSE) for Different Regression Models')
plt.xlabel('Regression Models')
plt.ylabel('Mean Squared Error (MSE)')
plt.legend()
plt.grid(True)
plt.show()

# Plotting the Accuracy
plt.figure(figsize=(10, 6))
plt.plot(models, accuracy_values, marker='o', label='Accuracy', color='green')
plt.title('Accuracy for Different Regression Models')
plt.xlabel('Regression Models')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)
plt.show()

#%%

# dummies and scaling
models = ['Linear', 'SVR', 'Random Forest', 'Gradient Boosting', 'XG Boost']
mse_values = [839240.23, 83623.51, 130913.70, 133939.26, 85635.95]

# Plotting the Mean Squared Error
plt.figure(figsize=(10, 6))
plt.plot(models, mse_values, marker='o', label='M.S.E')
plt.title('Mean Squared Error (MSE) for Different Regression Models')
plt.xlabel('Regression Models')
plt.ylabel('Mean Squared Error (MSE)')
plt.legend()
plt.grid(True)
plt.show()

#%%

# Label encoding
models = ['Linear', 'SVR', 'Random Forest','Gradient Boosting', 'XG Boost']
mse_values = [9157484.04, 9194809.95, 137789.78, 126511.28, 85780.12]

# Plotting the Mean Squared Error
plt.figure(figsize=(10, 6))
plt.plot(models, mse_values, marker='o', label='M.S.E')
plt.title('Mean Squared Error (MSE) for Different Regression Models')
plt.xlabel('Regression Models')
plt.ylabel('Mean Squared Error (MSE)')
plt.legend()
plt.grid(True)
plt.show()
#%%

#### Final Model with important features:

features=pd.DataFrame(data=XG2.feature_importances_,index=X.columns,columns=['Importance'])
important_features=features[features['Importance']>0.04]
print("Important Features: \n", important_features.sort_values(by="Importance"))

Xf= df[['smoker', 'coverage_level', 'medical_history', 'family_medical_history', 'occupation']]
xtrain,xtest,ytrain,ytest=train_test_split(Xf,y,test_size=0.33,random_state=42)
finalmodel=XGBRegressor(n_estimators=800, gama=0.5, max_depth=3, random_state=1)
finalmodel.fit(xtrain,ytrain)
ypred_train_final=finalmodel.predict(xtrain)
ypred_test_final=finalmodel.predict(xtest)

errors = abs(ypred_test_final - ytest)
mape = 100 * np.mean((errors / ytest))
accuracy = 100 - mape

print('Metrics for Final model of important features')
print("Mean squared error: ", mean_squared_error(ytest, ypred_test_final))
print('Average absolute error:', round(np.mean(errors), 2))

print('mean absolute percentage error (MAPE):', mape)
print('Accuracy:', round(accuracy, 2), '%.')

print("Train R2 score: ", round(r2_score(ytrain,ypred_train_final),4))
print("Test R2 score: ", round(r2_score(ytest,ypred_test_final),4))

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
