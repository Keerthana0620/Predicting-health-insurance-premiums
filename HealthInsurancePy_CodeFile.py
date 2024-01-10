# To add a new cell, type ''
# To add a new markdown cell, type '#%%[markdown]'

#%%[markdown]
#
# # Predicting Health Insurance Premiums in the US : Data Mining Project
# ## By: Keerthana Aravindhan, Amit Sopan Shendge, Vamsidhar Boddu.
# ### Date: Dec 14 2023
#
# 
# We have the Insurance dataset.
# The variables in the dataset are:  
# * `age` : Age in years
# * `gender` : Male / female
# * `bmi` : bmi count
# * `children` : number of children
# * `smoker` : Yes/no
# * `region` : northeast, northwest, southeast, southwest
# * `medical_history` : Diabetes/High blood pressure/Heart disease/None
# * `family_medical_history` : Diabetes/High blood pressure/Heart disease/None
# * `exercie` : frequently, occasionaly, Rarely, Never
# * `occupation` : White collar, blue collar, student, unemployed
# * `coverage_level` : Basic, premium, standard
# * `charges` : premium insurance charges amount. (target)


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
import statsmodels.api as sm
from scipy.stats import f_oneway
from scipy.stats import chi2_contingency
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

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

#Select Color Palette
sns.set_palette('Set2')

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
    return x

# converting categorical variables with get dummies
def OneHotEncoding(x):
    
    x = pd.get_dummies(x, columns=['gender', 'smoker', 'region', 'medical_history', 'family_medical_history', 'exercise_frequency', 'occupation', 'coverage_level'])
    boolean_columns = x.select_dtypes(include='bool').columns
    for i in boolean_columns:
        x[i] = x[i].astype(int)
    return x

##################################################
#<<<<<<<<<<<<<<<< End of Section >>>>>>>>>>>>>>>>#


#%%

###############
## Load Data ##
###############

df_input = pd.read_csv("insurance_dataset.csv")
df_input.head()

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
df_input.drop_duplicates(inplace=True)
df_input['medical_history'].fillna('None', inplace=True)
df_input['family_medical_history'].fillna('None', inplace=True)

# After filling
print("\nAfter filling missing values:\n")
print(df_input.isnull().sum())

#%%

### label encoding

df_label = df_input.copy()
# Convert columns to desired data types
print(df_label.dtypes)

# Converting categorical variables into int
df_label = Encoder(df_label)
print("\n")
print (df_label.head())

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
print("Data Types after label encoding conversion \n")
print(df_label.dtypes)

#%%

### One-hot encoding

df_encoded = df_input.copy()

# Converting categorical variables into dummies and int
df_encoded = OneHotEncoding(df_encoded)
print("Dataset after one hot encoding conversion \n")
print("\n")
print (df_encoded.head())


#%%
print("Data Types after one hot encoding conversion \n")
print(df_encoded.dtypes)

##################################################
#<<<<<<<<<<<<<<<< End of Section >>>>>>>>>>>>>>>>#

#%%

######################################################
## Exploratory data analysis with Statistical tests ##
######################################################

##### Charge Distribution #####

print('\nCharges Distribution')
sns.histplot(data=df_input, x='charges', bins=30, color='m')
plt.xlabel ('Charges ', size=12)
plt.ylabel('Count', size=12)
plt.title('Distribution of Charges', size=14)
plt.show()

# The dataset exhibits a notable absence of outliers, and 
# the distribution of charges aligns closely with a normal distribution.

#%%

##### Charges vs Age ##### 

# Scatter plot of charges for each age:

age = pd.pivot_table(df_input, values='charges', columns='age', aggfunc='mean')
melt_age = pd.melt(age, var_name='age', value_name='average_charges')

sns.scatterplot(x='age', y='average_charges', data=melt_age, alpha=0.5, color='blue')
plt.title('Scatter Plot of Average Charges for Each Age')
plt.xlabel('Age')
plt.ylabel('Average Charges')
plt.show()

# Age and average charges for each age is almost linear to each other.
# Thus, when age increase average charges also increase.

#%%

##### Charges vs Regions #####

# Barplot of charges across regions.
 
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.barplot(x='region', y='charges', data=df_input)
plt.title('Insurance Charges Across Regions')
plt.xlabel('Region')
plt.ylabel('Charges')
plt.show()

# The bar plot does not distinctly reveal variations in charges across different regions.

#%%

# One way ANOVA test of charges vs region

charges_by_region = [df_input['charges'][df_input['region'] == region] for region in df_input['region'].unique()]
f_statistic, p_value = f_oneway(*charges_by_region)

print(f"F-statistic: {f_statistic}")
print(f"P-value: {p_value}")

'''
There is a significant difference in charges between different regions.
'''
tukey_region = pairwise_tukeyhsd(endog=df_input['charges'], groups=df_input['region'], alpha=0.05)
print(tukey_region)

# The ANOVA test revealed significantly different mean insurance charges among four regions 
# Subsequently, Tukey HSD analysis indicated specific pairwise differences in mean charges.

#%%

# Inference from model with age and region interaction term.

model1 = sm.OLS.from_formula('charges ~ age + age : C(region)', data=df_input).fit()
print(model1.summary())

# The impact of age on charges is larger (31.1384) compared to the interaction terms of age with different levels of 'region.'
# But when compared within the regions. age with region = 0 (northeast) provide impact more on charges than other regions.
# which is also revealed from the bar chart.

# Analysis:
# Both visulization and anova test uncovered significant differences in mean insurance charges among the four regions. But from the inference, it says
# that neither age nor region is impacting charges.

#%%

##### Charges vs Smoker ##### 

# KDE plot of Charges for smoking and non smoking.

print("Smoker counts: ", df_input['smoker'].value_counts())
# Smoker count is more 

sns.set(style="whitegrid")
# Create a KDE plot
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df_input, x='charges', hue='smoker', fill=True, common_norm=False, palette='viridis')
plt.xlabel('Charges')
plt.ylabel('Density')
plt.title('KDE Plot of Charges vs. Smoker')
plt.show()

# The KDE plot clearly illustrates that more of Smokers tend to incur higher charges 
# compared to non-smokers.

#%%

# Statistical T-test between Charges and Smoker:

df_input.boxplot(column='charges', by='smoker', vert=False, flierprops={'marker': 'o', 'markerfacecolor': 'red', 'markersize': 5})
plt.suptitle('')  # Remove the default title
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.title('Box Plot of Charges by smoker')
plt.xlabel('Charges')
plt.ylabel('smoker')
plt.tight_layout()
plt.show()

charges = df_input['charges']
charges_smoker = charges[df_input['smoker']=='yes']
charges_nonsmoker = charges[df_input['smoker']=='no']

# T test between charges of smoker and non-smoker.
t_stat, p_value = stats.ttest_ind(charges_smoker, charges_nonsmoker)

alpha = 0.05
print(f"T test Result: \n P value: {p_value}")

""" Reject the null hypothesis; There is a significant difference in charges between smoker and nonsmoker """

#%%

# Inference from model with smoker against charges

model2 = sm.OLS.from_formula('charges ~ C(smoker)', data=df_input).fit()
print(model2.summary())

# From the results, we could see that smoker impacts more on charges than non smoker, same is viewed from EDA as well
# smoker add 5000.5740 unit to rise in charges than non smokers.

#%%[markdown]

# ### Analysis:
# Upon analyzing graphical representations, t-test results, and inferential findings, 
# a clear distinction emerges in mean charge values between smokers and non-smokers, with 
# smokers incurring higher charges.

#%%

##### Charges vs gender ##### 

# Barplot of charges among gender

print("Gender counts: ", df_input['gender'].value_counts())
# male      500107
# female    499893

plt.figure(figsize=(10, 6))
sns.barplot(data=df_input, x='gender', y='charges', palette='pastel')
plt.xlabel('Gender')
plt.ylabel('Charges')
plt.title('Barplot of Charges vs. Gender')
plt.show()

# On average, males tend to give higher charges than females. 
# Additionally, the maximum charge in the dataset is paid by a male, while the 
# minimum charges are associated with females.

#%%

# Statistical T-test between charges and gender.

df_input.boxplot(column='charges', by='gender', vert=False, flierprops={'marker': 'o', 'markerfacecolor': 'red', 'markersize': 5})
plt.suptitle('')  # Remove the default title
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.title('Box Plot of Charges by gender')
plt.xlabel('Charges')
plt.ylabel('gender')
plt.tight_layout()
plt.show()

charges_male = charges[df_input['gender']=='male']
charges_female = charges[df_input['gender']=='female']

# T test between charges of male and female.
t_stat, p_value = stats.ttest_ind(charges_female, charges_male)

print(f"T test Result: \n P value: {p_value}")

""" Reject the null hypothesis; There is a significant difference in charges between gender """

#%%[markdown]
# ### Analysis:
# Based on both the graphical representation and the results of the t-test, it is 
# evident that there is variation among the mean values of charges across male and female.

#%%

##### Charges vs Medical History #####

# Barplot of charges among medical history

medical = pd.pivot_table(df_input, values='charges', columns=['medical_history'], aggfunc='mean')
melt_medical = pd.melt(medical, var_name='medical_history', value_name='average_charges')

# Create a bar chart with 'hue' for gender
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")
sns.barplot(data=melt_medical, x='medical_history', y='average_charges', color='g')
plt.xlabel('medical_history')
plt.ylabel('Average Charges')
plt.title('Average Charges by Medical_history')
plt.tight_layout()
plt.show()

# The average charges are notably higher for individuals with heart disease, followed by those with diabetes 
# in medical_history.

#%%

# One way ANOVA test between charges and medical history.

# Boxplot charges vs medical_history
g1 = sns.boxplot(data = df_input, y = 'charges', x = 'medical_history', palette='Set3', showmeans = True, meanprops = {"marker": "o", "markerfacecolor":"red", "markeredgecolor":"black", "markersize":"6"}, showfliers = False)
g1.set(title ='Boxplot charges vs medical_history')
plt.xlabel('medical_history')
plt.ylabel('charges') 
plt.grid()
plt.show()

#%%

model_MH= ols('charges ~ C(medical_history)',data=df_input).fit()
result_MH = sm.stats.anova_lm(model_MH, type=1)
  
# Print the result
print(result_MH, "\n")

print(" There is significant difference among the means of medical_history groups of charges \n")

tukey_MH = pairwise_tukeyhsd(endog=df_input['charges'], groups=df_input['medical_history'], alpha=0.05)
print(tukey_MH)

#%%

# Chi sq test to check relation of gender with medical history

contingency_table = pd.crosstab(df_input['gender'], df_input['medical_history'])

chi2, p, dof, expected = chi2_contingency(contingency_table)

print("Chi-square test of independence between gender and medical history:")
print(f"Chi2 value: {chi2}")
print(f"P-value: {p}")

'''
There's no Significant assosciation between Gender and Medical history of an individual
'''

# Chi sq test to check relation of smoker with medical history

contingency_table = pd.crosstab(df_input['smoker'], df_input['medical_history'])

chi2, p, dof, expected = chi2_contingency(contingency_table)

print("\nChi-square test of independence between smoker and medical history:")
print(f"Chi2 value: {chi2}")
print(f"P-value: {p}")

'''
There is no significant assosciation between smoker and Medical history of an individual
'''
#%%[markdown]
# ### Analysis :
# Based on both the graphical representation and the results of the one-way ANOVA test, it is 
# evident that substantial variations exist among the mean values of charges across 
# different medical history groups.
# The Tukey HSD test further confirms that all pairwise differences are statistically significant.

# Also chi sq tests reveals that there is no significant association of medical history with gender or smoker.

#%%

# bar plots on medical_freq vs smoker vs charges

sns.set(style="whitegrid")
# Create a grouped bar chart
plt.figure(figsize=(12, 8))
sns.barplot(data=df_input, x='medical_history', y='charges', hue='smoker', palette='muted')

# Set plot labels and title
plt.xlabel('medical history')
plt.ylabel('Charges')
plt.title('Smoker vs. Medical History vs. Charges')
# Show the legend
plt.legend(title='Smoker')
# Show the plot
plt.show()

# Individuals who have heart disease and have a smoking habit tend to incur higher charges. 
# Conversely, the lowest charges are observed among individuals with no medical history and no smoking habit.

#%%

##### Charges vs Family Medical History ##### 

# Barplot of charges among family medical history

family_medical = pd.pivot_table(df_input, values='charges', columns=['family_medical_history'], aggfunc='mean')
# Melt the pivot table to a long format
melt_family_medical = pd.melt(family_medical, var_name='family_medical_history', value_name='average_charges')

# Create a bar chart with 'hue' for gender
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")
sns.barplot(data=melt_family_medical, x='family_medical_history', y='average_charges')
plt.xlabel('family_medical_history')
plt.ylabel('Average Charges')
plt.title('Average Charges by family_medical_history')
plt.tight_layout()
plt.show()

# The average charges are notably higher for individuals with heart disease, followed by those with diabetes
#  in family_medical_history.

#%%

# One way ANOVA test between charges and family medical history.

# Boxplot charges vs family_medical_history
g1 = sns.boxplot(data = df_input, y = 'charges', x = 'family_medical_history', palette='Set3', showmeans = True, meanprops = {"marker": "o", "markerfacecolor":"red", "markeredgecolor":"black", "markersize":"6"}, showfliers = False)
g1.set(title ='Boxplot charges vs family_medical_history')
plt.xlabel('family_medical_history')
plt.ylabel('charges') 
plt.grid()
plt.show()

#%%

model_FMH= ols('charges ~ C(family_medical_history)',data=df_input).fit()
result_FMH = sm.stats.anova_lm(model_FMH, type=1)
  
# Print the result
print(result_FMH, "\n")
print(" There is significant difference among the means of family_medical_history groups of charges")

tukey_FMH = pairwise_tukeyhsd(endog=df_input['charges'], groups=df_input['family_medical_history'], alpha=0.05)
print(tukey_FMH)

#%%[markdown]
# ### Analysis:
# Based on both the bar chart and the results of the one-way ANOVA test, it is 
# evident that substantial variations exist among the mean values of charges across 
# different family medical history groups.
# The Tukey HSD test further confirms that all pairwise differences are statistically significant.

#%%

##### Charges vs Occupation #####

# Bar plot for charges among occupation

print("Occupation counts: ", df_input['occupation'].value_counts())
occupation = pd.pivot_table(df_input, values='charges', columns=['occupation'], aggfunc='mean')
# Melt the pivot table to a long format
melt_occupation = pd.melt(occupation, var_name='occupation', value_name='average_charges')

# Create a bar chart with 'hue' for gender
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")
sns.barplot(data=melt_occupation, x='occupation', y='average_charges', color='pink')
plt.xlabel('occupation')
plt.ylabel('Average Charges')
plt.title('Average Charges by occupation')
plt.tight_layout()
plt.show()

#%%

# line graph reveals the count of each occupation.

occupation_counts = df_input['occupation'].value_counts().reset_index()
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

# The average charges are significantly higher for individuals with white-collar occupations, albeit with a 
# lower count, followed by those in blue-collar occupations. Conversely, the unemployed pay lower charges, 
# and this category represents a larger count within the dataset.

#%%

# One way ANOVA test between charges and occupation.

# Boxplot charges vs occupation
g1 = sns.boxplot(data = df_input, y = 'charges', x = 'occupation', palette='Set3', showmeans = True, meanprops = {"marker": "o", "markerfacecolor":"red", "markeredgecolor":"black", "markersize":"6"}, showfliers = False)
g1.set(title ='Boxplot charges vs occupation')
plt.xlabel('occupation')
plt.ylabel('charges') 
plt.grid()
plt.show()

#%%

model_OC= ols('charges ~ C(occupation)',data=df_input).fit()
result_OC = sm.stats.anova_lm(model_OC, type=1)
  
# Print the result
print(result_OC, "\n")

print(" There is significant difference among the means of occupation groups of charges")

tukey_OC = pairwise_tukeyhsd(endog=df_input['charges'], groups=df_input['occupation'], alpha=0.05)
print(tukey_OC)

# The Tukey HSD test further confirms that all pairwise differences are statistically significant.
# Least difference : The mean charges for Group 1 (student) are, on average, $495.5484 less than the mean charges for Group 2 (unemployed)

#%%[markdown]
# ### Analysis:
# Based on both the graph visualization and the results of the one-way ANOVA test, it is 
# evident that substantial variations exist among the mean values of charges across 
# different Occupation. The Tukey HSD test further confirms that all pairwise differences are statistically significant.
# And also from the graphs, it is evident that whitecollar occupation people is contributing more towards charges than other
# occupations though they are in less count.

#%%

##### Charges according to exercise frequency and medical history #####

# bar plot of charges among exercise freq and medical history together
sns.set(style="whitegrid")
# Create a grouped bar chart
plt.figure(figsize=(12, 8))
sns.barplot(data=df_input, x='exercise_frequency', y='charges', hue='medical_history', palette='muted')

# Set plot labels and title
plt.xlabel('Exercise Frequency')
plt.ylabel('Charges')
plt.title('Exercise Frequency vs. Medical History vs. Charges')
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

##### Charges according to exercise frequency and smoking #####

sns.set(style="whitegrid")
# Create a grouped bar chart
plt.figure(figsize=(12, 8))
sns.barplot(data=df_input, x='exercise_frequency', y='charges', hue='smoker', palette='PRGn')

# Set plot labels and title
plt.xlabel('Exercise Frequency')
plt.ylabel('Charges')
plt.title('Exercise Frequency vs. Smokers vs. Charges')
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

##### Charges vs Exercise Frequency #####

# One way ANOVA test between charges and exercise freq.

#Boxplot charges vs exercise_frequency

g1 = sns.boxplot(data = df_input, y = 'charges', x = 'exercise_frequency', palette='Set3', showmeans = True, meanprops = {"marker": "o", "markerfacecolor":"red", "markeredgecolor":"black", "markersize":"6"}, showfliers = False)
g1.set(xticklabels= charges)
g1.set(title ='Boxplot charges vs exercise_frequency')
plt.xlabel('exercise_frequency')
plt.ylabel('charges') 
plt.grid()
plt.show()

#%%

model_EF= ols('charges ~ C(exercise_frequency)',data=df_input).fit()
result_EF = sm.stats.anova_lm(model_EF, type=1)
  
# Print the result
print(result_EF, "\n")

print(" There is significant difference among the means of occupation groups of charges")

tukey_EF = pairwise_tukeyhsd(endog=df_input['charges'], groups=df_input['exercise_frequency'], alpha=0.05)
print(tukey_EF)

# The Tukey HSD test further confirms that all pairwise differences are statistically significant.
# Least difference : The mean charges for Group 1 (never) are, on average, $481.7174 more than the mean charges for Group 3 (rarely)

#%%

# Chi sq test to check relation of smoker with exercise frequency

contingency_table = pd.crosstab(df_input['smoker'], df_input['exercise_frequency'])

chi2, p, dof, expected = chi2_contingency(contingency_table)

print("Chi-square test for independence between smoker and exercise_frequency:")
print(f"Chi2 value: {chi2}")
print(f"P-value: {p}")

'''
There is significant assosciation between smoker and exercise_frequency of an individual
'''

#%%

### Coverage level vs Charges:

# Box plot of charges among coverage level.

plt.figure(figsize=(10, 6))
# Create a box plot between charges and coverage_level.
df_input.boxplot(column='charges', by='coverage_level', vert=False, flierprops={'marker': 'o', 'markerfacecolor': 'red', 'markersize': 5})
plt.suptitle('')  # Remove the default title
plt.xticks(rotation=45, ha='right') 
plt.title('Box Plot of Charges by coverage_level')
plt.xlabel('Charges')
plt.ylabel('coverage_level')
plt.tight_layout()
plt.show()

# The chart exactly reveals that Premium customers incur the highest charges, 
# trailed by Standard and Basic customers in descending order.

#%%

# ANOVA test of charges vs coverage level

model3 = sm.OLS.from_formula('charges ~ C(coverage_level)', data=df_input).fit()
print(model3.summary())

# from EDA and inference from model, we could see coverage level = 1 (Premium) impacts more on charges

##################################################
#<<<<<<<<<<<<<<<< End of Section >>>>>>>>>>>>>>>>#

#%%

#################
## Correlation ##
#################

# dist plot of charges
sns.distplot(df_input['charges'])

#%%

# correlation matrix
sns.set(rc={'figure.figsize':(14,8)})
correlation_matrix = df_label.corr()
#print(correlation_matrix)
sns.heatmap(correlation_matrix, annot=True, cmap='YlGnBu')
plt.show()

##################################################
#<<<<<<<<<<<<<<<< End of Section >>>>>>>>>>>>>>>>#

#%%

###################################################################
## Modeling without preprocessing ( scaling or one hot encoding) ##
###################################################################


#%%

#####################
## Split Data ##
#####################

X = df_label.drop('charges', axis=1)
y = df_label['charges']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("X_Train set shape: ", X_train.shape)
print("y_Train set shape: ", y_train.shape)
print("X_test set shape: ", X_test.shape)
print("y_test set shape: ", y_test.shape)

#%%
#####################
# Feature Selection 
#####################

### mutual_info_regression feature selection

# Calculate scores
importance = mutual_info_regression(X,y)
# plotting ranks
feat_importances = pd.Series(importance, df_label.columns[0:len(df_label.columns)-1])
feat_importances.plot(kind='barh', color ='teal')
plt.show()

# Top features influencing charges
# Smoker
# covergae_level
# medical_history
# family_medical_history

#%%

#####################
## Model Selection ##
#####################

########## Linear Regression Model ##########

lrmodel = LinearRegression()
lrmodel.fit(X_train, y_train)

y_pred_train = lrmodel.predict(X_train)
y_pred_test = lrmodel.predict(X_test)

#%%
# Evaluate the model
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)
errors = abs(y_pred_test - y_test)
mape = 100 * np.mean((errors / y_test))
accuracy = 100 - mape

print('Metrics for Linear regression model :')
print("Mean squared error: ", mean_squared_error(y_test, y_pred_test))
print("Mean absolute error: ", mean_absolute_error(y_test, y_pred_test))
print('Mean absolute percentage error (MAPE):', mape)
print('Accuracy:', round(accuracy, 2), '%.')

print(f'R-squared on the training set: {r2_train}')
print(f'R-squared on the test set: {r2_test}')

# Results:
# Metrics for Linear regression model :
# Mean squared error:  9157484.046080228
# Mean absolute error:  2472.4665104091882
# Mean absolute percentage error (MAPE): 15.890511144891947
# Accuracy: 84.11 %.
# R-squared on the training set: 0.5290890284527859
# R-squared on the test set: 0.5294375997176282

#%%
#### Interpretation of result:

#LR Results Plot
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
########## Support Vector Regression model ##########

svr = SVR(kernel='linear',C=3) # params after tuning through different kernels and C values.

svr.fit(X_train[:200000], y_train[:200000])
y_pred_train = svr.predict(X_train)
y_pred_test = svr.predict(X_test)

#%%
# Evaluate the model
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)
errors = abs(y_pred_test - y_test)
mape = 100 * np.mean((errors / y_test))
accuracy = 100 - mape

print('Metrics for SVM model :')
print("Mean squared error: ", mean_squared_error(y_test, y_pred_test))
print("Mean absolute error: ", mean_absolute_error(y_test, y_pred_test))
print('Mean absolute percentage error (MAPE):', mape)
print('Accuracy:', round(accuracy, 2), '%.')

print(f'R-squared on the training set: {r2_train}')
print(f'R-squared on the test set: {r2_test}')

# Results:
# Metrics for SVM model :
# Mean squared error:  9194809.948672732
# Mean absolute error:  2470.9781478290224
# Mean absolute percentage error (MAPE): 15.777191941162611
# Accuracy: 84.22 %.
# R-squared on the training set: 0.5272616504927936
# R-squared on the test set: 0.5275195874963398

# %%
########## Random Forest Regression model ##########

rfm = RandomForestRegressor(n_estimators=20, random_state=42) # params after tuning for n_estimators
rfm.fit(X_train, y_train)

y_pred_train = rfm.predict(X_train)
y_pred_test = rfm.predict(X_test)

#%%
# Evaluate the model
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)
errors = abs(y_pred_test - y_test)
mape = 100 * np.mean((errors / y_test))
accuracy = 100 - mape

print('Metrics for Random Forest model :')
print("Mean squared error: ", mean_squared_error(y_test, y_pred_test))
print("Mean absolute error: ", mean_absolute_error(y_test, y_pred_test))
print('Mean absolute percentage error (MAPE):', mape)
print('Accuracy:', round(accuracy, 2), '%.')

print(f'R-squared on the training set: {r2_train}')
print(f'R-squared on the test set: {r2_test}')

# Results :
# Metrics for Random Forest model :
# Mean squared error:  137789.78264207367
# Mean absolute error:  304.1638156085461
# Mean absolute percentage error (MAPE): 1.9674950418818578
# Accuracy: 98.03 %.
# R-squared on the training set: 0.998870404007834
# R-squared on the test set: 0.9929195955430363

#%%

#### Interpretation of result:
plt.figure(figsize= (10, 10))
plt.scatter (y_test, y_pred_test, color = 'red', label='Comparison of Prediction between Actual & Prediction data')
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

#%%

GBR1 = GradientBoostingRegressor()
GBR1.fit(X_train,y_train)
y_pred = GBR1.predict(X_test)
y_pred_train1=GBR1.predict(X_train)

# Evaluate the model
score = np.mean(cross_val_score(GBR1, X, y, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1))
baseline_errors = abs(y_pred - y_test)
baseline_mape = 100 * np.mean((baseline_errors / y_test))
baseline_accuracy = 100 - baseline_mape

print('Metrics for Gradient boosting model for baseline')
print("cross_val_score: ", round(score, 2))
print("Mean squared error: ", mean_squared_error(y_test, y_pred))
print("Mean absolute error: ", mean_absolute_error(y_test, y_pred))
print('Mean absolute percentage error (MAPE):', baseline_mape)
print('Accuracy:', round(baseline_accuracy, 2), '%.')

print("R-squared on the training set", round(r2_score(y_train,y_pred_train1),4))
print("R-squared on the testing set", round(r2_score(y_test,y_pred),4))

# Results:
# Metrics for Gradient boosting model for baseline
# cross_val_score:  -354295.11
# Mean squared error:  356467.4813048479
# Mean absolute error:  476.5097493527137
# Mean absolute percentage error (MAPE): 3.0855832369975253
# Accuracy: 96.91 %.
# R-squared on the training set 0.9818
# R-squared on the testing set 0.9817

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

# Evaluate the model
score = np.mean(cross_val_score(GBR2, X, y, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1))
errors = abs(y_pred - y_test)
mape = 100 * np.mean((errors / y_test))
accuracy = 100 - mape

print('Metrics for Gradient boosting model after tuning')
print("cross_val_score: ", round(score, 2))
print("Mean squared error: ", mean_squared_error(y_test, y_pred))
print("Mean absolute error: ", mean_absolute_error(y_test, y_pred))
print('Average absolute error:', round(np.mean(errors), 2))

print('mean absolute percentage error (MAPE):', mape)
print('Accuracy:', round(accuracy, 2), '%.')

print("R-squared on the training set", round(r2_score(y_train,y_pred_train2),4))
print("R-squared on the testing set", round(r2_score(y_test,y_pred),4))

# Results:
# Metrics for Gradient boosting model after tuning
# cross_val_score: -122656.38
# Mean squared error:  126511.28304750347
# Mean absolute error:  292.5691297341194
# Average absolute error: 292.57
# mean absolute percentage error (MAPE): 1.899894828242299
# Accuracy: 98.10 %.
# R-squared on the training set 0.9941
# R-squared on the testing set 0.9935

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

# plotting the variable importance from gradient boosting model
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

# Evaluate the model
score = np.mean(cross_val_score(XG1, X, y, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1))
baseline_errors = abs(y_pred - y_test)
baseline_mape = 100 * np.mean((baseline_errors / y_test))
baseline_accuracy = 100 - baseline_mape

print('Metrics for XGBoost model for baseline')
print("cross_val_score: ", round(score, 2))
print("Mean squared error: ", mean_squared_error(y_test, y_pred))
print("Mean absolute error: ", mean_absolute_error(y_test, y_pred))
print('Mean absolute percentage error (MAPE):', baseline_mape)
print('Accuracy:', round(baseline_accuracy, 2), '%.')

print("R-squared on the training set", round(r2_score(y_train,y_pred_train3),4))
print("R-squared on the testing set", round(r2_score(y_test,y_pred),4))

# Results:
# Metrics for XGBoost model for baseline
# cross_val_score:  -119241.14
# Mean squared error:  120503.16185165636
# Mean absolute error:  286.9262671867786
# Mean absolute percentage error (MAPE): 1.8464141812974073
# Accuracy: 98.15 %.
# R-squared on the training set 0.9939
# R-squared on the testing set 0.9938

#%%
#### Hyperparameter tuning :

# Running these codes for parameter tuning takes longer time:

# XG = XGBRegressor()
# search_grid = {'n_estimators': [500,1000,2000], 'gamma':[0,0.15,0.3,0.5,1], 'max_depth': [3,4,5,9], 'random_state': [1]}
# search = GridSearchCV(estimator=XG, param_grid=search_grid, scoring='neg_mean_squared_error', n_jobs=-1, cv=cv)
# search.fit(X_train,y_train)
# print(search.best_params_)
# print(search.best_score_)
# print(search.best_estimator_)

# Results:
# XGBRegressor(n_estimators=200, gamma=0, max_depth=4, random_state=1)

#%%

#### modeling XG boosting after tuning:

XG2 = XGBRegressor(n_estimators=2000, gamma=0.5, max_depth=4, random_state=1)
XG2.fit(X_train,y_train)
y_pred = XG2.predict(X_test)
y_pred_train4=XG2.predict(X_train)

# Evaluate the model
score = np.mean(cross_val_score(XG2, X, y, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1))
errors = abs(y_pred - y_test)
mape = 100 * np.mean((errors / y_test))
accuracy = 100 - mape

print('Metrics for XG Boost model after tuning')
print("cross_val_score: ", round(score, 2))
print("Mean squared error: ", mean_squared_error(y_test, y_pred))
print("Mean absolute error: ", mean_absolute_error(y_test, y_pred))
print('Average absolute error:', round(np.mean(errors), 2))

print('mean absolute percentage error (MAPE):', mape)
print('Accuracy:', round(accuracy, 2), '%.')

print("R-squared on the training set", round(r2_score(y_train,y_pred_train4),4))
print("R-squared on the testing set", round(r2_score(y_test,y_pred),4))

# Results:
# Metrics for XG Boost model after tuning
# cross_val_score:  -85530.39
# Mean squared error:  85780.12506339431
# Mean absolute error:  252.74865700371492
# Average absolute error: 252.75
# mean absolute percentage error (MAPE): 1.634090850498782
# Accuracy: 98.37 %.
# R-squared on the training set 0.9957
# R-squared on the testing set 0.9956

##############
# Analysis ##
#############

# SVR, Random Forest, Gradient Boosting and XG Boost outperforms Linear Regression.
# Compared to other models, XG Boost show the lowest MSE and high accuracy, suggesting better overall performance
# when scaling for numerical and one hot encoding for categorical variables are NOT done.

##################################################
#<<<<<<<<<<<<<<<< End of Section >>>>>>>>>>>>>>>>#

#%%

###################################################################
## Modeling with preprocessing ( scaling and one hot encoding) ##
###################################################################


#%%

#####################
## Split Data ##
#####################

X = df_encoded.drop(['charges'], axis=1)
y = df_encoded['charges']

#%%

# scaling numerical variables.
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

########## Support Vector Regression model ##########

svr = SVR(kernel='linear',C=3)

svr.fit(X_train[:200000], y_train[:200000])
y_pred_train = svr.predict(X_train)
y_pred_test = svr.predict(X_test)

#%%
# Evaluate the model
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)
errors = abs(y_pred_test - y_test)
mape = 100 * np.mean((errors / y_test))
accuracy = 100 - mape

print('Metrics for SVM model :')
print("Mean squared error: ", mean_squared_error(y_test, y_pred_test))
print("Mean absolute error: ", mean_absolute_error(y_test, y_pred_test))
print('Mean absolute percentage error (MAPE):', mape)
print('Accuracy:', round(accuracy, 2), '%.')

print(f'R-squared on the training set: {r2_train}')
print(f'R-squared on the test set: {r2_test}')

# Results:
# Metrics for SVM model :
# Mean squared error:  83623.51013089206
# Average absolute error: 250.49
# mean absolute percentage error (MAPE): 1.6204962885585796
# Accuracy: 98.38 %.
# Mean Absolute Error: 250.48841222402845
# R-squared: 0.9957029595193154

#%%

########## Linear Regression Model ##########

X_num = df_input[['age','gender', 'bmi','smoker', 'medical_history', 'family_medical_history', 'occupation','coverage_level']]

X_encoded = pd.get_dummies(X_num, columns=['gender','smoker', 'medical_history', 'family_medical_history', 'occupation','coverage_level'], drop_first=True)
from sklearn.preprocessing import StandardScaler, LabelEncoder
scaler = StandardScaler()
X_encoded[['age', 'bmi']] = scaler.fit_transform(X_encoded[['age', 'bmi']])

xtrain, xtest, ytrain, ytest = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

lrmodel = LinearRegression()
lrmodel.fit(xtrain, ytrain)

y_pred_train = lrmodel.predict(xtrain)
y_pred_test = lrmodel.predict(xtest)

#%%
# Evaluate the model
score = np.mean(cross_val_score(lrmodel, X_encoded, y, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1))
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)
errors = abs(y_pred_test - y_test)
mape = 100 * np.mean((errors / y_test))
accuracy = 100 - mape

print('Metrics for Linear regression model :')
print("Mean squared error: ", mean_squared_error(y_test, y_pred_test))
print("Mean absolute error: ", mean_absolute_error(y_test, y_pred_test))
print('Mean absolute percentage error (MAPE):', mape)
print('Accuracy:', round(accuracy, 2), '%.')

print(f'R-squared on the training set: {r2_train}')
print(f'R-squared on the test set: {r2_test}')

# Results
# Metrics for Linear regression model :
# Mean squared error:  839240.2361864425
# Mean absolute error:  753.3913842492952
# Mean absolute percentage error (MAPE): 4.853282984470397
# Accuracy: 95.15 %.
# R-squared on the training set: 0.9568890752339336
# R-squared on the test set: 0.956875174669567

#%%
#### Interpretation of Data:

#LR Results Plot
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
########## Random Forest Regression model ##########

rfm = RandomForestRegressor(n_estimators=20, random_state=42)
rfm.fit(X_train, y_train)

y_pred_train = rfm.predict(X_train)
y_pred_test = rfm.predict(X_test)

#%%
# Evaluate the model
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)
errors = abs(y_pred_test - y_test)
mape = 100 * np.mean((errors / y_test))
accuracy = 100 - mape

print('Metrics for Random Forest model :')
print("Mean squared error: ", mean_squared_error(y_test, y_pred_test))
print("Mean absolute error: ", mean_absolute_error(y_test, y_pred_test))
print('Mean absolute percentage error (MAPE):', mape)
print('Accuracy:', round(accuracy, 2), '%.')

print(f'R-squared on the training set: {r2_train}')
print(f'R-squared on the test set: {r2_test}')

#Result:
# Metrics for Random Forest model :
# Mean squared error:  130913.45540693443
# Mean absolute error:  297.5832270437939
# Mean absolute percentage error (MAPE): 1.9236481940773142
# Accuracy: 98.08 %.
# R-squared on the training set: 0.9989223609421376
# R-squared on the test set: 0.9932729394345039

# %%

#### Interpretation of result
plt.figure(figsize= (10, 10))
plt.scatter (y_test, y_pred_test, color = 'red', label='Comparison of Prediction between Actual & Prediction data')
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

#%%

GBR1 = GradientBoostingRegressor()
GBR1.fit(X_train,y_train)
y_pred = GBR1.predict(X_test)
y_pred_train1=GBR1.predict(X_train)

# Evaluate the model
score = np.mean(cross_val_score(GBR1, X, y, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1))
baseline_errors = abs(y_pred - y_test)
baseline_mape = 100 * np.mean((baseline_errors / y_test))
baseline_accuracy = 100 - baseline_mape

print('Metrics for Gradient boosting model for baseline')
print("cross_val_score: ", round(score, 2))
print("Mean squared error: ", mean_squared_error(y_test, y_pred))
print("Mean absolute error: ", mean_absolute_error(y_test, y_pred))
print('Mean absolute percentage error (MAPE):', baseline_mape)
print('Accuracy:', round(baseline_accuracy, 2), '%.')

print("R-squared on the training set", round(r2_score(y_train,y_pred_train1),4))
print("R-squared on the testing set", round(r2_score(y_test,y_pred),4))

# Results: 
# Metrics for Gradient boosting model for baseline
# cross_val_score:  -242295.93
# Mean squared error:  240289.17438117406
# Mean absolute error:  394.81148111206727
# Mean absolute percentage error (MAPE): 2.6250087976298646
# Accuracy: 97.37 %.
# R-squared on the training set 0.9878
# R-squared on the testing set 0.9877

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

GBR2 = GradientBoostingRegressor(n_estimators=200, learning_rate=0.6, subsample= 0.9, max_depth=10, random_state=1)
GBR2.fit(X_train,y_train)
y_pred = GBR2.predict(X_test)
y_pred_train2=GBR2.predict(X_train)

# Evaluate the model
score = np.mean(cross_val_score(GBR2, X, y, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1))
errors = abs(y_pred - y_test)
mape = 100 * np.mean((errors / y_test))
accuracy = 100 - mape

print('Metrics for Gradient boosting model after tuning')
print("cross_val_score: ", round(score, 2))
print("Mean squared error: ", mean_squared_error(y_test, y_pred))
print("Mean absolute error: ", mean_absolute_error(y_test, y_pred))
print('Average absolute error:', round(np.mean(errors), 2))

print('mean absolute percentage error (MAPE):', mape)
print('Accuracy:', round(accuracy, 2), '%.')

print("R-squared on the training set", round(r2_score(y_train,y_pred_train2),4))
print("R-squared on the testing set", round(r2_score(y_test,y_pred),4))

# Results: 
# Metrics for Gradient boosting model after tuning
# cross_val_score:  -131033.32
# Mean squared error:  133939.26538929652
# Mean absolute error:  299.7717524620667
# Average absolute error: 299.77
# mean absolute percentage error (MAPE): 1.927994322062208
# Accuracy: 98.07 %.
# R-squared on the training set 0.9969
# R-squared on the testing set 0.9931

#%%

########## XG Boost Regression model ##########

#### Baseline model:

XG1 = XGBRegressor()
XG1.fit(X_train,y_train)
y_pred = XG1.predict(X_test)
y_pred_train3 = XG1.predict(X_train)

# Evaluate the model
score = np.mean(cross_val_score(XG1, X, y, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1))
baseline_errors = abs(y_pred - y_test)
baseline_mape = 100 * np.mean((baseline_errors / y_test))
baseline_accuracy = 100 - baseline_mape

print('Metrics for XGBoost model for baseline')
print("cross_val_score: ", round(score, 2))
print("Mean squared error: ", mean_squared_error(y_test, y_pred))
print("Mean absolute error: ", mean_absolute_error(y_test, y_pred))
print('Mean absolute percentage error (MAPE):', baseline_mape)
print('Accuracy:', round(baseline_accuracy, 2), '%.')

print("R-squared on the training set", round(r2_score(y_train,y_pred_train3),4))
print("R-squared on the testing set", round(r2_score(y_test,y_pred),4))

# Results:
# Metrics for XGBoost model for baseline
# cross_val_score:  -108625.43
# Mean squared error:  107719.9956771001
# Mean absolute error:  274.5330017235559
# Mean absolute percentage error (MAPE): 1.7731064820493123
# Accuracy: 98.23 %.
# R-squared on the training set 0.9946
# R-squared on the testing set 0.9945

# NMSE - lower, AAE - lower, MAPE - lower, R2 - higher.

#%%
#### Hyperparameter tuning :

# Running these codes for parameter tuning takes longer time:

# XG = XGBRegressor()
# search_grid = {'n_estimators': [500,1000,2000], 'gamma':[0,0.15,0.3,0.5,1], 'max_depth': [3,4,5,9], 'random_state': [1]}
# search = GridSearchCV(estimator=XG, param_grid=search_grid, scoring='neg_mean_squared_error', n_jobs=-1, cv=cv)
# search.fit(X_train,y_train)
# print(search.best_params_)
# print(search.best_score_)
# print(search.best_estimator_)

# Results:
# XGBRegressor(n_estimators=200, gamma=0, max_depth=4, random_state=1)

#%%

#### modeling gradient boosting after tuning:

XG2 = XGBRegressor(n_estimators=2000, gamma=0.5, max_depth=4, random_state=1)
XG2.fit(X_train,y_train)
y_pred = XG2.predict(X_test)
y_pred_train4=XG2.predict(X_train)

# Evaluate the model
score = np.mean(cross_val_score(XG2, X, y, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1))
errors = abs(y_pred - y_test)
mape = 100 * np.mean((errors / y_test))
accuracy = 100 - mape

print('Metrics for XG Boost model after tuning')
print("cross_val_score: ", round(score, 2))
print("Mean squared error: ", mean_squared_error(y_test, y_pred))
print("Mean absolute error: ", mean_absolute_error(y_test, y_pred))
print('Average absolute error:', round(np.mean(errors), 2))

print('mean absolute percentage error (MAPE):', mape)
print('Accuracy:', round(accuracy, 2), '%.')

print("R-squared on the training set", round(r2_score(y_train,y_pred_train4),4))
print("R-squared on the testing set", round(r2_score(y_test,y_pred),4))

# Results:  
# Metrics for XG Boost model after tuning
# cross_val_score:  -85213.3
# Mean squared error:  85635.9558945334
# Mean absolute error:  252.49568506275912
# Average absolute error: 252.5
# mean absolute percentage error (MAPE): 1.6322867577811113
# Accuracy: 98.37 %.
# R-squared on the training set 0.9958
# R-squared on the testing set 0.9956

#%%

##############
# Analysis ##
#############

# SVR, Random Forest, Gradient Boosting and XG Boost outperforms Linear Regression.
# Compared to other models, XG Boost show the lowest MSE and high accuracy, suggesting better overall performance
# when scaling for numerical and one hot encoding for categorical variables are NOT done.


##################################################
#<<<<<<<<<<<<<<<< End of Section >>>>>>>>>>>>>>>>#

#%%

# Scaling and one hot encoding
models = ['Linear', 'SVR', 'Random Forest', 'Gradient Boosting', 'XG Boost']
mse_values = [839240.23, 83623.51, 130913.45, 133939.26, 85635.95]

# Plotting the Mean Squared Error
plt.figure(figsize=(10, 6))
plt.plot(models, mse_values, marker='o', label='M.S.E')
plt.title('Mean Squared Error (MSE) for Different Regression Models after scaling and one hot encoding')
plt.xlabel('Regression Models')
plt.ylabel('Mean Squared Error (MSE)')
plt.legend()
plt.grid(True)
plt.show()

#%%

# no scaling or one hot encoding - only did label encoding
models = ['Linear', 'SVR', 'Random Forest','Gradient Boosting', 'XG Boost']
mse_values = [9157484.04, 9194809.95, 137789.78, 126511.28, 85780.12]

# Plotting the Mean Squared Error
plt.figure(figsize=(10, 6))
plt.plot(models, mse_values, marker='o', label='M.S.E')
plt.title('Mean Squared Error (MSE) for Different Regression Models without scaling and one hot encoding')
plt.xlabel('Regression Models')
plt.ylabel('Mean Squared Error (MSE)')
plt.legend()
plt.grid(True)
plt.show()

#%%

######################
# Model Evaluation ###
######################

# Scaling and one-hot encoding plays a crucial role in enhancing the performance of linear regression and Support Vector Regression (SVR).
# Overall, tree models consistently outperformed in both scenarios with and without preprocessing steps.
# But comparing the MSE for all the models in both the scenarios, SVR is giving less MSE value and also a balance between bias and variance
# after preprocessing.

###################################################################################

# %%
