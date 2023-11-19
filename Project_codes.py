# To add a new cell, type ''
# To add a new markdown cell, type '#%%[markdown]'

#%%[markdown]
#
# # DM Project
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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
#import rfit 

#%%

df_input = pd.read_csv("insurance_dataset.csv")
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



#%%

#Analysis 
####### Charges Analysis 

plt.hist(df.charges, bins = 20, alpha=0.5, edgecolor='black', color="green",linewidth=1)
plt.xlabel('Charges')
plt.ylabel('Density')
plt.title('Charges Distribution')
plt.legend()
plt.tight_layout()
plt.show()

# 14000 to 20000 more density charges

#%%

####### Charges vs Region:

df['region'].value_counts()
# northeast have max people and south east have minimum
# ANOVA test to check the charges mean between region.

plt.figure(figsize=(10, 6))

# Create a box plot between income and ethnic.
df.boxplot(column='charges', by='region', vert=False, flierprops={'marker': 'o', 'markerfacecolor': 'red', 'markersize': 5})
plt.suptitle('')  # Remove the default title
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.title('Box Plot of Charges by Region')
plt.xlabel('Charges')
plt.ylabel('Region')
plt.tight_layout()
plt.show()

# Using ANOVA test to find mean income between ethnic.
from bioinfokit.analys import stat
res = stat()
res.anova_stat(df=df, res_var='charges', anova_model='charges ~ C(region)')
alpha = 0.05
print("ANOVA test Result: ")
print(res.anova_summary)
print("\n p value < alpha :  There is significant difference in charges between region")

# perform multiple pairwise comparison (Tukey's HSD)
# unequal sample size data, tukey_hsd uses Tukey-Kramer test

# res = stat()
# res.tukey_hsd(df=df, res_var='charges', xfac_var='region', anova_model='charges ~ C(region)')
# print("\nTukey's HSD Test")
# print(res.tukey_summary)


#%%

###### Charges vs Gender:

df['gender'].value_counts()
# male      500107
# female    499893
# Average charges for each gender
df[df['gender'] == 'male']['charges'].mean()  # 17236.318028914015
df[df['gender'] == 'female']['charges'].mean() # 16233.702372522359
# Average of male charges is more compared to female. And male pay the max charge in data, female pay the min charges amount.
charges = df['charges']
charges_male = charges[df['gender']=='male']
charges_female = charges[df['gender']=='female']

# Create kde plot for male and female.
plt.style.use('seaborn-v0_8-deep')
sns.kdeplot(charges_female, label = 'female', color = "m", alpha = 0.5)
sns.kdeplot(charges_male, label = 'male', color = "y", alpha = 0.5)

plt.xlabel('Charges')
plt.ylabel('Density')
plt.title('Charges Distribution for Female and Male')
plt.legend()
plt.tight_layout()
plt.show()

# T test between Income of male and female.
t_stat, p_value = stats.ttest_ind(charges_female, charges_male)

print(f"T test Result: \n P value: {p_value}")
alpha = 0.05
# Check the p-value
if p_value < alpha:  # You can choose your significance level (e.g., 0.05)
    print(" Reject the null hypothesis; There is a significant difference in income between gender")
else:
    print(" There is no significant difference in income between gender")

#%%

####### Charges vs coverage_level:

df['coverage_level'].value_counts()
# northeast have max people and south east have minimum
# ANOVA test to check the charges mean between region.

plt.figure(figsize=(10, 6))

# Create a box plot between income and ethnic.
df.boxplot(column='charges', by='coverage_level', vert=False, flierprops={'marker': 'o', 'markerfacecolor': 'red', 'markersize': 5})
plt.suptitle('')  # Remove the default title
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.title('Box Plot of Charges by coverage_level')
plt.xlabel('Charges')
plt.ylabel('coverage_level')
plt.tight_layout()
plt.show()

#%%
####### Charges vs Occupation

# Create box plot between marital and income.
sns.boxplot(x='occupation', y='charges', data=df, color='#f2f2f2')
plt.show()

from bioinfokit.analys import stat
res = stat()
res.anova_stat(df=df, res_var='charges', anova_model='charges ~ C(occupation)')
alpha = 0.05
print("ANOVA test Result: ")
print(res.anova_summary)
print("\n p value < alpha :  There is significant difference in charges between region")


#%%

####### Charges vs medical_history

medical = pd.pivot_table(df, values='charges', columns=['medical_history'], aggfunc='mean')
# Melt the pivot table to a long format
meltmedical = pd.melt(medical, var_name='medical_history', value_name='average_charges')

# Create a bar chart with 'hue' for gender
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")
sns.barplot(data=meltmedical, x='medical_history', y='average_charges')
plt.xlabel('medical_history')
plt.ylabel('Average Charges')
plt.title('Average Charges by Medical_history')
plt.tight_layout()
plt.show()

#%%

####### Charges vs family_medical_history

medical = pd.pivot_table(df, values='charges', columns=['family_medical_history'], aggfunc='mean')
# Melt the pivot table to a long format
meltmedical = pd.melt(medical, var_name='family_medical_history', value_name='average_charges')

# Create a bar chart with 'hue' for gender
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")
sns.barplot(data=meltmedical, x='family_medical_history', y='average_charges')
plt.xlabel('family_medical_history')
plt.ylabel('Average Charges')
plt.title('Average Charges by family_medical_history')
plt.tight_layout()
plt.show()

#%%

####### Charges vs exercise_frequency

medical = pd.pivot_table(df, values='charges', columns='exercise_frequency', aggfunc='mean')
# Melt the pivot table to a long format
meltmedical = pd.melt(medical, var_name='exercise_frequency', value_name='average_charges')

# Create a bar chart with 'hue' for gender
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")
sns.barplot(data=meltmedical, x='exercise_frequency', y='average_charges')
plt.xlabel('exercise_frequency')
plt.ylabel('Average Charges')
plt.title('Average Charges by exercise_frequency')
plt.tight_layout()
plt.show()

#%%

###### Charges vs smoker:

df['smoker'].value_counts()
# male      500107
# female    499893
# Average charges for each gender
df[df['smoker'] == 'yes']['charges'].mean()  # 17236.318028914015
df[df['smoker'] == 'no']['charges'].mean() # 16233.702372522359
# Average of male charges is more compared to female. And male pay the max charge in data, female pay the min charges amount.
charges = df['charges']
charges_smoker = charges[df['smoker']=='yes']
charges_nonsmoker = charges[df['smoker']=='no']

# Create kde plot for male and female.
plt.style.use('seaborn-v0_8-deep')
sns.kdeplot(charges_smoker, label = 'smoker', color = "m", alpha = 0.5)
sns.kdeplot(charges_nonsmoker, label = 'nonsmoker', color = "y", alpha = 0.5)

plt.xlabel('Charges')
plt.ylabel('Density')
plt.title('Charges Distribution for smoker and nonsmoker')
plt.legend()
plt.tight_layout()
plt.show()

# T test between Income of male and female.
t_stat, p_value = stats.ttest_ind(charges_smoker, charges_nonsmoker)

print(f"T test Result: \n P value: {p_value}")
alpha = 0.05
# Check the p-value
if p_value < alpha:  # You can choose your significance level (e.g., 0.05)
    print(" Reject the null hypothesis; There is a significant difference in income between gender")
else:
    print(" There is no significant difference in income between gender")

#%%

###### Charges vs bmi

plt.figure(figsize=(10, 6))

sns.scatterplot(x='bmi', y='charges', data=df, alpha=0.5)

plt.title('Scatter Plot between Charges and BMI')
plt.xlabel('BMI')
plt.ylabel('Charges')
plt.show()

bmi = pd.pivot_table(df, values='charges', columns='bmi', aggfunc='mean')
meltbmi = pd.melt(bmi, var_name='age', value_name='average_charges')

sns.scatterplot(x='bmi', y='average_charges', data=meltbmi, alpha=0.5)

plt.title('Scatter Plot between Charges and bmi')
plt.xlabel('bmi')
plt.ylabel('Charges')
plt.show()

#%%

###### Charges vs age

plt.figure(figsize=(10, 6))

sns.scatterplot(x='age', y='charges', data=df, alpha=0.5)

plt.title('Scatter Plot between Charges and age')
plt.xlabel('age')
plt.ylabel('Charges')
plt.show()

age = pd.pivot_table(df, values='charges', columns='age', aggfunc='mean')
meltage = pd.melt(age, var_name='age', value_name='average_charges')

sns.scatterplot(x='age', y='average_charges', data=meltage, alpha=0.5)

plt.title('Scatter Plot between Charges and age')
plt.xlabel('age')
plt.ylabel('Charges')
plt.show()

#%%

sns.pairplot(df[['age', 'gender', 'bmi', 'children', 'smoker', 'region', 'medical_history', 'family_medical_history', 'exercise_frequency', 'occupation', 'coverage_level', 'charges']])
plt.show()

#%%

correlation_matrix = df.corr()

# Display the correlation matrix
print(correlation_matrix)

#%%

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.show()

#%%

# checking numerical present variables with correlation with charges.
df1 = df.drop(['gender', 'smoker', 'region', 'medical_history', 'family_medical_history', 'exercise_frequency', 'occupation', 'coverage_level'], axis=1)
print(df1.dtypes)
correlation_matrix = df1.corr()
print(correlation_matrix)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.show()


#%%

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

scaler = StandardScaler()
df[['age', 'bmi', 'children']] = scaler.fit_transform(df[['age', 'bmi', 'children']])
X = df.drop(['gender', 'smoker', 'region', 'medical_history', 'family_medical_history', 'exercise_frequency', 'occupation', 'coverage_level', 'charges'], axis=1)
y = df['charges']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')


#%%


###################################################################################3

# %%
# Question 2
# Using the statsmodels package, 
# build a logistic regression model for survival. Include the features that you find plausible. 
# Make sure categorical variables are use properly. If the coefficient(s) turns out insignificant, drop it and re-build.
# 
# 

from statsmodels.formula.api import glm
import statsmodels.api as sm

survivedLogit = glm(formula='survived ~ age+C(pclass)+C(sex)+C(sibsp)+C(parch)+fare+C(embarked)', data=titanic, family=sm.families.Binomial())
survivedLogitfit = survivedLogit.fit()
print( survivedLogitfit.summary() )

# The model turns out having very high p-value for sibsp, parch, fare and embarked for almost all levels. 
# Thus, removing them for simpler model.

#%%
#  
# rebuilding:
survivedLogit = glm(formula='survived ~ age+C(pclass)+C(sex)', data=titanic, family=sm.families.Binomial())
survivedLogitfit = survivedLogit.fit()
print( survivedLogitfit.summary() )

#%% 
# Question 3
# Interpret your result. 
# What are the factors and how do they affect the chance of survival (or the survival odds ratio)? 
# What is the predicted probability of survival for a 30-year-old female with a second class ticket, 
# no siblings, 3 parents/children on the trip? Use whatever variables that are relevant in your model.
# 

# Intercept: The intercept is the log-odds of survival when all other predictor variables
#  are zero (2.855). odds ratio - exp(2.8255)
# exp(-0.9289) - odds ratio of second class. exp(-2.1722) - odds ratio of third class.
# exp(-2.6291) - odds ratio of male
# exp(-0.0161) - odds ratio of age
print(np.exp(survivedLogitfit.params)) # calculates exp of coeff

# logistic regression model in log-odds (logit) form:
# Logit(p) = 2.8255 + (-0.0161 * age) - 2.6291 (if male) - 0.9289 (if pclass 2) - 2.1722 (if pclass 3)

# logistic regression model in odds ratio form:
# p/(1-p) = 16.87 * 0.984062^age * 0.072142 (if male) * 0.394998 (if pclass 2) * 0.113926 (if pclass 3)

predict = survivedLogitfit.predict( {'age':30, 'pclass':2, 'sex':'female'}) # You can either put in a dataframe or dictionary with all the relevant values here to make a prediction.
print(f'\nThe model prediction of the survival probabilty is {(predict[0]*100).__round__(1)}%')

# 
#%%
# Question 4
# Now use the sklearn package, perform the same model and analysis as in Question 3. 
# In sklearn however, it is easy to set up the train-test split before we build the model. 
# Use 67-33 split to solve this problem. 
# Find out the accuracy score of the model.
# 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

X = titanic[['age','pclass','sex']]
y = titanic['survived']
X.head()
#X.dtypes
# converting object datatype to int
X.loc[X['sex'] == 'male', 'sex'] = 1
X.loc[X['sex'] == 'female', 'sex'] = 0

# splitting the dataset to train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy.round(2))
# 
# 
#%%
# Question 5
# Try three different cut-off values at 0.3, 0.5, and 0.7. What are the 
# a)	Total accuracy of the model
# b)	The precision of the model for 0 and for 1
# c)	The recall rate of the model for 0 and for 1
# 
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import classification_report

cutOffValues = [0.3, 0.5, 0.7]
for cutOff in cutOffValues:
    print(f"For cutoff = {cutOff}")
    y_pred_prob = model.predict(X_test)
    y_pred = [1 if prob >= cutOff else 0 for prob in y_pred_prob]
    print(classification_report(y_test, y_pred))

# 
#%% 
# Question 6
# By using cross-validation, re-do the logit regression, and evaluate 
# the 10-fold average accuracy of the logit model. 
# Use the same predictors you had from previous questions.
#
from numpy import mean
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

model = LogisticRegression()
kf = KFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
print('Average Accuracy: %.3f ' % (mean(scores)))
#

# %%
