import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.metrics import accuracy_score

# Setting seaborn visualization style
sns.set()

# Importing test and train datasets
df_train = pd.read_csv('../Titanic/data/train.csv')
df_test = pd.read_csv('../Titanic/data/test.csv')

# Caveat: I'm using Atom IDE and Hydrogen for inline visualization
# View the first line of train dataset
df_train.head() #default 5

# View the first line of test dataset
df_test.head()

# Looking more accurately at df_train's data
# View the last line of df_train
df_train.tail()

# .info prints the summary of a dataframe
df_train.info()

# .describe prints the summary statistics of numeric columns of the dataframe
df_train.describe()

# This is for having an idea about what data looks like, so now we can start doing EDA


'''
Exploratory Data Analysis (EDA)
'''

# Using univariate visualization method to see our target variable
sns.countplot(x='Survived', data=df_train)

# Creating a bad Predictive model for baseline
df_test['Survived'] = 0
df_test.head()
df_test[['PassengerId', 'Survived']].to_csv('data/predictions/no_survivors.csv', index=False)

# Using univariate visualization method to see gender variable
sns.countplot(x='Sex', data=df_train)

# Using bivariate visualization method to check if there is a
# relationship between gender and target variable
sns.catplot(x='Survived', col='Sex', kind='count', data=df_train)
# Looks like women were more likely to survive than man

# Checking more accurately how many men and women Survived
df_train.groupby(['Sex']).Survived.sum()
# Calculating the proportion of survivors for each gender
print(df_train[df_train.Sex == 'female'].Survived.sum()/df_train[df_train.Sex == 'female'].Survived.count())
print(df_train[df_train.Sex == 'male'].Survived.sum()/df_train[df_train.Sex == 'male'].Survived.count())
# 74.2% of women survived, while only 18.9% of men survived

# Building second shitty model where all women survived and man didn't
# Again, just for baseline for future models
df_test['Survived'] = df_test.Sex == 'female' #Creates 'Survived' column with True for females and False for males
df_test['Survived'] = df_test.Survived.apply(lambda x: int(x)) #Changes Bool True/False for 1/0
df_test.head()
df_test[['PassengerId', 'Survived']].to_csv('data/predictions/women_survived.csv', index=False)
