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
df_test[['PassengerId', 'Survived']].to_csv('data/predictions/no_survivors.csv', index=False)

# Checking created no_survivors.csv
df_badmodel = pd.read_csv('data/predictions/no_survivors.csv')
df_badmodel.head()
