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

# Using bivariate visualization method to check if there is a
# relationship between other variables and target variable
sns.catplot(x='Survived', col='Pclass', kind='count', data=df_train)
# It seems that first class passengers are more likely to survive and third class are less likely

sns.catplot(x='Survived', col='Embarked', kind='count', data=df_train)
# It seems that passengers that embarked at southampton are more unlikely to survive

# EDA with numerical variables
# Using univariate distribution with histograms
sns.distplot(df_train.Fare, kde=False)
# Most passengers payed less than $100
df_train.groupby('Survived').Fare.hist(alpha=0.5)
# In red we see the survivors
# so it seems that passengers who payed more had higher chance of surviving

# As we have missing values in Age column, we first have to clean that up
df_train_drop = df_train.dropna()
sns.distplot(df_train_drop.Age, kde=False)

# Visualizing the age distribution of survivors and non survivors
df_train_drop.groupby('Survived').Age.hist(alpha=0.5)
# In red we see the survivors
# so it seems that passengers who were between 18-40 had higher chance of surviving
