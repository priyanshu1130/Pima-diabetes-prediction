
import numpy as np
import pandas as pd

import seaborn as sns
sns.set(color_codes=True)

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split


from sklearn import metrics


# In[3]:


# reading the CSV file into pandas dataframe

diabetes_data=pd.read_csv('diabetes.csv')


# In[4]:


# Check top few records to get a feel of the data structure

diabetes_data.head()


# 
# It shows that there are eight independent variables (Pregnancies , Glucose , BloodPressur , SkinThickness , Insulin , BMI , DiabetesPedigreeFunction , Age) and one dependent variable (Outcome).

# In[5]:


# To get the shape of the dataset

diabetes_data.shape


# In[6]:



# To show the detailed summary 

diabetes_data.info()


# In[7]:


#Lets analysze the distribution of the dependent column

diabetes_data.describe()


# In[8]:


# To check the missing values in the dataset
diabetes_data.isnull().values.sum()


# It shows that there are no null values (missing values) in the dataset. But, it does not make sense. It seems very likely that zero values encode missing data.We replace 0 by NaN values to count the missing values.

# In[9]:


#Replace 0 to NaN

diabetes_data[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']]=diabetes_data[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.nan)
diabetes_data.head()


# In[10]:


diabetes_data.isnull().sum()


# ## Handling the Missing values by replacing NaN to median

# In[11]:


#Replace NaN to mean value to explore dataset

diabetes_data['Glucose'].fillna(diabetes_data['Glucose'].median(),inplace=True)
diabetes_data['BloodPressure'].fillna(diabetes_data['BloodPressure'].median(),inplace=True)
diabetes_data['SkinThickness'].fillna(diabetes_data['SkinThickness'].median(),inplace=True)
diabetes_data['Insulin'].fillna(diabetes_data['Insulin'].median(),inplace=True)
diabetes_data['BMI'].fillna(diabetes_data['BMI'].median(),inplace=True)


# In[12]:


diabetes_data.head()


# # Exploratory Data Analysis

# In[13]:


diabetes_data.groupby('Outcome').size()


# In[14]:


# Distplot

fig, ax2 = plt.subplots(4, 2, figsize=(16, 16))
sns.distplot(diabetes_data['Pregnancies'],ax=ax2[0][0])
sns.distplot(diabetes_data['Glucose'],ax=ax2[0][1])
sns.distplot(diabetes_data['BloodPressure'],ax=ax2[1][0])
sns.distplot(diabetes_data['SkinThickness'],ax=ax2[1][1])
sns.distplot(diabetes_data['Insulin'],ax=ax2[2][0])
sns.distplot(diabetes_data['BMI'],ax=ax2[2][1])
sns.distplot(diabetes_data['DiabetesPedigreeFunction'],ax=ax2[3][0])
sns.distplot(diabetes_data['Age'],ax=ax2[3][1])


# The plots show that (Glucose, Blood_pressure , BMI,SkinThickness) are normally distributed.
# whiile (Preganancies, insulin, age, DiabetesPedigreeFunction) are rightly skewed. 

# In[15]:


fig, ax2 = plt.subplots(4, 2, figsize=(16, 16))
sns.barplot(diabetes_data['Outcome'],diabetes_data['Pregnancies'],ax=ax2[0][0])
sns.barplot(diabetes_data['Outcome'],diabetes_data['Glucose'],ax=ax2[0][1])
sns.barplot(diabetes_data['Outcome'],diabetes_data['BloodPressure'],ax=ax2[1][0])
sns.barplot(diabetes_data['Outcome'],diabetes_data['SkinThickness'],ax=ax2[1][1])
sns.barplot(diabetes_data['Outcome'],diabetes_data['Insulin'],ax=ax2[2][0])
sns.barplot(diabetes_data['Outcome'],diabetes_data['BMI'],ax=ax2[2][1])
sns.barplot(diabetes_data['Outcome'],diabetes_data['DiabetesPedigreeFunction'],ax=ax2[3][0])
sns.barplot(diabetes_data['Outcome'],diabetes_data['Age'],ax=ax2[3][1])


# ## Checking relation between features and checking for multicollinearity

# In[16]:


diabetes_data.corr()


# we can see observe some relatioin such as:
# Bloodpressure is depdendent on Age variable(0.324915) , Similarly, Glocose level and Skin thickness depends on age 
# we can also see significant relation between pregnancies and age (0.544341) but we have not considered it as case of multicollinearity because according to rule of thumb collinearity is expected for value > (0.70 or 0.80).

# In[17]:


corr=diabetes_data.corr()
sns.heatmap(corr)


# In[18]:


corr['Outcome'].sort_values(ascending=False)


# order of dependencies of independent features on dependent_feature

# ## Scaling the data

# we use StandardScaler
# AS there are many different parameters with different units of their measurements (eg cost,weight ) so we need to scale down each feature into single unit of variance
# StandardScaler is useful for the features that follow a Normal distribution.
# 
# In StandardScaler mean=0 and variance =1. This operation is performed feature-wise in an independent way.

# In[19]:


from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
X =  pd.DataFrame(ss.fit_transform(diabetes_data.drop(["Outcome"],axis = 1),),
        columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age'])


# In[21]:


x=X
y=diabetes_data['Outcome']


# ## Test Train Split

# In[29]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=.3,random_state=3,stratify=y)


# Using Logistic regression algorithm

# In[30]:


from sklearn.linear_model import LogisticRegression
LR_classifier=LogisticRegression(C=1,penalty='l2')
LR_classifier.fit(X_train,Y_train)

from sklearn.metrics import accuracy_score,confusion_matrix

print("Train Set Accuracy:"+str(accuracy_score(Y_train,LR_classifier.predict(X_train))*100))
print("Test Set Accuracy:"+str(accuracy_score(Y_test,LR_classifier.predict(X_test))*100))


# Using KNeighboursClassifier algorithm

# In[31]:


from sklearn.neighbors import KNeighborsClassifier
knn= KNeighborsClassifier(n_neighbors=9)
knn.fit(X_train,Y_train)

print("Train Set Accuracy:"+str(accuracy_score(Y_train,knn.predict(X_train))*100))
print("Test Set Accuracy:"+str(accuracy_score(Y_test,knn.predict(X_test))*100))


# In[ ]:





# In[32]:


import pickle
pickle_out = open("LR_classifier.pkl","wb")
pickle.dump(LR_classifier, pickle_out)
pickle_out.close()


# In[ ]:




