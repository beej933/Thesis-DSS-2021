#!/usr/bin/env python
# coding: utf-8

# ## Importing libraries

# In[1]:


## To read the csv files 
import pandas as pd
import numpy as np

## For the data visualization 
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

## To train the models
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import datasets
from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation, SpectralClustering, MeanShift, DBSCAN, OPTICS, Birch

## For the evalution of the models
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics.cluster import pair_confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn import cluster


## To ignore unnecassry warnings
import warnings
warnings.filterwarnings('ignore')


# ## Exploratory Data Anaylsis (EDA)
# https://www.kaggle.com/saumya5679/covid-19-prediction-97-eda

# In[2]:


## Reading the data (using read_csv) and taking a look at the first 5 lines using the head() method.
df = pd.read_csv('Covid Dataset.csv')
df.head()


# In[3]:


## Using the info() method to output some general information about the dataframe like as missing values, types etc
df.info()


# In[4]:


## Printing out column names using columns function
df.columns


# In[5]:


## Checking the shape of the data
df.shape


# ## Checking missing values

# In[6]:


## Checking missing values
df.isnull().sum()


# ## Checking the list of the catagorical and numerical columns and its length 

# In[7]:


## Using a loop to check the list of catagorical and numerical columns and its length.

## Creating a list for any catagorical columns
cat_cols = []

## Creating a list for any numerical columns
num_cols = []

## Cearting a list for binary columns
binary_cols = []

for i in df.columns:
    if df[i].dtypes =='object':
        cat_cols.append(i)        
    else:
        if df[i].nunique() == 2:
            binary_cols.append(i)
        else:
            num_cols.append(i) 


# In[8]:


print(cat_cols)
print("Length of catagorical columns : ",len(cat_cols))
print(num_cols)
print("Length of numerical columns : ",len(num_cols))
print(binary_cols)
print("Length of the binary columns: ", len(binary_cols))


# # Descriptive Anaylsis 

# In[9]:


## In order to see statistics on non-numerical features, 
## we explicitly indicate data types of interest in the include parameter.
df.describe(include = ['object', 'bool'])


# ## Data vizualisation

# ### COVID-19 (target)

# In[10]:


## A countplot for target variable
sns.countplot(x='COVID-19',data=df)
plt.show()


# In[11]:


df["COVID-19"].value_counts().plot.pie(explode=[0.1,0.1],autopct='%1.1f%%',shadow=True)
plt.title('number of cases');


# ### Breathing Problem

# In[12]:


## A countplot on Breathing Problem variable
sns.countplot(x='Breathing Problem',data=df)
plt.show()


# In[13]:


## A bivariate Analysis between Breathing Problem and Covid-19
sns.countplot(x='Breathing Problem',hue='COVID-19',data=df)
plt.show()


# ## Fever

# In[14]:


## A countplot on fever with target variable (Covid-19)
sns.countplot(x='Fever',hue='COVID-19',data=df);
plt.show()


# ## Dry Cough

# In[15]:


## A countplot on Dry Cough variable
sns.countplot(x='Dry Cough',hue='COVID-19',data=df)
plt.show()


# ## Sore throat

# In[16]:


## A countplot on Sore Throat variable
sns.countplot(x='Sore throat',hue='COVID-19',data=df)
plt.show()


# # Feature transformation
# 
# ### Label Encoding

# In[17]:


## Label encoding
from sklearn.preprocessing import LabelEncoder

labelEnc = LabelEncoder()
for i in list(df.columns):
    if df[i].dtype == 'object':
        df[i] = labelEnc.fit_transform(df[i])


# In[18]:


df.head()


# In[19]:


## Counting the value for the target variable
df.dtypes.value_counts()


# In[20]:


## A histogram plot on dataset variables
df.hist(figsize=(20,15));


# ## Checking the correlation between the features 

# In[21]:


corr = df.corr()
corr


# In[22]:


## Plotting correlation matrix
plt.figure(figsize=(10,10))
sns.heatmap(df.corr(), annot=True)
corr.style.background_gradient(cmap='cool',axis=None)


# # Feature Engineering & feature selection 

# In[23]:


## Printing the correlation in ascending order
print(corr["COVID-19"].sort_values(ascending=False))


# In[24]:


## Checking the percentage of missing values, unique values and percentage of categories consisting of one value
stats = []
for col in df.columns:
    stats.append((col, df[col].nunique(), df[col].isnull().sum() * 100 / df.shape[0], df[col].value_counts(normalize=True, dropna=False).values[0] * 100, df[col].dtype))
    
    ## Exploratory Data Anaylsis
stats_df = pd.DataFrame(stats, columns=['Feature', 'Unique values', 'Percentage of missing values', 'Percentage of category with one value', 'Type'])
stats_df.sort_values('Percentage of missing values', ascending=False)


# In[25]:


## Removing columns that exceed the 90% threshold of consisting of only one category values or missing values 
good_cols = list(df.columns)
for col in df.columns:
    
    ## Removing columns with a high NA rate
    na_rate = df[col].isnull().sum()/ df.shape[0]
    
    ## Removing columns with high unbalanced values rate
    unbalanced_rate = df[col].value_counts(normalize=True, dropna=False).values[0]
    
    if na_rate > 0.9 and unbalanced_rate > 0.9:
        good_cols.remove(col)         


# In[26]:


## Extracting the columns that do not exceed the threshold of missing values or being one value column
df = df[good_cols]
df.head()


# #### Feature that are going to be deleted :
# Running Nose / Asthma /Chronic Lung Disease / Headache / Heart Disease / Diabetes / Fatigue / Gastrointestinal / Wearing Masks / Sanitization from Market

# In[27]:


df=df.drop(['Wearing Masks','Sanitization from Market'],axis=1)


# In[28]:


corr=df.corr()
corr.style.background_gradient(cmap='cool',axis=None)


# # Extract Dependent and Independent Variables

# In[29]:


x=df.drop('COVID-19',axis=1)
y=df['COVID-19']


# # Split Train Test

# In[30]:


## Splitting the data in train and remaining set
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)


# # Dealing with Imbalanced Data

# ### Standard Scaling

# In[31]:


stdd = StandardScaler()
x_train = stdd.fit_transform(x_train)


# ### SMOTE for unbalanced data

# In[32]:


smote = SMOTE(sampling_strategy='minority')
x_train,y_train = smote.fit_resample(x_train,y_train) 


# # K-means with elbow method
# https://www.kaggle.com/drscarlat/compare-10-unsupervised-clustering-algorith-iris#Compare-various-clustering-algorithms-on-the-iris-dataset

# In[33]:


## Using the elbow method to find optimal number of clusters 
cs = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(x_train)
    cs.append(kmeans.inertia_)
plt.plot(range(1, 11), cs)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('CS')
plt.show()


# # Training and testing

# In[34]:


## Defining all the clustering algorithms 
algorithms = []
algorithms.append(KMeans(n_clusters=2, random_state=1))
algorithms.append(AffinityPropagation())
algorithms.append(SpectralClustering(n_clusters=2, random_state=1,
                                     affinity='nearest_neighbors'))
algorithms.append(AgglomerativeClustering(n_clusters=2))
algorithms.append(MeanShift(bandwidth=2))
algorithms.append(DBSCAN())
algorithms.append(OPTICS())
algorithms.append(Birch())


# # Training of data and evaluation on the test data

# In[39]:


data = []
for algo in algorithms:
    algo.fit(x_train)
    y_pred = algo.fit_predict(x_test)
    data.append(({
        'Accuracy': accuracy_score(y_test, y_pred),
        'ARI': metrics.adjusted_rand_score(y_test, y_pred),
        'AMI': metrics.adjusted_mutual_info_score(y_test, y_pred,
                                                 average_method='arithmetic'),
        'Homogeneity': metrics.homogeneity_score(y_test, y_pred),
        'Completeness': metrics.completeness_score(y_test, y_pred),
        'V-measure': metrics.v_measure_score(y_test, y_pred)}))

results = pd.DataFrame(data=data, columns=['Accuracy', 'ARI', 'AMI', 'Homogeneity',
                                           'Completeness', 'V-measure'],
                       index=['K-means', 'Affinity', 
                              'Spectral', 'Agglomerative', 'MeanShift', 'DBSCAN', 'OPTICS', 'BIRCH'])

round(results, 3)


# In[36]:


k = [1, 2, 3, 4, 5, 6, 7, 8, 10]
scores = []
 
for n_cluster in k:
    y_pred = KMeans(n_clusters = n_cluster, max_iter=1000, random_state=47).fit_predict(x_test)
    score = metrics.homogeneity_completeness_v_measure(y_test, y_pred)
    scores.append(score)
 
# plotting the scores against the value of k
plt.plot(k, [s[0] for s in scores], 'r', label='Homogeneity')
plt.plot(k, [s[1] for s in scores], 'b', label='Completeness')
plt.plot(k, [s[2] for s in scores], 'y', label='V-Measure')
plt.xlabel('Value of K')
plt.ylabel('homogeneity_completeness_v_measure')
plt.legend(loc=4)
plt.show()


# # Contigency matrix and Pair Confusion matrix

# In[37]:


## Contigency matrics
contingency = contingency_matrix(y_test, y_pred)
contingency


# In[38]:


## Pair Confusion Matrix
pair_confusion = pair_confusion_matrix(y_test, y_pred)
pair_confusion

