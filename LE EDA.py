#!/usr/bin/env python
# coding: utf-8

# # PREDICTING AVERAGE LIFE EXPECTANCY OF A PERSON OF A COUNTRY
# 
# 
# 
# 

# # Importing Necessary Packages

# In[1]:


get_ipython().system('pip install missingno')


# In[2]:


get_ipython().system('pip install -U scikit-learn==0.20.3')


# In[3]:


import sklearn
import missingno


# In[4]:


#Importing Array and Data Frame operations packages.

import pandas as pd
import numpy as np

#Importing the visualization libraries.

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

#Importing statistical packages.

import scipy.stats as sp
from scipy.stats.mstats import winsorize

#Importing the Simple Imputer from sklearn, for Imputations.

from sklearn.impute import SimpleImputer

#Importing Data preprocessing tools.

from sklearn.preprocessing import MinMaxScaler

#Importing the model evaluation techniques.

from sklearn.metrics import mean_squared_error,r2_score,explained_variance_score
from sklearn.model_selection import cross_val_score

#Importing data splitting tools.

from sklearn.model_selection import train_test_split,KFold,GridSearchCV

#Importing several models to fit our training set.

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

#Importing warnings to ignore the warnings that we come through.

import warnings
warnings.filterwarnings("ignore")


# # Importing Dataset

# In[18]:



import types
import pandas as pd
from botocore.client import Config
import ibm_boto3

def __iter__(self): return 0

# @hidden_cell
# The following code accesses a file in your IBM Cloud Object Storage. It includes your credentials.
# You might want to remove those credentials before you share the notebook.
client_857eb0742adc4184bad6b2192e62efe1 = ibm_boto3.client(service_name='s3',
    ibm_api_key_id='HIJzKvKKfkEAi3bP8S0a4EABkcR-wG9WnmWFJicKts51',
    ibm_auth_endpoint="https://iam.cloud.ibm.com/oidc/token",
    config=Config(signature_version='oauth'),
    endpoint_url='https://s3.eu-geo.objectstorage.service.networklayer.com')

body = client_857eb0742adc4184bad6b2192e62efe1.get_object(Bucket='lifeexpectancy-donotdelete-pr-bizzfsonu6mrov',Key='Life Expectancy Data.csv')['Body']
# add missing __iter__ method, so pandas accepts body as file-like object
if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )

df= pd.read_csv(body)
df.head()


# ### . Preparing the dataset

# In[19]:


df.shape


# Our data has 2938 examples and 22 columns in which 21 are features and 1 (the Life Expectancy column) is the target variable.We will be using the feature values to predict the target variable. 
# 
# ##### A brief summary of the columns that we have in the dataset-:
# 
# 1. Country -: There are 193 unique countries.
# 
# 
# 2. Year -: The data is from the year 2000 to 2015 for the various countries.
# 
# 
# 3. Status -: Representing whether country is Developing or Developed.
# 
# 
# 4. Life Expectancy -: Average age that a person of a specific country will live.
# 
# 
# 5. Adult Mortality -: Probability of dying between 15 and 60 years per 1000 population,considering both the sexes.
# 
# 
# 6. Infant Deaths -: Number of Infant Deaths per 1000 population.
# 
# 
# 7. Alcohol -: Recorded per capita (15+) consumption (in litres of pure alcohol).
# 
# 
# 8. Percentage Expenditure -: Expenditure on health as a percentage of Gross Domestic Product (GDP) per capita(%).
# 
# 
# 9. Hepatitis B -: Hepatitis B immunization coverage among 1-year-olds (%).
# 
# 
# 10. Measles -: Number of reported cases per 1000 population.
# 
# 
# 11. Bmi -: Average Body Mass Index of entire population.
# 
# 
# 12. Under Five Deaths -: Number of under-five deaths per 1000 population.
# 
# 
# 13. Polio -: Polio immunization coverage among 1-year-olds (%).
# 
# 
# 14. Total Expenditure -: General government expenditure on health as a percentage of total government expenditure (%).
# 
# 
# 15. Diphtheria -: Diphtheria immunization coverage among 1-year-olds (%).
# 
# 
# 16. Hiv/Aids -: Deaths per 1000 live births due to HIV/AIDS (0-4 years).
# 
# 
# 17. GDP -: Gross Domestic Product per capita (in USD).
# 
# 
# 18. Population -: Population of the country.
# 
# 
# 19. Thinness 1-19 Years -: Prevalence of thinness among children and adolescents for Age 10 to 19 (% ).
# 
# 
# 20. Thinness 5-9 Years -: Prevalence of thinness among children for Age 5 to 9(%).
# 
# 
# 21. Income Composition Of Resources -: Human Development Index in terms of income composition of resources (index ranging from 0 to 1).
# 
# 
# 22. Schooling -: Number of years of Schooling(years).
# 
# 
# Now let's check the data types of all the columns.
# 

# Since all of our columns are expected to be numeric from the description above,(except Status and Country column), so getting their data types as float or int will verify that there are no unusual charaters present in them.
# 
# If unusual characters are present in them then the data type of the numeric columns will be obtained as 'object' and not 'numeric'.

# In[20]:


df.dtypes


#  
# 
# The datatypes of each of the variables is correct.This tells us that there is no other character serving as a missing value in the dataset which should be replaced by NaN.
# 

# In[21]:


df.columns


# The column names have unnecessary trailing spaces and they are not following any specific pattern.So renaming them with the same pattern will be beneficial in accessing them quickly.
#     
# I will be using the pattern in which each new word will begin with a capital letter and words will be seperated by an underscore.Provided there will be no extra spaces at the starting or the end.

# In[22]:


df.columns=df.columns.str.strip().str.title().str.replace(' ','_').str.replace('-','_')


# In[23]:


df.columns


# ### .Summarizing the data

# In[24]:


#Generating the descriptive statistics of each numerical column.
df.describe(include=np.number)


# 
# The above table represents the statistical summary of the numeric columns of our dataset.Now we will calculate mean of the numeric columns , grouped by each country.This will give us an idea of average data of that country in the 16 years.

# In[25]:


#Grouping data frame by Country.
grouped_data=df.groupby('Country').mean().drop('Year',axis=1)
grouped_data.head()


#     
# Now I am interested in knowing which country has the maximum and minimum average Life Expectancy in the 16 years record.
#     

# In[26]:


print("The country having the Maximum average Life Expectancy in 16 years record is \n")
print(grouped_data[grouped_data['Life_Expectancy']==grouped_data['Life_Expectancy'].max()]['Life_Expectancy'],'\n')
print("The country having the Minimum average Life Expectancy in 16 years record is \n")
print(grouped_data[grouped_data['Life_Expectancy']==grouped_data['Life_Expectancy'].min()]['Life_Expectancy'])


# 
# ##### We get to know that Japan has the highest average Life Expectancy in the last 16 years record whereas Sierra Leone has the least average Life Expectancy.
# 

# In[27]:


df.describe(include='object')


# We have two columns of 'object' data type and the above table represents the summary of Country and Status columns.
# 
# In the later stage , when we will be apply machine learning models then we will have to change these two columns to numeric datatype to feed it into the algorithm.So I will be converting them now.

# In[28]:


df['Country'].value_counts()


# In[29]:


#Creating a dictionary of all the 193 unique countries as keys and the codes assigned to them as their respective values.
unique_country=sorted(df['Country'].unique())
dict_country={}
for i in range(1,len(unique_country)+1):
    dict_country[unique_country[i-1]]=i
dict_country


# The above output represents the unique codes assigned to the 193 different countries.

# In[30]:


#Replacing the values of Country column with codes in the Data Frame.
df['Country']=df['Country'].map(dict_country)


# In[31]:


df.head()


# In[32]:


df['Status'].value_counts()


# In[33]:


#Mapping 'Developing' to 0 and 'Developed' to 1 in the 'Status' column.
df['Status']=df['Status'].map({'Developed':1,'Developing':0})


# 
# #### 1 represents "Developed"
#     
# #### 0 represents "Developing"

# In[34]:


df.head()


# # Exploratory Data Analysis (EDA)

# 
# Exploratory Data Analysis (EDA) is an approach to analyzing data sets and to summarize their main characteristics, often with visual methods.
# 
# 
# This step let's us know more about the patterns our data is following and helps us in determining the relationship between the variables . Finally , observing those patterns we arrive to some important conclusions.

# ### .Univariate Distributions (Continuous Variables)

# In[36]:


plt.figure(figsize=(40,50))
sns.set(font_scale=(1.4))
subset=sorted(df.drop(['Year','Status','Country'],axis=1).columns)
df.hist(column=subset,layout=(5,4),figsize=(40,50),grid=False)
plt.show()


# All the distributions of the continuous variables are either positively skewed or negatively skewed.
# Now let's see their skewness measure.
# (Skewness measure of Normal Distribution is 0) 

# In[37]:


print("Skewness of Continuous Variables")
skewness=[]
for column in subset:
    skewness.append(np.round(sp.skew(df[column].dropna()),2))
skew=pd.DataFrame(skewness,columns=['Skewness Measure'],index=subset)
skew


# ### .Univariate Distributions (Discrete Variables)

# Status, Country and Year are the discrete variables but there is no point of seeing the distribution of Year and Country.

# In[38]:


sns.set(font_scale=(1.0))
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
sns.countplot(df['Status'])
plt.subplot(1,2,2)
plt.pie(df['Status'].value_counts().values,labels=['Developing','Developed'],explode=(0,0.1),autopct='%1.1f%%')


# 
# ###### The bar graph above tells us that there is very few data belonging to the "developed" countires and majority of it has come from the "developing" ones.
# 
# ###### Pie chart gives us some more specific details that, only 17.4% of the data is obtained from the 'Developed' countries whereas 82.6% of it is obtained from the countries that are still 'Developing'.

# ### .Bivariate Distributions

# Our goal is to predict Average Life Expectancy of a person of a country so in this section we will draw plots of different variables with the Life Expectancy variable to see their relationship and draw conclusions from it.
#     
# Now first let us check the trend of Life Expectancy in the 16 years record.For that we will have to group the data by year and then plot the average Life Expectancy recorded in every year.

# In[39]:


grouped_year=df.groupby('Year').Life_Expectancy.mean()
life_expectancy=grouped_year.values
year=grouped_year.index
plt.plot(year,life_expectancy,marker='o')
plt.title(" Average Life Expectancy Trend in 16 Years record")
plt.xlabel("Year")
plt.ylabel("Life Expectancy ")


# ##### We can see that there is approximately a linear trend between Life Expectancy and Year.Also every year there is an increase in the average Life Expectancy.
# 
# 

# In[40]:


sns.scatterplot(x='Adult_Mortality',y='Life_Expectancy',data=df,hue='Status')
plt.title("Scatter Plot of Adult Mortality vs Life Expectancy")
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")


# ###### There is a high negative correlation between Life Expectancy and Adult Mortality.

# In[41]:


sns.scatterplot(x='Alcohol',y='Life_Expectancy',data=df,hue='Status')
plt.title("Scatter Plot of Alcohol vs Life Expectancy")
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")


# ###### There is a somewhat positive correlation between Life Expectancy and Alcohol.

# In[42]:


sns.scatterplot(x='Infant_Deaths',y='Life_Expectancy',data=df,hue='Status')
plt.title("Scatter Plot of Infant Deaths vs Life Expectancy")
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")


# ###### There seems to be very less or no correlation between Infant Deaths and Life Expectancy.

# In[43]:


sns.scatterplot(x='Percentage_Expenditure',y='Life_Expectancy',data=df,hue='Status')
plt.title("Scatter Plot of Percentage Expenditure vs Life Expectancy")
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")


# ###### There seems to be  some correlation between Percentage Expenditure and Life Expectancy.

# In[44]:


sns.scatterplot(x='Hepatitis_B',y='Life_Expectancy',data=df,hue='Status')
plt.title("Scatter Plot of Hepatitis B vs Life Expectancy")
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")


# ###### There seems to be very less or no correlation between Heaptitis B and Life Expectancy.

# In[45]:


sns.scatterplot(x='Bmi',y='Life_Expectancy',data=df,hue='Status')
plt.title("Scatter Plot of Bmi vs Life Expectancy")
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")


# ###### There is a positive correlation between Bmi and Life Expectancy.

# In[46]:


sns.scatterplot(x='Diphtheria',y='Life_Expectancy',data=df,hue='Status')
plt.title("Scatter Plot of Diphtheria vs Life Expectancy")
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")


# ###### There is a positive correlation between Diphtheria and Life Expectancy.

# In[47]:


sns.scatterplot(x='Gdp',y='Life_Expectancy',data=df,hue='Status')
plt.title("Scatter Plot of Gdp vs Life Expectancy")
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")


# ###### There is a positive correlation between Gdp and Life Expectancy.

# In[48]:


sns.scatterplot(x='Measles',y='Life_Expectancy',data=df,hue='Status')
plt.title("Scatter Plot of Measles vs Life Expectancy")
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")


# ###### There seems to be no correlation between Measles and Life Expectancy.

# In[49]:


sns.scatterplot(x='Under_Five_Deaths',y='Life_Expectancy',data=df,hue='Status')
plt.title("Scatter Plot of Under Five Deaths vs Life Expectancy")
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")


# ###### There seems to be very less or no correlation between Under Five Deaths and Life Expectancy.

# In[50]:


sns.scatterplot(x='Polio',y='Life_Expectancy',data=df,hue='Status')
plt.title("Scatter Plot of Polio vs Life Expectancy")
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")


# ###### There is a positive correlation between Polio and Life Expectancy.

# In[51]:


sns.scatterplot(x='Total_Expenditure',y='Life_Expectancy',data=df,hue='Status')
plt.title("Scatter Plot of Total Expenditure vs Life Expectancy")
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")


# ###### There is a slight positive correlation between Total Expenditure and Life Expectancy.

# In[52]:


sns.scatterplot(x='Hiv/Aids',y='Life_Expectancy',data=df,hue='Status')
plt.title("Scatter Plot of Hiv/Aids vs Life Expectancy")
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")


# ###### There is a negative correlation between Life Expectancy and Hiv/Aids.

# In[53]:


sns.scatterplot(x='Population',y='Life_Expectancy',data=df,hue='Status')
plt.title("Scatter Plot of Population vs Life Expectancy")
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")


# ###### There is no correlation between Life Expectancy and Population.

# In[54]:


sns.scatterplot(x='Thinness__1_19_Years',y='Life_Expectancy',data=df,hue='Status')
plt.title("Scatter Plot of Thinness 1-19 Years vs Life Expectancy")
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")


# ###### There is a slight negative correlation between Life Expectancy and Thinness 1-19 Years.

# In[55]:


sns.scatterplot(x='Thinness_5_9_Years',y='Life_Expectancy',data=df,hue='Status')
plt.title("Scatter Plot of Thinness 5-9 Years vs Life Expectancy")
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")


# ###### There is a slight negative correlation between Life Expectancy and Thinness 5-9 Years.

# In[56]:


sns.scatterplot(x='Income_Composition_Of_Resources',y='Life_Expectancy',data=df,hue='Status')
plt.title("Scatter Plot of Income_Composition_Of_Resources vs Life Expectancy")
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")


# ###### There is a positive high correlation between Life expectancy and Income Composition of Resources.

# In[57]:


sns.scatterplot(x='Schooling',y='Life_Expectancy',data=df,hue='Status')
plt.title("Scatter Plot of Schooling vs Life Expectancy")
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")


# ###### There is a positive high correlation between Life expectancy and Schooling.

# In[58]:


sns.boxplot(y='Life_Expectancy',x='Status',data=df)


# From the above graph we can interpret that the minimum Life Expectancy of "Developed" countries is equal to the median Life expectancy of the "Developing" countries.
# 
# ###### This results in the inference that "Developed" countries have a higher average Life Expectancy as compared to the "Developing" countries.
# 
# This result can also be verified by the scatter plots plotted above.All the 'orange' dots that represent the 'Developed' countries are mainly present in the region of high 'Life Expectancy' whereas all the 'blue' dots that represent the 'Developing' countries are mainly present in the region of low 'Life Expectancy'.

# Now we will plot the correlation matrix between different features and visualise it with the help of a heatmap.

# In[59]:


corr_matrix=df.corr()
corr_matrix


# 
# 
# 
# Its hard to draw results by seeing the above matrix so now we will visualise the correlation matrix with the help of a heatmap.
# 
# ###### A heatmap is nothing but a visual correlation matrix which shows different correlation values with different colors so we can easily differentiate them.

# In[60]:


plt.figure(figsize=(40,40))
sns.set(font_scale=2.4)
sns.heatmap(corr_matrix,annot=True,cmap='viridis',cbar=False)


# In the heatmap above, the warmer colors show higher and positive correlation, while the colder ones show low or negative correlation.
# 
# There is an observable correlation between the following columns-:
# 
# 1. Adult Mortality and Life Expectancy have a high negative correlation.
# 
# 
# 2. Life Expectancy has a high positive correlation with Income Composition of Resources and Schooling.
# 
# 
# 3. Life Expectancy has a positive correaltion with Bmi and a negative correaltion with Hiv/Aids.
# 
# 
# 4. Infant Deaths is 100% correlated with Under Five Deaths.
# 
# 
# 5. Percentage Expenditure has a very high positive correlation with Gdp.
# 
# 
# 6. Thinness 5-9 Years has a very high positive correlation with Thinness 1-19 Years.
# 
# 
# 7. Income Composition of Resources has a high positive correlation with Schooling.

# ### Final Conclusions

# 1. Developed countries have a higher average Life Expectancy than the Developing Countries.
# 
# 
# 2. A Country having low Adult Moratlity Rate has a high Life Expectancy and vice versa.
# 
# 
# 3. A Country which has more number of Schooling Years has a high Life Expectancy.
# 
# 
# 4. More the Income Composition of Resources of a Country ,more is its Life Expectancy.
# 
# 
# 5. More the average Bmi of the entire population of a country, more is its average Life Expectancy.
# 
# 
# 6. Less the number of deaths by Hiv/Aids,more is the average Life Expectancy of the country.

# # Data Preprocessing

# It is one of the most important step before proceeding to the model development for prediction.It involves many steps and I will be listing the ones, that I am going to use-:
# 
# ###### 1. Treating Missing Values
# 
# There are many reasons for missing values in a dataset.These may include, loss of recorded data , data not available at the time of survey, recorded data is incomplete or it may be recorded incorrectly.Missing data leads to invalid and inappropriate conclusions so it needs to be treated properly.There are many ways of handling missing values.Here I will be using Simple Imputer to impute the missing values.
# 
# ###### 2. Outlier Analysis
# 
# Sometimes there maybe some values in the dataset that are either overestimated or underestimated,diverting from the patterns that they should be following.These values are nothing but outliers.Presence of outliers degrades our model and gives poor predictions.So they should be analysed and treated well.
# 
# ###### 3. Standardization of Data
# 
# Sometimes our dataset values may have large differences between their ranges.These differences cause problems for the machine learning models.This is when standardization comes into role.It converts all the input data into a common standard format which can then be fed into the model for making predictions.
# 
# ###### 4. Feature Selection
# 
# It refers to selecting the most important features that are actually responsible for making the prediction.Sometimes a lot number of unimportant features may result in poor accuracy of the model so in that case feature selection is a technique that should be used.Although having a very less number of features for making predictions, is also not advisable.

# ### Handling Missing Values

# In[61]:


df.isnull().sum()


# In[62]:


missing_cols=[]
for column in df.columns:
    if df[column].isnull().sum()!=0:
        missing_cols.append(column)
missing_cols


# In[63]:


def null_values(data):
    null_number=data.isnull().sum().values
    null_percent=(null_number/data.isnull().count()*100).values
    null_table=pd.DataFrame({'No.of null values':null_number,'Percentage of null values':null_percent},index=data.columns)
    print("The null values and their percentages in the dataset are")
    return null_table
null_values(df)


# In[64]:


sns.set(font_scale=(1.0))
import matplotlib.pyplot as plt
plt.bar(height=null_values(df)['Percentage of null values'].values,x=df.columns)
plt.xticks(rotation=90)
plt.title("Graph showing Percentage of missing values in each column")
plt.show()


# The dataset has a high number of missing values, especially in the Population,Gdp,Alcohol,Total Expenditure,Schooling,Income Composition of Resources and Hepatitis B column.So removing these records will highly shrink our dataset.We will try to impute the missing values with some other value.

# In[65]:


missingno.matrix(df)


# The above matrix displays the missingness pattern in the dataset.We can make the following inferences from it -:
# 
# 1. Life Expectancy and Adult Mortality have missing values exactly at the same place,i.e. they have nullity  correlation equal to 1.
# 
# 
# 2. Polio and Diphtheria also have nullity correlation equal to 1.
# 
# 
# 3. Missing values in Alcohol and Total Expenditure correlate really well with each other.
# 
# 
# 4. Bmi, Thinness 5-9 Years and Thinness 1-19 Years have nullity correlation as 1.
# 
# 
# 5. Missing values in Gdp and Population seem to have a slight correlation between them.
# 
# 
# 6. Missing values in Schooling and Income Composition of Resources correlate really well with each other.
# 
# 
# 7. Missing values in Hepatitis B are not related to any column.
# 
# 
# Now we will see the nullity correlation with the help of a heatmap and verify the above results.

# In[66]:


missingno.heatmap(df,cbar=False)


# In the heatmap, the lighter shades indicate a low nullity correlation whereas the darker shades indicate a high nullity correlation between the variables.
# 
# 
# The heatmap successfully verifies our results derived from the above matrix.Now we move to part of getting rid of these NaNs by imputing them with some other value.

# #### Imputing Missing Values Using Sci-kit learn's Simple  Imputer

# Here I am using Simple Imputer from sklearn which uses the technique -: Replacing the missing values with the strategy chosen.(Here I will be using 'mean').
# 

# In[67]:


iter=SimpleImputer()
# The initial_strategy='mean' by default.
imp_df_array=iter.fit_transform(df)

#This returns a two dimensional array with no column labels and no index.So we will have to convert it into a Dataframe.

imputed_df=pd.DataFrame(imp_df_array,columns=df.columns,index=df.index)
imputed_df.head()


# In[68]:


imputed_df.shape


# In[69]:


#Checking whether the null values are present after imputation or not.
null_values(imputed_df)


# We can see that all the null values have been removed from the dataset.So now we have a clean dataset for further processing.
# 
# 
# Now let's compare the distribution of the observed and the imputed values by plotting the side by side density and box plots to verify that there is not a huge difference between the two distributions.(We will be plotting the graphs only for those features which had missing values).

# In[70]:


sns.set(font_scale=2.0)
i=1
plt.figure(figsize=(40,50))
for column in missing_cols:
    plt.subplot(5,3,i)
    sns.distplot(df[column],hist=False,color='cyan',kde_kws={'linewidth':6},label='Observed Data Distribution')
    sns.distplot(imputed_df[column],hist=False,color='red',kde_kws={'linewidth':6},label='Imputed Data Distribution')
    plt.legend()
    i=i+1


# The imputed density plots have the same distribution as the observed distribution for most of the variables.The distribution only varies slightly for Alcohol, Total Expenditure, Gdp, Hepatitis B, Income Composition of Resources and Schooling. column.Now let's check their distribution using the side by side box plots.

# In[71]:


sns.set(font_scale=2.0)
i=1
plt.figure(figsize=(40,50))
for column in missing_cols:
    plt.subplot(5,3,i)
    combined=pd.DataFrame({'Observed Data Distribution':df[column],'Imputed Data Distribution':imputed_df[column]})
    sns.boxplot(data=combined)
    plt.title(column)
    i=i+1


# The box plot distribution also shows the same distribution between observed and imputed values,except in Alcohol, Total Expenditure, Gdp, Hepatitis B and Schooling columns, where the distribution has spread slightly downwards, changing the minimum value as well.
# 
# Now we will check the two statistics, mean and standard deviation,of the both observed and imputed data.

# In[72]:


observed_statistic=np.round(df[missing_cols].aggregate(['mean','std']),2).T
imputed_statistic=np.round(imputed_df[missing_cols].aggregate(['mean','std']),2).T
comparison_table=pd.concat([observed_statistic,imputed_statistic],axis=1)
comparison_table.columns=['Mean Observed','Std Observed','Mean Imputed','Std Imputed']
comparison_table


# The mean and standard deviation values are also almost same for all the variables except for, Hepatitis B, Population and Gdp.
# 
# Now after the data imputation we move on to our next important step,i.e. Understanding and Treating Outliers. 

# ### Outlier Detection and Treatment

# ###### 1. Outlier Detection 
# 
#       i Using Box Plots
#     
#       ii Using Inter Quartile Range Method
#     
#     
# ###### 2. Dealing with Outliers
#    
#       i Dropping or
#    
#       ii Winsorization or
#    
#       iii Log/Square Root Transformation

# #### Outlier Detection Using Box Plots

# We will start by plotting the box plot for each of the continuous variables of the dataset and see whether they contain outliers or not , and if yes then how many.

# In[73]:


continuous_vars=imputed_df.drop(['Year','Country','Status'],axis=1).columns
plt.figure(figsize=(40,50))
i=1
for cols in continuous_vars:
    plt.subplot(5,4,i)
    sns.boxplot(y=cols,data=imputed_df)
    plt.title(cols)
    i=i+1


# 
# 
# There are certain columns which either have none or some outliers,e.g. Income Composition of Resources,Schooling,Bmi and Alcohol.But the majority of them have a large number of outliers.
# 
# These outliers may be of certain information to us. Like unexpected high number of Infant and Under Five Deaths may be due to the high number of deaths of children suffering from Measles or Hiv/Aids.Or it may also be due to the less immunization coverage (in %) for Polio or Diphtheria.
# 
# So in this case, removing them is not an option.They might be of certain information to us rather than just measurement error.
# 

# #### Outlier Detection Using Inter Quartile Range Method

# Previously we had visualised outliers present in each of the continuous variables but we didn't get the exact number of outliers in each one of them.
# 
# With this method we can see the number and percentage of outliers in each of the continuous variables.
# 
# We will create a Data Frame containing number of outliers and their percentages in the respective columns.

# In[74]:


numbers=[]
percentages=[]
for cols in continuous_vars:
    outlier_number=0
    Q1,Q3=np.percentile(imputed_df[cols],[25,75])
    IQR=Q3-Q1
    lower=Q1-1.5*IQR
    upper=Q3+1.5*IQR
    for value in imputed_df[cols]:
        if ((value<lower) or (value>upper)):
            outlier_number+=1
    outlier_percent=np.round(outlier_number/len(imputed_df[cols])*100,2)
    numbers.append(outlier_number)
    percentages.append(outlier_percent)
outlier_table=pd.DataFrame({"Number of Outliers":numbers,'Percentage of Outliers':percentages},index=continuous_vars)
outlier_table


# 
# 

# The table successfully verifies our visual result, that a lot of columns have a large number of outliers present in them.
# 
# Now comes the part of dealing with them.

# #### Dealing With Outliers (WINSORIZATION)

# As we discussed earlier, dropping outliers in this case will result in the shrinkage and loss of important data.Also each variable has its own unique set of outliers. 
# 
# ###### So the best step that we can take here is WINSORIZATION.
# 
# It is a way to minimize the influence of outliers in your data by either:
# 
#      i  Assigning the outlier a lower weight, or
#    
#      ii Changing the value so that it is close to other values in the set.
#    
# This will put on limits on the extreme values of each variable until all the outliers are modified.The upper and lower limits are defined for each column separately.Limits tell us how much data is to be cut on each side(upper and lower).They are set by visually analysing the outliers present on each side of the boxplot.A satisfactory limit is said to be reached when there are no outliers present.(We usually start by assigning small values to the limits.)
# 
# Finally, we will visualize the original and the winsorized columns using side by side boxplots and when each of the outliers have been treated, then we will create our new data frame with the winsorized columns.

# In[75]:


sns.set(font_scale=2.0)
i=1
wins_data={}
lower_limit=[0.01,0.0,0.0,0.0,0.0,0.15,0.0,0.0,0.0,0.095,0.0,0.12,0.0,0.0,0.0,0.0,0.0,0.05,0.05]
upper_limit=[0.0,0.03,0.12,0.002,0.15,0.0,0.2,0.0,0.15,0.0,0.02,0.0,0.19,0.15,0.12,0.04,0.04,0.0,0.05]
plt.figure(figsize=(40,50))
for column in continuous_vars:
    plt.subplot(5,4,i)
    
    #Performing Winsorization technique for each continuous variable
    
    wins_data[column]=winsorize(imputed_df[column],limits=(lower_limit[i-1],upper_limit[i-1]))
    combined=pd.DataFrame({'Original':imputed_df[column],'Winsorized':wins_data[column]})
    sns.boxplot(data=combined)
    plt.title(column)
    i=i+1


# 
# 
# From the above boxplot comparison we can see that all the winsorized columns obtained, are now free from outliers.Now we create our new Data Frame having the winsorized columns.

# In[76]:


winsorized_df=pd.DataFrame(wins_data)
winsorized_df.head()


# In[77]:


winsorized_df.shape


# Our winzorized data frame only contains the continuous variables,i.e. only 19 columns out of 22.We have to concat the rest of the three categorical columns to form our complete data frame. 
# 
# After this step we will move on to our next step,i.e. Data Scaling

# In[78]:


winsorized_df=pd.concat([winsorized_df,imputed_df[['Year','Country','Status']]],axis=1)
winsorized_df.head()


# ### Standardization of Data 

# 
# In our dataset some of the columns like, Income Composition of Resources, are between 0 and 1. Some of them are between 1-100,or even in range of thousands, but there are also some columns like, Population which are in millions and billions.Also the unit of measurement for every feature is different.Hence our dataset requires scaling of features for proper prediction of Life Expectancy.(Only the features values are scaled and the target is kept free from scaling.)
# 
# Data Scaling will convert every value of features into a common data format.We will be using MinMaxScaler from sklearn library.
# 
# MinMaxScaler rescales the data set such that all feature values are in the range [0,1]. This transformation is often used as an alternative to zero mean, unit variance scaling.

# In[79]:


#Separating the features and the target variable.
X=winsorized_df.drop('Life_Expectancy',axis=1)
y=winsorized_df['Life_Expectancy']


# In[80]:


#Scaling the features.
scaler=MinMaxScaler()
X_scaled=scaler.fit_transform(X)


# 
# The scaler.fit_transform() function returns a 2-D array with no column labels and no indexes.So we will have have to convert the features back into a data frame form to feed it into the machine learning model.

# In[81]:


X_df=pd.DataFrame(X_scaled,columns=winsorized_df.drop('Life_Expectancy',axis=1).columns,index=winsorized_df.index)


# In[82]:


X_df.head()


# Now the feature scaling has been successfully done.And we can see that the feature values range between [0,1].
# 
# We come to our last step of Data Preprocessing ,i.e. Feature Selection.

# ### Feature Selection

# Presence of  unimportant features sometime result in poor predictions.So removing them and using only specific features will improve the score of the model.
# 
# 
# Here I will be using one of the ensemble methods, i.e. Random Forest Regressor, to get the importance of all the features.

# In[83]:


rf=RandomForestRegressor()
rf.fit(X_df,y)
feat=rf.feature_importances_
feature_imp=pd.Series(feat,index=X_df.columns).sort_values(ascending=False)
feature_imp


# Visualising these variable importance values using a barplot

# In[84]:


sns.set(font_scale=(1.0))
feature_imp.plot(kind='bar')


# From the visualisation above, we can see that there are only three variables , Income Composition of Resources, Hiv/Aids and Adult Mortality which are of very high importance.Some of them have very low importances and there are even some variables whose importance seems to be negligible in the graph.
# 
# Removing those variables which have negligible importance will surely help in increasing the score of our model.
# 
# ###### Important Features -:
# 
# ###### 'Hiv/Aids', 'Adult_Mortality', 'Income_Composition_Of_Resources', 'Schooling', 'Under_Five_Deaths', 'Bmi', 'Thinness_5_9_Years', 'Year', 'Alcohol',  'Thinness__1_19_Years', 'Country',' Total_Expenditure'.
# 
# We will be using only these 12 features for our model prediction.

# In[85]:


features=X_df[['Hiv/Aids','Adult_Mortality','Income_Composition_Of_Resources','Schooling','Under_Five_Deaths','Bmi','Thinness_5_9_Years','Year','Alcohol','Thinness__1_19_Years','Country','Total_Expenditure']]


# # Model Development and Evaluation

# There are many regression models available in the scikit-learn package.But we have to choose which model to apply to get a better result.
# 
# We can fit different models to our dataset and then decide which model is best for our data.This can be done using the cross validation technique.(Or K Fold Cross Validation)
# 
# 
# Cross Validation is a resampling procedure used to evaluate machine learning models on a dataset.It is called K Fold Cross Validation when we specify a parameter k as a fixed integer value.'k' refers to the number of groups we want our dataset set to be split into.For each k-fold in your dataset, the model is trained on k – 1 folds of the dataset.And then it is tested on the kth model.Score of the model is recorded for each of the predictions and the final score is obtained by taking the mean of all the scores.The better the score ,the better the model.
# 
# Models I will be using here are-:
# 
# ###### Linear Regression, Lasso, Ridge , Decision Tree Regressor, Elastic Net Model and Random Forest Regressor.

# In[86]:


#Creating model instances.
lr=LinearRegression()
lasso=Lasso()
ridge=Ridge()
enet=ElasticNet()
dt=DecisionTreeRegressor()
randomforest=RandomForestRegressor(random_state=42)
regressors=['Linear Regression','Lasso','Ridge','Elastic Net','Decision Tree','Random Forest']
models=[lr,lasso,ridge,enet,dt,randomforest]


# In[87]:


mean_score=[]
#Specifying the K Fold splits.
kfold=KFold(n_splits=5,shuffle=True,random_state=1)
for model in models:
    #Calculating mean score for each of the six models.
    result=cross_val_score(model,features,y,cv=kfold,scoring='r2')
    mean_score.append(np.round(result.mean()*100,2))
score_table=pd.DataFrame(mean_score,index=regressors,columns=["R2 score"])
score_table


# From the above table we can see that Elastic Net model has least model score whereas Random Forest has highest score of approx. 96%.
# 
# ###### So we will be using Random Forest Regressor for Predicting Life Expectancy.

# ### Splitting the datset into training and testing data

# In[88]:


X_train,X_test,y_train,y_test=train_test_split(features,y,test_size=0.3,random_state=0)


# In[89]:


X_train.shape


# In[90]:


y_train.shape


# In[91]:


X_test.shape


# In[92]:


y_test.shape


# ### Hyperparameter Tuning (Using GridSearchCV)

# Now after splitting the dataset we have to fit Random Forest Regressor Model on our dataset.But there are various parameters of Random Forest model which further affect the performance of the model.
# 
# So our aim is to find the best parameters which will give the best model performance.These best parameters are found with the help of scikit-learn's GridSearchCV module.It contains a list of parameter settings as one of its parameter to try different values and give the one which provides the best score for that model.

# In[93]:


param_grid={
# Number of trees in random forest
'n_estimators':[100,150,200],
# Number of features to consider at every split
'max_features':['auto', 'sqrt'],
#Criteria 
'criterion':['mse','mae']
}
CV_rf= GridSearchCV(estimator=randomforest, param_grid=param_grid, cv= 5)
CV_rf.fit(X_train, y_train)


# After applying GridSearchCV the best parameters for a model are obtained by the below given command.

# In[94]:


CV_rf.best_params_


# ###### The best parameters for our Random Forest Model are -: 
# 
# 1. Criterion = The function to measure the quality of split = 'mse' where it stands for 'mean standard error'
# 
# 2. Maximum number of features to consider when looking for the best split = 'auto'
# 
# 3. Number of trees in the forest =100

# Now we will create our random forest regressor model using these parameters.

# In[95]:


def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    r2=r2_score(y_test,predictions)
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    print('R2 score = {:0.2f}%.'.format(r2*100))
    return accuracy


# Here I will be evaluating my model on the basis of three metrics.
# 
# ###### 1.  Mean Absolute Percentage Error (or Accuracy = (100 - Mean Absolute Percentage Error)% ) 
# 
# ###### 2. Mean Absolute Error 
# 
# ###### 3. R2 score
# 
# ###### Lower the Mean Absolute percentage Error , the better is the model (or higher the Accuracy percentage, the better is the model)
# 
# ###### Lower the Mean Absolute Error , the better is the model.
# 
# ###### Higher the R2 score , the better is the model.

# In[96]:


best_grid = CV_rf.best_estimator_
grid_accuracy = evaluate(best_grid, X_test, y_test)


# Finally we have created our model.
# 
# We have got a really high Accuracy of approx. 98% , a R2 score of approx. 96% and a very low mean absolute error of approx. 1.20 degrees on applying the model on the test dataset.
# Now we will feed some data into our model to get the predictions.

# In[100]:


#Remember to give the features values in same order as you defined your final features dataset.
le=best_grid.predict([[250,54.5,0.95,15.5,500,56.8,20.5,2018,50.9,67.8,164,40.8]])
print('The Average Life Expectancy of a person of a country with given features is = ', le," Years")


# ### Creating Scoring Endpoints

# In[101]:


get_ipython().system('pip install watson-machine-learning-client')


# In[102]:


from watson_machine_learning_client import WatsonMachineLearningAPIClient


# In[103]:


wml_credentials={
  "apikey": "uNQt8F_bqobi8gLXdudM9sr2eluuFREAtoYM4qmZl5zd",
  "instance_id": "4e1f0a63-e01d-4558-9c08-2d2f1a658c8c",
  "url": "https://eu-gb.ml.cloud.ibm.com"
}


# In[104]:


client = WatsonMachineLearningAPIClient(wml_credentials)


# In[105]:


model_props = {client.repository.ModelMetaNames.AUTHOR_NAME: "Anchal",
   client.repository.ModelMetaNames.AUTHOR_EMAIL: "anchalagarwal.du.or.21@gmail.com",
   client.repository.ModelMetaNames.NAME: "Life_Expectancy"}


# In[106]:


model_artifact =client.repository.store_model(best_grid, meta_props=model_props)


# In[107]:


published_model_uid = client.repository.get_model_uid(model_artifact)


# In[108]:


published_model_uid


# In[109]:


deployment = client.deployments.create(published_model_uid, name="Life_Expectancy")


# In[110]:


scoring_endpoint = client.deployments.get_scoring_url(deployment)


# In[111]:


scoring_endpoint


# In[ ]:




