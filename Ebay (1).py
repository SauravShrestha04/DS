#!/usr/bin/env python
# coding: utf-8

# # Analyzing Used cars Listings on eBay Kleinanzeigen

# ## Introduction##

# In this project we will be analysing dataset of used cars from **ebay Kleinanzeigen**, a classifieds section of German ebay website.In addition, the dataset was originally scraped and uploaded to Kaggle [link](https://www.kaggle.com/orgesleka/used-cars-database/data), a few modification from the original dataset has been made.
# <br>In addition, the objective of this project is to first clean the data and analyze the included used car listings.
# 

# In[1]:


import pandas as pd
import numpy as np
autos=pd.read_csv('autos.csv',encoding='Windows-1252')


# In[2]:


autos


# 

# In[3]:


autos.info()
print('The dimension of the dataframe {}'.format(autos.shape))


# We can clearly see that the total number of rows and columns of the dataframe is *5000* and *20* respectively.However we can also see that some data have *null-values*.In the total entries of 50000,some fields are missing approximately 100 some 500.So we will have to analyse the data and take necessary actions where applicable.<br/> In addition, we also have to change the *camelcase* column names to *snakecase*.
# 

# In[4]:


autos.head()


# In[5]:


autos.columns


# In the cell below, we will reassign the column names with a modified column names like *camelcased to snakecase* ,using ' _ ' in between of the same column name like *yearOfRegistration* to *registration_year*.

# In[6]:


autos.columns=['date_crawled','name','seller','offer_type','price','ab_test','vehicle_type','registration_year','gear_box','power_ps','model','odometer', 'registration_month', 'fuel_type', 'brand','unrepaired_damage', 'ad_created', 'num_photos', 'postal_code','last_seen']


# In[7]:


autos.head()


# The data dictionary provided with data is as follows:
# 
# **date_crawled **- When this ad was first crawled. All field-values are taken from this date.<br>
# **name** - Name of the car.<br>
# **seller** - Whether the seller is private or a dealer.<br>
# **offer_type** - The type of listing<br>
# **price** - The price on the ad to sell the car.<br>
# **ab_test** - Whether the listing is included in an A/B test.<br>
# **vehicletype** - The vehicle Type.<br>
# **registration_year** - The year in which the car was first registered.<br>
# **gear_box**- The transmission type.<br>
# **power_ps** - The power of the car in PS.<br>
# **model**- The car model name.<br>
# **kilometer** - How many kilometers the car has driven.<br>
# **registration_month** - The month in which the car was first registered.<br>
# **fuel_type** - What type of fuel the car uses.<br>
# **brand** - The brand of the car.<br>
# **unrepaired_damage** - If the car has a damage which is not yet repaired.<br>
# **ad_created** - The date on which the eBay listing was created.<br>
# **num_photos** - The number of pictures in the ad.<br>
# **postal_code** - The postal code for the location of the vehicle.<br>
# **last_seen** - When the crawler saw this ad last online.<br>

# In[8]:


autos.describe()


# In[9]:


autos.describe(include='all')


# As we can see, the *seller*, *num_photos* and *offer_type* columns have repetitive values , which therefore has to be dropped.

# In[10]:


autos['num_photos'].value_counts()


# In addition, the *num_photos* columns has 0.0 for every column, so we will drop that whole row as well.

# In[11]:


autos=autos.drop(['num_photos','seller','offer_type'],axis=1)


# In[12]:


autos.head()


# There are some extra characters like ',' and '$' in the price column, we will have to remove those characters for further analysis.

# In[13]:


autos['price']=(autos['price']
                .str.replace(',','')
                .str.replace('$','')
                .astype(int)
               )
autos['price'].head()


# In[14]:


autos['odometer'].head()


# In the cell below, we will remove **km**.

# In[15]:


autos['odometer']=(autos['odometer']
                   .str.replace(',','')
                   .str.replace('km','')
                   .astype(int))
autos['odometer'].head()


# In[16]:


autos.rename({'odometer':'odometer_km'},axis=1,inplace=True)
autos['odometer_km'].head()


# In[17]:


print('Minumimum price {}'.format(min(autos['price'])))
print('Maximum price {}'.format(max(autos['price'])))
print('Minimum odometer km {}'.format(min(autos['odometer_km'])))
print('Maximum odometer km {}'.format(max(autos['odometer_km'])))


# In[18]:


autos['odometer_km'].value_counts()


# In[19]:


print(autos['price'].unique().shape)
print('\n')
print(autos['price'].describe())
print('\n')
print(autos['price'].value_counts().head())


# In[20]:


print(autos['price'].value_counts().sort_index(ascending=False).head())


# In[21]:


print(autos['price'].value_counts().sort_index(ascending=True).head(20))


# The price column has price from 0 to 99,999,999 but for the analysis purpose we will only include price in range of 1 to 351000, so that the data doesnt have any outliers in further analysis.For the purpose of including **range** of the dataset we use **df[df["col"].between(x,y)]** function.

# In[22]:


autos=autos[autos['price'].between(1,351000)]

print(autos['price'].describe())


# In[23]:


autos.columns


# In[24]:


autos[['date_crawled','last_seen','ad_created','registration_year','registration_month']].head()


# In[25]:


autos[['registration_year','registration_month']].describe()


# In[26]:


print(autos['date_crawled'].value_counts().head())
print(autos['date_crawled'].str[:10]
      .value_counts(normalize=True,dropna=False)
      .sort_index())
print(autos['ad_created'].str[:10]
      .value_counts(normalize=True,dropna=False)
      .sort_index())
print(autos['last_seen'].str[:10]
      .value_counts(normalize=True,dropna=False)
      .sort_index())



# In[27]:


print(autos['registration_year'].describe())
print('\n')
print(min(autos['registration_year']))

print(max(autos['registration_year']))


# In[28]:


#to determine what percentage of the date are outliers.
(~autos['registration_year'].between(1900,2016)).sum()/autos.shape[0]


# In[29]:


#percentage of data that fall within the year frame of (1900-2016)
(autos['registration_year'].between(1900,2016)).sum()/autos.shape[0]


# In[30]:


autos=autos[autos['registration_year'].between(1900,2016)]
autos['registration_year'].value_counts(normalize=True).head()


# In[31]:


autos.head()


# In[32]:


brand_counts=autos['brand'].value_counts(normalize=True)
common_brand=brand_counts[brand_counts>0.05].index
print(common_brand)


# In[ ]:





# In[33]:


dict_brand={}
for data in common_brand:
    brand_name=autos[autos['brand']==data]
    price_mean=brand_name['price'].mean()
    dict_brand[data]=int(price_mean)
dict_brand


# In[34]:


bmp_series=pd.Series(dict_brand)
pd.DataFrame(bmp_series,columns=['mean_price'])


# In[35]:


mile_dict={}
for data in common_brand:
    brand_only=autos[autos['brand']==data]
    mile_average=brand_only['odometer_km'].mean()
    mile_dict[data]=int(mile_average)
    


# In[36]:


mean_mileage = pd.Series(mile_dict).sort_values(ascending=False)
mean_prices = pd.Series(dict_brand).sort_values(ascending=False)


# In[37]:


brand_info=pd.DataFrame(mean_mileage,columns=['mean_mileage'])
brand_info['mean_prices']=mean_prices


# In[38]:


brand_info.head()


# In[ ]:




