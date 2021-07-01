#!/usr/bin/env python
# coding: utf-8

# ## Collaborative Filtering

# The technique we're going to take a look at is Collaborative filtering.It is based on the fact that relationships exist between products and people's interests. Many recommendation systems use collaborative filtering to find these relationships and to give an accurate recommendation of a product that the user might like or be interested in.
# 

# ### Item Based Collaborative Filtering
# 
# Item-item collaborative filtering, or item-based, or item-to-item, is a form of collaborative filtering for recommender systems based on the similarity between items calculated using people's ratings of those items.

# ![ItemBased.jpg](attachment:ItemBased.jpg)

# ### Dataset

# This data set consists of:
# 	* 100,000 ratings (1-5) from 943 users on 1682 movies. 
# 	* Each user has rated at least 20 movies. 

# ## Preprocessing

# In[1]:


import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
import math


# In[2]:


ratings = pd.read_csv(r'C:\Users\P Sai Deekshith\Recommender Systems\ml-100k\ratings.csv')
movies = pd.read_csv(r'C:\Users\P Sai Deekshith\Recommender Systems\ml-100k\movies.csv')
ratings = pd.merge(movies,ratings).drop(['genres','timestamp'],axis=1)
#print(ratings.shape)
ratings.head()


# In[3]:


userRatings = ratings.pivot_table(index=['userId'],columns=['title'],values='rating')
userRatings.head()
#print("Before: ",userRatings.shape)
userRatings = userRatings.dropna(thresh=10, axis=1).fillna(0,axis=1)
userRatings.fillna(0, inplace=True)
userRatings
# print("After: ",userRatings.shape)


# ### Pearson Correlation
# 
# It is used to find similar Items.
# 
# ![PC.png](attachment:PC.png)
# 
# The values given by the formula vary from r = -1 to r = 1, where 1 forms a direct correlation between the two entities (it means a perfect positive correlation) and -1 forms a perfect negative correlation. 
# 
# 
# In our case, a 1 means that the two items are very similar while a -1 means the opposite.
# 

# In[4]:


corrMatrix = userRatings.corr(method='pearson')
corrMatrix.head(100)


# In[5]:


def get_similar(movie_name,rating):
    similar_ratings = corrMatrix[movie_name]*(rating-2.5)
    similar_ratings = similar_ratings.sort_values(ascending=False)
    #print(type(similar_ratings))
    return similar_ratings


# In[6]:


romantic_lover = [("(500) Days of Summer (2009)",5),("Alice in Wonderland (2010)",3),("Aliens (1986)",1),("2001: A Space Odyssey (1968)",2)]
similar_movies = pd.DataFrame()
for movie,rating in romantic_lover:
    similar_movies = similar_movies.append(get_similar(movie,rating),ignore_index = True)

similar_movies.head(10)


# In[7]:


similar_movies.sum().sort_values(ascending=False).head(20)


# In[8]:


action_lover = [("Amazing Spider-Man, The (2012)",5),("Mission: Impossible III (2006)",4),("Toy Story 3 (2010)",2),("2 Fast 2 Furious (Fast and the Furious 2, The) (2003)",4)]
similar_movies = pd.DataFrame()
for movie,rating in action_lover:
    similar_movies = similar_movies.append(get_similar(movie,rating),ignore_index = True)

similar_movies.head(10)
similar_movies.sum().sort_values(ascending=False).head(20)


# # Evalution
# We use RMSE(Root Mean Square Error) to evaluate the recommendations.
# 

# In[9]:


true_ratings = ratings.groupby(ratings['title']).mean()
true_ratings.drop('userId', axis=1, inplace=True)
true_ratings.head()


# In[10]:


last = pd.DataFrame(similar_movies.sum().sort_values(ascending=False).head(20))

last.reset_index(level=0, inplace=True)
last.columns = ['title','predicted_rating']
last


# In[11]:


true_ratings


# In[12]:


final = pd.merge(last,true_ratings,on="title" ,how="inner")
final


# In[14]:


def rmseItemBased():
    rmse = math.sqrt(mean_squared_error(final['predicted_rating'], final['rating']))
    print("The RMSE value is : " + str(rmse))

    return rmse


# In[15]:


#rmseItemBased()


# In[ ]:




