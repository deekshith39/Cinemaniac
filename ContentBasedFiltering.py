#!/usr/bin/env python
# coding: utf-8

# # Content Based Filtering

# This algorithm recommends movies which are similar to the ones that user liked before.
# 
# For example, if a person has liked the movie “Inception”, then this algorithm will recommend movies that fall under the same genre. 

# ![contentbased.png](attachment:contentbased.png)

# ### Dataset

# This dataset (ml-latest) describes 5-star rating and free-text tagging activity from [MovieLens](http://movielens.org), a movie recommendation service. 
# 
# It contains 22884377 ratings and 586994 tag applications across 34208 movies which was created by 247753 users.

# ## Preprocessing

# First, let's import all the required modules:

# In[1]:


import pandas as pd
import numpy as np
from surprise import accuracy
from math import sqrt
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


movies_df = pd.read_csv(r'C:\Users\P Sai Deekshith\Recommender Systems\ml-100k\movies.csv')

ratings_df = pd.read_csv(r'C:\Users\P Sai Deekshith\Recommender Systems\ml-100k\ratings.csv')


# In[3]:


movies_df.head()


# In[4]:


ratings_df.head()


# Extract the year from Title column to a separate column:

# In[5]:


movies_df['year'] = movies_df.title.str.extract('(\(\d\d\d\d\))',expand=False)
movies_df['year'] = movies_df.year.str.extract('(\d\d\d\d)',expand=False)
movies_df['title'] = movies_df.title.str.replace('(\(\d\d\d\d\))', '')
movies_df['title'] = movies_df['title'].apply(lambda x: x.strip())

movies_df.head(50)


# Splitting the genre:

# In[6]:


movies_df['genres'] = movies_df.genres.str.split('|')
movies_df.head()


# #### On Hot Encoding

# In[7]:


moviesWithGenres_df = movies_df.copy()

for index, row in movies_df.iterrows():
    for genre in row['genres']:
        moviesWithGenres_df.at[index, genre] = 1
        
moviesWithGenres_df = moviesWithGenres_df.fillna(0)
moviesWithGenres_df.head()


# Removing timestamp column from ratings dataframe: 

# In[8]:


ratings_df = ratings_df.drop('timestamp', 1)
ratings_df.head()


# ### User Input: 

# In[9]:


# userInput = [
#             {'title':'Breakfast Club, The', 'rating':5},
#             {'title':'Toy Story', 'rating':3.5},
#             {'title':'Jumanji', 'rating':2},
#             {'title':'Pulp Fiction', 'rating':5},
#             {'title':'Akira', 'rating':4.5}
#          ] 
# inputMovies = pd.DataFrame(userInput)
# inputMovies
userInput = [
            {'title':'Grumpier Old Men', 'rating':3.180094},
            {'title':'Toy Story', 'rating':3.894802},
            {'title':'Jumanji', 'rating':3.221086},
            {'title':'Waiting to Exhale', 'rating':2.879727},
            {'title':'Father of the Bride Part II', 'rating':3.080811}
         ] 
inputMovies = pd.DataFrame(userInput)
inputMovies
# 1	3.894802
# 2	3.221086
# 3	3.180094
# 4	2.879727
# 5	3.080811


# Making a DataFrame out of UserInput by adding movieId to it.

# In[10]:


inputId = movies_df[movies_df['title'].isin(inputMovies['title'].tolist())]
inputMovies = pd.merge(inputId, inputMovies)
inputMovies = inputMovies.drop('genres', 1).drop('year', 1)

inputMovies.head()


# Creating a boolean matrix for the userInput based on the genre:

# In[11]:


userMovies = moviesWithGenres_df[moviesWithGenres_df['movieId'].isin(inputMovies['movieId'].tolist())]
userMovies = userMovies.reset_index(drop=True)
userGenreTable = userMovies.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)

userGenreTable


# We have to learn from input's preference and try to predict the ratings.
# 
# We can do this by using the input's ratings and multiplying them into the input's genre table and summing up the resulting table by column.
# This is called dot product between a matrix and a vector.
# 
# 

# In[12]:


inputMovies['rating']


# In[13]:


userProfile = userGenreTable.transpose().dot(inputMovies['rating'])
userProfile


# Now, we have formed the User's Profile.

# Extracting the genre table from the original dataframe.

# In[14]:


genreTable = moviesWithGenres_df.set_index(moviesWithGenres_df['movieId'])
genreTable = genreTable.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)
genreTable.head()


# Computing the weighted verage of every movie based on the input profile:

# In[15]:


recommendationTable_df = ((genreTable*userProfile).sum(axis=1))/(userProfile.sum())
recommendationTable_df.head()


# In[16]:


recommendationTable_df = recommendationTable_df.sort_values(ascending=False)
recommendationTable_df.head()


# ### The Final Recommendation Table

# In[17]:


movies_df = movies_df.loc[movies_df['movieId'].isin(recommendationTable_df.head(100).keys())]
movies_df.head(20)


# ### Evaluation

# We use RMSE(Root Mean Square Error) to evaluate the recommendations.

# Lower values of RMSE indicate better fit. RMSE is a good measure of how accurately the model predicts the response, and it is the most important criterion for fit if the main purpose of the model is prediction.

# ![RMSE.png](attachment:RMSE.png)

# In[18]:


recommended_rating = pd.DataFrame(recommendationTable_df)
recommended_rating['movieId'] = recommended_rating.index
recommended_rating.reset_index(drop=True, inplace=True)
recommendationTable_df.reset_index(drop=True, inplace=True)
recommended_rating['recommended_ratings'] = recommendationTable_df
recommended_rating.head()


# In[19]:


recommended_rating.drop([0], axis=1, inplace=True)
recommended_rating.head()


# In[20]:


finalEvaluationTable = recommended_rating
finalEvaluationTable['recommended_ratings'] *= 5
finalEvaluationTable.head()


# In[21]:


finalEvaluationTable = finalEvaluationTable.sort_values('movieId')
finalEvaluationTable.head()


# In[22]:


true_ratings = ratings_df.groupby(ratings_df['movieId']).mean()
true_ratings.drop('userId', axis=1, inplace=True)
true_ratings.head()


# In[23]:


recommendedMovieId = finalEvaluationTable['movieId']


# In[24]:


true_ratings['movieId'] = true_ratings.index
true_ratings.reset_index(drop=True, inplace=True)
true_ratings = true_ratings[true_ratings.movieId.isin(recommendedMovieId)]['rating']

true_ratings.head()


# In[25]:


true_ratings = pd.DataFrame(true_ratings)
true_ratings.reset_index(drop=True, inplace=True)
true_ratings.head()


# In[26]:


finalEvaluationTable.head()
finalEvaluationTable.reset_index(drop=True, inplace=True)
finalEvaluationTable['true_ratings'] = true_ratings
#finalEvaluationTable.reset_index(drop=True, inplace=True)
finalEvaluationTable.dropna(inplace=True)
finalEvaluationTable.head()


# In[27]:


#finalEvaluationTable = finalEvaluationTable[finalEvaluationTable.movieId.isin(inputMovies.movieId)]


# In[30]:


def rmseContentBased():   
    rmse = sqrt(mean_squared_error(finalEvaluationTable['true_ratings'], finalEvaluationTable['recommended_ratings']))
    print("The RMSE value is : " + str(rmse))
    return rmse


# In[33]:


#rmseContentBased()


# ### Advantages and Disadvantages of Content-Based Filtering
# 
# ##### Advantages
# 
# -   Learns user's preferences
# -   Highly personalized for the user
# 
# ##### Disadvantages
# 
# -   Doesn't take into account what others think of the item, so low quality item recommendations might happen
# -   Extracting data is not always intuitive
# -   Determining what characteristics of the item the user dislikes or likes is not always obvious
# 

# In[ ]:




