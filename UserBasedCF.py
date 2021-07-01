#!/usr/bin/env python
# coding: utf-8

# # Collaborative Filtering

# The technique we're going to take a look at is Collaborative filtering.It is based on the fact that relationships exist between products and people's interests. Many recommendation systems use collaborative filtering to find these relationships and to give an accurate recommendation of a product that the user might like or be interested in.
# 

# As hinted by its alternate name, this technique uses other users to recommend items to the input user.

# ![collaborativefiltering.png](attachment:collaborativefiltering.png)

# ### Dataset

# This dataset (ml-latest) describes 5-star rating and free-text tagging activity from [MovieLens](http://movielens.org), a movie recommendation service. 
# 
# It contains 22884377 ratings and 586994 tag applications across 34208 movies which was created by 247753 users.

# ## Preprocessing
# 

# In[1]:


import pandas as pd
from math import sqrt
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')


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

movies_df.head()


# Splitting the genre:

# In[6]:


movies_df = movies_df.drop('genres', 1)
movies_df.head()


# In[7]:


ratings_df = ratings_df.drop('timestamp', 1)
ratings_df.head()


# ### User Based Recommendation System

# User-Based Collaborative Filtering is a technique used to predict the items that a user might like on the basis of ratings given to that item by the other users who have similar taste with that of the target user.
# 
# The process for creating a User Based recommendation system is as follows:
# 
# -   Select a user with the movies the user has watched
# -   Based on his rating to movies, find the top X neighbours 
# -   Get the watched movie record of the user for each neighbour.
# -   Calculate a similarity score using some formula
# -   Recommend the items with the highest score

# #### User Input

# In[8]:


userInput = [
            {'title':'Grumpier Old Men', 'rating':3.180094},
            {'title':'Toy Story', 'rating':3.894802},
            {'title':'Jumanji', 'rating':3.221086},
            {'title':'Waiting to Exhale', 'rating':2.879727},
            {'title':'Father of the Bride Part II', 'rating':3.080811}
            ] 
inputMovies = pd.DataFrame(userInput)
inputMovies


# #### Add movieId to input user
# 
# With the input complete, let's extract the input movies's ID's from the movies dataframe and add them into it.
# 
# We can achieve this by first filtering out the rows that contain the input movies' title and then merging this subset with the input dataframe.

# In[9]:


inputId = movies_df[movies_df['title'].isin(inputMovies['title'].tolist())]
inputMovies = pd.merge(inputId, inputMovies)
inputMovies = inputMovies.drop('year', 1)

inputMovies


# #### The users who has rated the same movies
# 
# Now with the movie ID's in our input, we can now get the subset of users that have watched and reviewed the movies in our input.
# 

# In[10]:


userSubset = ratings_df[ratings_df['movieId'].isin(inputMovies['movieId'].tolist())]
userSubset.head()


# We now group up the rows by user ID.
# 

# In[11]:


userSubsetGroup = userSubset.groupby(['userId'])


# e.g. the one with userID=1
# 

# In[12]:


userSubsetGroup.get_group(1)


# Let's also sort these groups so the users that share the most movies in common with the input have higher priority which provides a richer recommendation since we won't go through every single user.
# 

# In[13]:


userSubsetGroup = sorted(userSubsetGroup,  key=lambda x: len(x[1]), reverse=True)


# In[14]:


userSubsetGroup[0:3]


# #### Similarity of users to input user
# 
# Next, we are going to compare all users to our specified user and find the one that is most similar.  
# we're going to find out how similar each user is to the input through the **Pearson Correlation Coefficient**. It is used to measure the strength of a linear association between two variables. The formula for finding this coefficient between sets X and Y with N values can be seen in the image below. 
# 
# Why Pearson Correlation?
# 
# Pearson correlation is invariant to scaling, i.e. multiplying all elements by a nonzero constant or adding any constant to all elements. For example, if you have two vectors X and Y,then, pearson(X, Y) == pearson(X, 2 * Y + 3). This is a pretty important property in recommendation systems because for example two users might rate two series of items totally different in terms of absolute rates, but they would be similar users (i.e. with similar ideas) with similar rates in various scales .
# 
# ![PC.png](attachment:PC.png)
# 
# The values given by the formula vary from r = -1 to r = 1, where 1 forms a direct correlation between the two entities (it means a perfect positive correlation) and -1 forms a perfect negative correlation. 
# 
# 
# In our case, a 1 means that the two users have similar tastes while a -1 means the opposite.
# 

# We will select a subset of users to iterate through. This limit is imposed because we don't want to waste too much time going through every single user.
# 

# In[15]:


userSubsetGroup = userSubsetGroup[0:100]


# In[16]:


pearsonCorrelationDict = {}

for name, group in userSubsetGroup:
    group = group.sort_values(by='movieId')
    inputMovies = inputMovies.sort_values(by='movieId')
    nRatings = len(group)
    temp_df = inputMovies[inputMovies['movieId'].isin(group['movieId'].tolist())]
    tempRatingList = temp_df['rating'].tolist()
    tempGroupList = group['rating'].tolist()
    Sxx = sum([i**2 for i in tempRatingList]) - pow(sum(tempRatingList),2)/float(nRatings)
    Syy = sum([i**2 for i in tempGroupList]) - pow(sum(tempGroupList),2)/float(nRatings)
    Sxy = sum( i*j for i, j in zip(tempRatingList, tempGroupList)) - sum(tempRatingList)*sum(tempGroupList)/float(nRatings)
    
    if Sxx != 0 and Syy != 0:
        pearsonCorrelationDict[name] = Sxy/sqrt(Sxx*Syy)
    else:
        pearsonCorrelationDict[name] = 0


# In[17]:


pearsonCorrelationDict.items()


# In[18]:


pearsonDF = pd.DataFrame.from_dict(pearsonCorrelationDict, orient='index')
pearsonDF.columns = ['similarityIndex']
pearsonDF['userId'] = pearsonDF.index
pearsonDF.index = range(len(pearsonDF))
pearsonDF.head()


# #### The top x similar users to input user
# 
# Now let's get the top 50 users that are most similar to the input.
# 

# In[19]:


topUsers=pearsonDF.sort_values(by='similarityIndex', ascending=False)[0:50]
topUsers.head()


# Now, let's start recommending movies to the input user.
# 
# #### Rating of selected users to all movies
# 
# We're going to do this by taking the weighted average of the ratings of the movies using the Pearson Correlation as the weight. But to do this, we first need to get the movies watched by the users in our pearsonDF from the ratings dataframe and then store their correlation in a new column called similarityIndex. This is achieved below by merging of these two tables.
# 

# In[20]:


topUsersRating=topUsers.merge(ratings_df, left_on='userId', right_on='userId', how='inner')
topUsersRating.head()


# Now all we need to do is simply multiply the movie rating by its weight (The similarity index), then sum up the new ratings and divide it by the sum of the weights.
# 

# We can easily do this by simply multiplying two columns, then grouping up the dataframe by movieId

# In[21]:


topUsersRating['weightedRating'] = topUsersRating['similarityIndex']*topUsersRating['rating']
topUsersRating.head()


# In[22]:


tempTopUsersRating = topUsersRating.groupby('movieId').sum()[['similarityIndex','weightedRating']]
tempTopUsersRating.columns = ['sum_similarityIndex','sum_weightedRating']
tempTopUsersRating.head()


# In[23]:


recommendation_df = pd.DataFrame()

recommendation_df['weighted average recommendation score'] = tempTopUsersRating['sum_weightedRating']/tempTopUsersRating['sum_similarityIndex']
recommendation_df['movieId'] = tempTopUsersRating.index
recommendation_df.head()


# In[24]:


recommendation_df = recommendation_df.sort_values(by='weighted average recommendation score', ascending=False)
recommendation_df.head(10)


# In[25]:


movies_df.loc[movies_df['movieId'].isin(recommendation_df.head(10)['movieId'].tolist())]


# ### Evaluation

# We use RMSE(Root Mean Square Error) to evaluate the recommendations.

# Lower values of RMSE indicate better fit. RMSE is a good measure of how accurately the model predicts the response, and it is the most important criterion for fit if the main purpose of the model is prediction.

# ![RMSE.png](attachment:RMSE.png)

# In[26]:


recommendation_df.head()


# In[27]:


rec_df = recommendation_df[['weighted average recommendation score', 'movieId']]
rec_df.reset_index(drop=True, inplace=True)
recommendedMovieId = rec_df['movieId']
rec_df.head()


# In[28]:


true_ratings = ratings_df.groupby(ratings_df['movieId']).mean()
true_ratings.drop('userId', axis=1, inplace=True)
true_ratings.head()


# In[29]:


true_ratings['movieId'] = true_ratings.index
true_ratings.reset_index(drop=True, inplace=True)
true_ratings = true_ratings[true_ratings.movieId.isin(recommendedMovieId)]

true_ratings.head()


# In[30]:


true_ratings = pd.DataFrame(true_ratings)
true_ratings.reset_index(drop=True, inplace=True)
true_ratings.head()


# In[34]:


rec_df = rec_df.sort_values('movieId')
rec_df.reset_index(drop=True, inplace=True)
rec_df['true_ratings'] = true_ratings['rating']
rec_df.head()


# In[37]:


#rec_df = rec_df[rec_df.movieId.isin(inputMovies.movieId)]
def rmseUserBased():
    rmse = sqrt(mean_squared_error(rec_df['true_ratings'], rec_df['weighted average recommendation score']))
    print("The RMSE value is : " + str(rmse))
    return rmse


# In[38]:


#rmseUserBased()


# ### Advantages and Disadvantages of Collaborative Filtering
# 
# ##### Advantages
# 
# -   Takes other user's ratings into consideration
# -   Doesn't need to study or extract information from the recommended item
# -   Adapts to the user's interests which might change over time
# 
# ##### Disadvantages
# 
# -   Approximation function can be slow
# -   There might be a low of amount of users to approximate
# -   Privacy issues when trying to learn the user's preferences
# 

# In[ ]:





# In[ ]:




