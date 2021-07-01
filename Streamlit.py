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

import streamlit as st
import pandas as pd
import numpy as np
from surprise import accuracy
from math import sqrt
import time
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def get_data():
    movies_df = pd.read_csv(r'C:\Users\P Sai Deekshith\Recommender Systems\ml-100k\movies.csv')

    ratings_df = pd.read_csv(r'C:\Users\P Sai Deekshith\Recommender Systems\ml-100k\ratings.csv')

    movies_df.head()

    ratings_df.head()


    movies_df['year'] = movies_df.title.str.extract('(\(\d\d\d\d\))', expand=False)
    movies_df['year'] = movies_df.year.str.extract('(\d\d\d\d)', expand=False)
    movies_df['title'] = movies_df.title.str.replace('(\(\d\d\d\d\))', '')
    movies_df['title'] = movies_df['title'].apply(lambda x: x.strip())

    movies_df.head(50)

    movies_df['genres'] = movies_df.genres.str.split('|')
    movies_df.head()

    moviesWithGenres_df = movies_df.copy()

    for index, row in movies_df.iterrows():
        for genre in row['genres']:
            moviesWithGenres_df.at[index, genre] = 1

    moviesWithGenres_df = moviesWithGenres_df.fillna(0)
    moviesWithGenres_df.head()

    ratings_df = ratings_df.drop('timestamp', 1)
    ratings_df.head()

    return movies_df, ratings_df, moviesWithGenres_df

def getUserInput(movies_df, ratings_df):

    movie1 = st.selectbox('Movie 1', movies_df.title, key="1")
    rating1 = st.slider('Give a rating', 0.0, 5.0, key="1")

    movie2 = st.selectbox('Movie 2', movies_df.title, key="2")
    rating2 = st.slider('Give a rating', 0.0, 5.0, key="2")

    movie3 = st.selectbox('Movie 3', movies_df.title, key="3")
    rating3 = st.slider('Give a rating', 0.0, 5.0, key="3")

    movie4 = st.selectbox('Movie 4', movies_df.title, key="4")
    rating4 = st.slider('Give a rating', 0.0, 5.0, key="4")

    movie5 = st.selectbox('Movie 5', movies_df.title, key="5")
    rating5 = st.slider('Give a rating', 0.0, 5.0, key="5")


    st.text("")


    userInput = [
        {'title': movie1, 'rating': rating1},
        {'title': movie2, 'rating': rating2},
        {'title': movie3, 'rating': rating3},
        {'title': movie4, 'rating': rating4},
        {'title': movie5, 'rating': rating5}
    ]
    st.subheader("Input Rating Table : ")
    st.text("")
    inputMovies = pd.DataFrame(userInput)
    inputMovies

    st.text("")

    inputId = movies_df[movies_df['title'].isin(inputMovies['title'].tolist())]
    inputMovies = pd.merge(inputId, inputMovies)
    inputMovies = inputMovies.drop('genres', 1).drop('year', 1)

    return inputMovies

def contentBased(inputMovies, moviesWithGenres_df, movies_df):

    userMovies = moviesWithGenres_df[moviesWithGenres_df['movieId'].isin(inputMovies['movieId'].tolist())]
    userMovies = userMovies.reset_index(drop=True)
    userGenreTable = userMovies.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)

    userProfile = userGenreTable.transpose().dot(inputMovies['rating'])

    genreTable = moviesWithGenres_df.set_index(moviesWithGenres_df['movieId'])
    genreTable = genreTable.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)

    recommendationTable_df = ((genreTable * userProfile).sum(axis=1)) / (userProfile.sum())

    recommendationTable_df = recommendationTable_df.sort_values(ascending=False)
    movies_df = movies_df.loc[movies_df['movieId'].isin(recommendationTable_df.head(100).keys())]
    movies_df = movies_df.head(20)
    st.text("")
    st.subheader("Final Recommendation Table : ")
    st.text("")
    st.dataframe(movies_df.title)

def collaborativeBased(inputMovies, movies_df):
    userSubset = ratings_df[ratings_df['movieId'].isin(inputMovies['movieId'].tolist())]

    userSubsetGroup = userSubset.groupby(['userId'])

    userSubsetGroup.get_group(1)

    userSubsetGroup = sorted(userSubsetGroup, key=lambda x: len(x[1]), reverse=True)

    userSubsetGroup = userSubsetGroup[0:100]

    pearsonCorrelationDict = {}

    for name, group in userSubsetGroup:
        group = group.sort_values(by='movieId')
        inputMovies = inputMovies.sort_values(by='movieId')
        nRatings = len(group)
        temp_df = inputMovies[inputMovies['movieId'].isin(group['movieId'].tolist())]
        tempRatingList = temp_df['rating'].tolist()
        tempGroupList = group['rating'].tolist()
        Sxx = sum([i ** 2 for i in tempRatingList]) - pow(sum(tempRatingList), 2) / float(nRatings)
        Syy = sum([i ** 2 for i in tempGroupList]) - pow(sum(tempGroupList), 2) / float(nRatings)
        Sxy = sum(i * j for i, j in zip(tempRatingList, tempGroupList)) - sum(tempRatingList) * sum(
            tempGroupList) / float(nRatings)

        if Sxx != 0 and Syy != 0:
            pearsonCorrelationDict[name] = Sxy / sqrt(Sxx * Syy)
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

    topUsers = pearsonDF.sort_values(by='similarityIndex', ascending=False)[0:50]
    topUsers.head()
    topUsersRating = topUsers.merge(ratings_df, left_on='userId', right_on='userId', how='inner')
    topUsersRating.head()

    topUsersRating['weightedRating'] = topUsersRating['similarityIndex'] * topUsersRating['rating']
    topUsersRating.head()

    tempTopUsersRating = topUsersRating.groupby('movieId').sum()[['similarityIndex', 'weightedRating']]
    tempTopUsersRating.columns = ['sum_similarityIndex', 'sum_weightedRating']
    tempTopUsersRating.head()

    recommendation_df = pd.DataFrame()

    recommendation_df['weighted average recommendation score'] = tempTopUsersRating['sum_weightedRating'] / \
                                                                 tempTopUsersRating['sum_similarityIndex']
    recommendation_df['movieId'] = tempTopUsersRating.index
    recommendation_df.head()

    recommendation_df = recommendation_df.sort_values(by='weighted average recommendation score', ascending=False)
    recommendation_df.head(10)

    st.text("")

    movies_df = movies_df.loc[movies_df['movieId'].isin(recommendation_df.head(10)['movieId'].tolist())]
    st.subheader("Final Recommendation Table : ")
    st.text("")
    st.dataframe(movies_df.title)

if __name__ == '__main__':
    movies_df, ratings_df, moviesWithGenres_df = get_data()
    st.title("Cinemaniac : Movie Recommendation")
    st.text("")
    st.write("Please enter your ratings for five different movies : ")
    st.sidebar.title('Select the Algorithm/Technique')
    options = ['Content Based Filtering', 'Collaborative Filtering']
    st.sidebar.text("")
    selection = st.sidebar.radio('', options)
    userInput = getUserInput(movies_df, ratings_df)
    st.sidebar.text("")
    st.sidebar.text("")
    st.sidebar.write("Content Based algorithm recommends movies which are similar to the ones that user liked before. For example, if a person has liked the movie “Inception”, then this algorithm will recommend movies that fall under the same genre. [read more..](https://developers.google.com/machine-learning/recommendation/content-based/basics)")
    st.sidebar.text("")
    st.sidebar.write("Collaborative Filtering technique we're going to take a look at is Collaborative filtering. It is based on the fact that relationships exist between products and people's interests. [read more..](https://developers.google.com/machine-learning/recommendation/collaborative/basics)")
    if st.button("RECOMMEND"):
        if selection == 'Content Based Filtering':
            contentBased(userInput, moviesWithGenres_df, movies_df)
        if selection == 'Collaborative Filtering':
            collaborativeBased(userInput, movies_df)