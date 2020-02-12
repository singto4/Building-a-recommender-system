#Dataframe manipulation library.
import pandas as pd

#Reading The Data:
movies_data = 'movies.csv'

#Setting max-row display option to 20 rows.
pd.set_option('display.max_rows', 20)

#Defining additional NaN identifiers.
missing_values = ['na','--','?','-','None','none','non']

#Then we read the data into pandas data frames.
movies_df = pd.read_csv(movies_data, na_values=missing_values)
print(movies_df)
#Data Cleaning and Pre-processing:
#movies_df data set:
#Using regular expressions to find a year stored between parentheses
#We specify the parentheses so we don't conflict with movies that have years in their titles.
movies_df['year'] = movies_df.title.str.extract('(\(\d\d\d\d\))',expand=False)

#Removing the parentheses.
movies_df['year'] = movies_df.year.str.extract('(\d\d\d\d)',expand=False)

#Note that expand=False simply means do not add this adjustment as an additional column to the data frame.
#Removing the years from the 'title' column.
movies_df['title'] = movies_df.title.str.replace('(\(\d\d\d\d\))', '')

#Applying the strip function to get rid of any ending white space characters that may have appeared, using lambda function.
movies_df['title'] = movies_df['title'].apply(lambda x: x.strip())

#Every genre is separated by a | so we simply have to call the split function on |.
movies_df['genres'] = movies_df.genres.str.split('|')

#Filling year NaN values with zeros.
movies_df.year.fillna(0, inplace=True)

#Converting columns year from obj to int16 and movieId from int64 to int32 to save memory.
movies_df.year = movies_df.year.astype('int16')
movies_df.movieId = movies_df.movieId.astype('int32')

#First let's make a copy of the movies_df.
movies_with_genres = movies_df.copy(deep=True)

#Let's iterate through movies_df, then append the movie genres as columns of 1s or 0s.
#1 if that column contains movies in the genre at the present index and 0 if not.
x = []
for index, row in movies_df.iterrows():
    x.append(index)
    for genre in row['genres']:
        movies_with_genres.at[index, genre] = 1 
        
#Filling in the NaN values with 0 to show that a movie doesn't have that column's genre.
movies_with_genres = movies_with_genres.fillna(0)

#Content-Based Recommender System:
#Creating User’s Profile
#So on a scale of 0 to 5, with 0 min and 5 max, see User’s movie ratings below.
User_movie_ratings = [
            {'title':'Predator', 'rating':4.9},
            {'title':'Final Destination', 'rating':4.9},
            {'title':'Heat', 'rating':4},
            {'title':"Beverly Hills Cop", 'rating':3},
            {'title':'Exorcist, The', 'rating':4.8},
            {'title':'Waiting to Exhale', 'rating':3.9},
            {'title':'Jumanji', 'rating':4.5},
            {'title':'Toy Story', 'rating':5.0}
         ] 
User_movie_ratings = pd.DataFrame(User_movie_ratings)

#Let’s add movie Id to User_movie_ratings by extracting the movie IDs from the movies_df data frame above.
#Extracting movie Ids from movies_df and updating User_movie_ratings with movie Ids.
User_movie_Id = movies_df[movies_df['title'].isin(User_movie_ratings['title'])]

#Merging User movie Id and ratings into the User_movie_ratings data frame. 
#This action implicitly merges both data frames by the title column.
User_movie_ratings = pd.merge(User_movie_Id, User_movie_ratings, on='title')

#Dropping information we don't need such as genres
User_movie_ratings = User_movie_ratings.drop(['genres'], 1)

#Learning User’s Profile
#filter the selection by outputing movies that exist in both User_movie_ratings and movies_with_genres.
User_genres_df = movies_with_genres[movies_with_genres.movieId.isin(User_movie_ratings.movieId)]

#Clean User_genres_df
#First, let's reset index to default and drop the existing index.
User_genres_df.reset_index(drop=True, inplace=True)

#Next, let's drop redundant columns(1)
User_genres_df.drop(['movieId', 'title', 'genres', 'year'], axis=1, inplace=True)

#Building User’s Profile
#Let's find the dot product of transpose of User_genres_df by User rating column.
User_profile = User_genres_df.T.dot(User_movie_ratings.rating)

#Deploying The Content-Based Recommender System.
#let's set the index to the movieId.
movies_with_genres = movies_with_genres.set_index(movies_with_genres.movieId)

#Deleting four unnecessary columns.
movies_with_genres.drop(['movieId','title','genres','year'], axis=1, inplace=True)

#Multiply the genres by the weights and then take the weighted average.
recommendation_table_df = (movies_with_genres.dot(User_profile)) / User_profile.sum()

#Let's sort values from great to small
recommendation_table_df.sort_values(ascending=False, inplace=True)

#first we make a copy of the original movies_df
copy = movies_df.copy(deep=True)

#Then we set its index to movieId
copy = copy.set_index('movieId', drop=True)

#Next we enlist the top 20 recommended movieIds we defined above
top_20_index = recommendation_table_df.index[:20].tolist()

#finally we slice these indices from the copied movies df and save in a variable
recommended_movies = copy.loc[top_20_index, :]

print(recommended_movies)