import numpy as np
import pandas as pd


ratings_data = pd.read_csv('u.csv', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])


item_data = pd.read_csv('item.csv', sep='|', encoding='latin-1', header=None)
item_data.columns = ['item_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown', 'Action',
                     'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                     'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']


user_params = np.zeros(24)


for _, row in ratings_data.iterrows():
    user_id = row['user_id']
    item_id = row['item_id']
    rating = row['rating']


    movie_features = item_data.loc[item_data['item_id'] == item_id, 'Action':'Western'].values[0]


    user_params += rating * np.array(movie_features)


user_params /= len(ratings_data)


print(user_params)
