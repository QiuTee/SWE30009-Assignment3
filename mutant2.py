import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

data = {
    'user_id': [1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10],
    'item_id': [1, 2, 3, 1, 3, 2, 3, 4, 1, 4, 1, 2, 5, 2, 4, 1, 5, 3, 4, 1, 5, 2, 4],
    'rating': [5, 3, 4, 4, 2, 3, 4, 2, 1, 5, 3, 4, 5, 4, 2, 2, 5, 4, 1, 3, 5, 4, 2]
}

df = pd.DataFrame(data)

# Metamorphic Relations 1 : Consistency under row 
# Permutation of rows of user matrix - product
# df = df.sample(frac=1).reset_index(drop=True)


#Metamorphic Relations 2 : Consistency when ranking ratio changes
scale_factor = 2
df['rating'] = df['rating'] * scale_factor




user_item_matrix = df.pivot(index='user_id', columns='item_id', values='rating').fillna(1)
user_item_matrix_csr = csr_matrix(user_item_matrix.values)

user_similarity = cosine_similarity(user_item_matrix_csr)

def predict_ratings(user_similarity, user_item_matrix):
    mean_user_rating = user_item_matrix.mean(axis=1)
    ratings_diff = (user_item_matrix - mean_user_rating[:, np.newaxis]) 
    pred =  user_similarity.dot(ratings_diff) / np.array([np.abs(user_similarity).sum(axis=1)]).T
    pred = mean_user_rating[:, np.newaxis] + pred
    return pred

pred_ratings = predict_ratings(user_similarity, user_item_matrix.to_numpy())

pred_ratings_df = pd.DataFrame(pred_ratings, index=range(1, user_item_matrix.shape[0] + 1), columns=range(1, user_item_matrix.shape[1] + 1))

print("Predicted Ratings DataFrame:")
print(pred_ratings_df)

def recommend_products(user_id, pred_ratings_df, num_recommendations=2):
    user_ratings = pred_ratings_df.loc[user_id].sort_values(ascending=False)
    return user_ratings.index[:num_recommendations]

recommendations_user1 = recommend_products(1, pred_ratings_df)
recommendations_user2 = recommend_products(5, pred_ratings_df)
print(f"Recommended products for user 1: {recommendations_user1.tolist()}")
print(f"Recommended products for user 5: {recommendations_user2.tolist()}")
