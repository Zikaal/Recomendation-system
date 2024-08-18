from surprise import Reader,Dataset
import pandas as pd
from surprise import accuracy
from surprise import SVD
from surprise.model_selection import GridSearchCV
from surprise.model_selection import train_test_split

def get_top_n_recommendations(predictions,n=10):
    top_n = {}

    for uid, iid, true_r, est, _ in predictions:
        if uid not in  top_n:
            top_n[uid] = []
        top_n[uid].append((iid,est))

    for uid, user_rating in top_n.items():
        user_rating.sort(key=lambda x:x[1], reverse=True)
        top_n[uid] = user_rating[:n]
    return top_n

def get_user_recommendations(user_id, top_n):
    if user_id in top_n:
        return top_n[user_id]
    else:
        return []

# Load and cleaning data
data = Dataset.load_builtin('ml-100k')

raw_data = data.raw_ratings

df = pd.DataFrame(raw_data, columns=['user_id','item_id','rating','timestamp'])

duplicates = df.duplicated()
print(f'Num of duplicates: {duplicates.sum()}')

df = df.drop_duplicates()

missing_values = df.isnull().sum()
print(f'Missing values in each columns:\n{missing_values}')

df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

print(df.head(30))

#Training models

reader = Reader(rating_scale=(1,5))
data = Dataset.load_from_df(df[['user_id', 'item_id', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.25)

#SVD model

param_grid = {
    'n_factors': [20, 50, 100],
    'reg_all': [0.02, 0.05, 0.1],
    'lr_all': [0.005, 0.01, 0.02]
}

gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3)
gs.fit(data)

best_algo = gs.best_estimator['rmse']
best_algo.fit(trainset)

predictions_best = best_algo.test(testset)
accuracy_rmse_svd= accuracy.rmse(predictions_best)

all_predictions = best_algo.test(trainset.build_testset())

top_n = get_top_n_recommendations(all_predictions, n=10)

user_id = '196'
user_recommendations = get_user_recommendations(user_id, top_n)

for movie_id, rating in user_recommendations:
    print(f'Movie: {movie_id}, Prediction rating: {rating}')