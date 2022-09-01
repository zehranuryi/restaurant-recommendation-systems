# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 10:01:29 2022

@author: zehra
"""

import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer


main_data = pd.read_csv("zomato.csv")

print(main_data.columns)

zomato = main_data.drop(['url', 'votes', 'rest_type', 'online_order', 'book_table', 'dish_liked', 'approx_cost(for two people)', 'reviews_list', 'menu_item', 'listed_in(type)', 'listed_in(city)'],axis=1)
zomato = zomato[['name', 'rate', 'cuisines', 'location', 'address', 'phone']]

zomato.duplicated().any()
zomato.drop_duplicates(subset=['name'], keep='last', inplace = True)

zomato.isnull().any()
zomato.dropna(inplace=True)

rate_control = zomato.rate.value_counts()

zomato = zomato.loc[zomato.rate !='NEW']
zomato = zomato.loc[zomato.rate !='-'].reset_index(drop=True)
zomato['rate'] = zomato['rate'].str.replace('/5', '').str.strip().astype('float')

zomato.info()
rate_control = zomato.rate.value_counts()

restaurants = list(zomato['name'].unique())
zomato['mean_rating'] = 0

for i in range(len(restaurants)):
    zomato['mean_rating'][zomato['name'] == restaurants[i]] = zomato['rate'][zomato['name'] == restaurants[i]].mean()

zomato['mean_rating'] = round(zomato['mean_rating'], 2)

tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
tfidf_matrix = tfidf.fit_transform( zomato['cuisines'])

cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

def RestaurantRecommender(restaurant_name, cosine_similarities = cosine_similarities):
    
    restaurant_name_list = []
    restaurant_id = zomato[zomato['name'] == restaurant_name].index[0]
    score_series = pd.Series(cosine_similarities[restaurant_id]).sort_values(ascending=False)
    top30_indexes = list(score_series.iloc[0:31].index)
    for each in top30_indexes:
        if restaurant_name != (zomato['name'])[each]:
            restaurant_name_list.append(list(zomato['name'])[each])
        else: pass
    df_new = pd.DataFrame(columns=['name', 'cuisines', 'location', 'mean_rating', 'address', 'phone'])
    for each in restaurant_name_list:
        df_new = df_new.append(pd.DataFrame(zomato[['name', 'cuisines', 'location', 'mean_rating', 'address', 'phone']][zomato.name == each].sample()))
    df_new = df_new.drop_duplicates(subset=['name', 'cuisines', 'location', 'mean_rating', 'address', 'phone'], keep=False)
    df_new = df_new.sort_values(by='mean_rating', ascending=False).head(10)
    
    print('TOP %s RESTAURANTS LIKE %s WITH SIMILAR CUISINES AND LOCATION: ' % (str(len(df_new)), restaurant_name))
    
    return df_new

def RestaurantRecommenderByCuisine(cuisine_name, cosine_similarities = cosine_similarities):
    
    restaurant_name_list = []
    restaurant_id = zomato[zomato['cuisines'] == cuisine_name].index[0]
    score_series = pd.Series(cosine_similarities[restaurant_id]).sort_values(ascending=False)
    top30_indexes = list(score_series.iloc[0:31].index)
    for each in top30_indexes:
        restaurant_name_list.append(list(zomato['name'])[each])
    df_new = pd.DataFrame(columns=['name', 'cuisines', 'location', 'mean_rating', 'address', 'phone'])
    for each in restaurant_name_list:
        df_new = df_new.append(pd.DataFrame(zomato[['name', 'cuisines', 'location', 'mean_rating', 'address', 'phone']][zomato.name == each].sample()))
    df_new = df_new.drop_duplicates(subset=['name', 'cuisines', 'location', 'mean_rating', 'address', 'phone'], keep=False)
    df_new = df_new.sort_values(by='mean_rating', ascending=False).head(10)
    
    return df_new
