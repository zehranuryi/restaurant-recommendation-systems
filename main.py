from fastapi import FastAPI
import uvicorn

from restaurant_recommendation_systems import RestaurantRecommender, RestaurantRecommenderByCuisine

app = FastAPI()

@app.post("/recommended_restaurants_for/{restaurant_name}")
async def read_item(restaurant_name):
    return RestaurantRecommender(restaurant_name = restaurant_name)

@app.post("/recommended_restaurants_by/{cuisine_name}")
async def read_item(cuisine_name):
    return RestaurantRecommenderByCuisine(cuisine_name = cuisine_name)