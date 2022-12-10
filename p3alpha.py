#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os

from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender

base_path = "data"

# File path

interactions_df_path = os.path.join(
    base_path, "interactions_and_impressions.csv")
items_length_df_path = os.path.join(base_path, "data_ICM_length.csv")
items_type_df_path = os.path.join(base_path, "data_ICM_type.csv")
users_df_path = os.path.join(base_path, "data_target_users_test.csv")

# Read csv

dtype = {0: int, 1: int, 2: str, 3: int}
interactions_df = pd.read_csv(
    filepath_or_buffer=interactions_df_path, dtype=dtype)
items_length_df = pd.read_csv(filepath_or_buffer=items_length_df_path)
items_types_df = pd.read_csv(filepath_or_buffer=items_type_df_path)
users_df = pd.read_csv(filepath_or_buffer=users_df_path)

# remove only viewed elements
implicit_ratings_df = pd.DataFrame()
implicit_ratings_df = interactions_df.groupby(
    ['user_id', 'item_id'], as_index=False
).max(['data'])

# Calculate implicit ratings

df = implicit_ratings_df
implicit_ratings_df = df[df['data'] == 1].reset_index(drop=True)

# ----- Factorize dataframe -----

users_ids = implicit_ratings_df["user_id"].sort_values().unique()
items_ids = implicit_ratings_df["item_id"].sort_values().unique()


num_users = users_df['user_id'].shape[0]  # Users to recommend
num_total_users = users_ids.shape[0]  # Users of whom we have interactions
num_items = items_ids.shape[0]
print("Found {} users and {} items".format(num_users, num_items))
print("There are {} users with interactions and {} to recommend".format(
    num_total_users, num_users))

items_mapped_ids, items_original_ids = pd.factorize(items_ids)

item_mapped_id_to_original_id = pd.Series(
    items_original_ids, index=items_mapped_ids)
item_original_id_to_mapped_id = pd.Series(
    items_mapped_ids, index=items_original_ids)

users_mapped_ids, users_original_ids = pd.factorize(users_ids)

user_mapped_id_to_original_id = pd.Series(
    users_original_ids, index=users_mapped_ids)
user_original_id_to_mapped_id = pd.Series(
    users_mapped_ids, index=users_original_ids)

# ----- Define URM -----

URM = np.zeros((num_total_users, num_items), dtype=np.int8)

for user_id in users_mapped_ids:
    df = implicit_ratings_df
    user_items = df[df['user_id'] == user_id]['item_id']
    for item_id in user_items:
        item_id = item_original_id_to_mapped_id[item_id]
        URM[user_id, item_id] = 1

# ----- Calculate recommendations -----

rec = P3alphaRecommender(
    URM_train=URM,
    verbose=True
)
