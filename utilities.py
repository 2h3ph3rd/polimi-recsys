#!/usr/bin/env python3

import pandas as pd
import numpy as np
from scipy import sparse
import os
import csv

from Recommenders.Recommender_utils import check_matrix


base_path = "data"
interactions_df_path = os.path.join(
    base_path, "interactions_and_impressions.csv")
items_length_df_path = os.path.join(base_path, "data_ICM_length.csv")
items_type_df_path = os.path.join(base_path, "data_ICM_type.csv")
users_df_path = os.path.join(base_path, "data_target_users_test.csv")

dtype = {0: int, 1: int, 2: str, 3: int}
interactions_df = pd.read_csv(
    filepath_or_buffer=interactions_df_path,
    dtype=dtype,
    keep_default_na=False  # avoid NaN
)
dtype = {0: int, 1: int, 2: int}
items_length_df = pd.read_csv(
    filepath_or_buffer=items_length_df_path, dtype=dtype)
items_types_df = pd.read_csv(
    filepath_or_buffer=items_type_df_path, dtype=dtype)
users_df = pd.read_csv(filepath_or_buffer=users_df_path)

items_ids = items_types_df["item_id"].unique()
items_ids = np.append(items_ids, interactions_df["item_id"].unique())
items_ids = np.unique(items_ids)  # do also sorting

users_ids = interactions_df["user_id"].sort_values().unique()
features_ids = items_types_df["feature_id"].sort_values().unique()

num_users = users_ids.shape[0]
num_items = items_ids.shape[0]
num_items_with_feature = items_ids.shape[0]
num_items_with_interaction = interactions_df["item_id"].unique().shape[0]
num_features = features_ids.shape[0]
num_users_to_recommend = users_df['user_id'].shape[0]

print("Found {} users with interactions and {} to recommend".format(
    num_users, num_users_to_recommend))
print("Found {} items, {} with interactions and {} with {} features".format(
    num_items, num_items_with_interaction, num_items_with_feature, num_features))

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

features_mapped_ids, features_original_ids = pd.factorize(features_ids)

feature_mapped_id_to_original_id = pd.Series(
    features_original_ids, index=features_mapped_ids)
feature_original_id_to_mapped_id = pd.Series(
    features_mapped_ids, index=features_original_ids)


def save_recommendations(rec):
    with open("submission.csv", 'w') as csvfile:

        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['user_id', 'item_list'])

        for user_original_id in users_df['user_id']:

            user_mapped_id = user_original_id_to_mapped_id[user_original_id]
            items_list = rec.recommend(user_mapped_id, 10)

            for i in range(len(items_list)):
                items_list[i] = item_mapped_id_to_original_id[items_list[i]]
            items_list = map(str, items_list)
            items_list = " ".join(items_list)

            csvwriter.writerow([user_original_id, items_list])

            pretty_print_progress(
                user_original_id, num_users_to_recommend, "Saving recommendations")


def pretty_print_progress(current, total, prepend, interval=100):
    if current == total - 1:
        print(" " * 100, end="\r")
        print(prepend, "finished!")
    elif current % interval == 0:
        print("%s %8s of %8s" % (prepend, current, total), end="\r")


def combine(ICM: sparse.csr_matrix, URM: sparse.csr_matrix):
    return sparse.hstack((URM.T, ICM), format='csr')


def save_sparse_matrix(m, filename='m.npz'):
    m = sparse.csr_matrix(m)
    path = os.path.join(base_path, filename)
    sparse.save_npz(path, m)
    print("Saved", filename)


def load_sparse_matrix(filename):
    path = os.path.join(base_path, filename)
    print("Loaded", filename)
    return sparse.load_npz(path)


def linear_scaling_confidence(URM_train, alpha):
    C = check_matrix(URM_train, format="csr", dtype=np.float32)
    C.data = 1.0 + alpha * C.data

    return C
