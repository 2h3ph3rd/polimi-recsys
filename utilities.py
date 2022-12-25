#!/usr/bin/env python3

import pandas as pd
import numpy as np
from scipy import sparse
import os
import csv

from src.Recommenders.Recommender_utils import check_matrix


def save_recommendations(rec):

    # Paths to datasets
    base_path = "data"
    interactions_df_path = os.path.join(
        base_path, "interactions_and_impressions.csv")
    items_length_df_path = os.path.join(base_path, "data_ICM_length.csv")
    items_type_df_path = os.path.join(base_path, "data_ICM_type.csv")
    users_df_path = os.path.join(base_path, "data_target_users_test.csv")

    # Read datasets
    dtype = {0: int, 1: int, 2: str, 3: int}
    interactions_df = pd.read_csv(
        filepath_or_buffer=interactions_df_path,
        dtype=dtype,
        keep_default_na=False  # avoid NaN
    )

    items_types_df = pd.read_csv(
        filepath_or_buffer=items_type_df_path, dtype=dtype)
    users_df = pd.read_csv(filepath_or_buffer=users_df_path)

    # Item IDs mapping
    items_ids = items_types_df["item_id"].unique()
    items_ids = np.append(items_ids, interactions_df["item_id"].unique())
    items_ids = np.unique(items_ids)

    items_mapped_ids, items_original_ids = pd.factorize(items_ids)

    item_mapped_id_to_original_id = pd.Series(
        items_original_ids,
        index=items_mapped_ids
    )

    # User IDs mapping
    users_ids = interactions_df["user_id"].sort_values().unique()
    users_mapped_ids, users_original_ids = pd.factorize(users_ids)
    user_original_id_to_mapped_id = pd.Series(
        users_mapped_ids,
        index=users_original_ids
    )

    num_users_to_recommend = users_df['user_id'].shape[0]

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


def numpy_to_sparse(m):
    return sparse.csr_matrix(m)


def save_sparse_matrix(m, filename, base_path='./data'):
    path = os.path.join(base_path, filename)
    sparse.save_npz(path, m)
    print("Saved", filename)


def load_sparse_matrix(filename, base_path='./data'):
    path = os.path.join(base_path, filename)
    print("Loaded", filename)
    return sparse.load_npz(path)


# def linear_scaling_confidence(URM_train, alpha):
#     C = check_matrix(URM_train, format="csr", dtype=np.float32)
#     C.data = 1.0 + alpha * C.data
#     return C
