#!/usr/bin/env python3

import pandas as pd
import numpy as np
from scipy import sparse
import os
import csv

from Recommenders.Recommender_utils import check_matrix


def save_recommendations(rec, users_df, user_original_id_to_mapped_id, item_mapped_id_to_original_id, num_users_to_recommend):
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


def save_sparse_matrix(m, base_path, filename='m.npz'):
    m = sparse.csr_matrix(m)
    path = os.path.join(base_path, filename)
    sparse.save_npz(path, m)
    print("Saved", filename)


def load_sparse_matrix(base_path, filename):
    path = os.path.join(base_path, filename)
    print("Loaded", filename)
    return sparse.load_npz(path)


# def linear_scaling_confidence(URM_train, alpha):
#     C = check_matrix(URM_train, format="csr", dtype=np.float32)
#     C.data = 1.0 + alpha * C.data
#     return C
