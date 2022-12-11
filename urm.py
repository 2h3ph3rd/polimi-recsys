#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
from scipy import sparse

import utilities


def calculate_URM():
    df = utilities.interactions_df

    ratings_df = df.groupby(
        ['user_id', 'item_id'], as_index=False
    ).max(['data'])

    num_users = utilities.num_users
    num_items = utilities.num_items

    URM = np.zeros((num_users, num_items), dtype=np.int8)

    for user_mapped_id in range(num_users):

        df = ratings_df
        user_original_id = utilities.user_mapped_id_to_original_id[user_mapped_id]
        user_items = df[df['user_id'] == user_original_id]['item_id']

        for item_id in user_items:
            item_id = utilities.item_original_id_to_mapped_id[item_id]
            URM[user_mapped_id, item_id] = 1

        utilities.pretty_print_progress(
            user_mapped_id, num_users, "Calculating URM")

    URM = sparse.csr_matrix(URM)
    path = os.path.join(utilities.base_path, "urm.npz")
    sparse.save_npz(path, URM)
