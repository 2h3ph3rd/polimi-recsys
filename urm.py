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
    ).sum(['data'])

    num_users = utilities.num_users
    num_items = utilities.num_items

    URM = np.zeros((num_users, num_items), dtype=np.float16)

    for user_mapped_id in range(num_users):

        df = ratings_df
        user_original_id = utilities.user_mapped_id_to_original_id[user_mapped_id]
        user_items = df[df['user_id'] == user_original_id]

        for i in user_items.index:
            item_id = user_items.loc[i, 'item_id']
            user_interactions = user_items.loc[i, 'data']

            df = utilities.items_length_df
            item_length = df[df['item_id'] == item_id]['data']

            weight = 0
            if item_length.shape[0] == 0:
                if user_interactions == 0:
                    weight = 0.5
                else:
                    weight = 1
            else:
                item_length = item_length.iloc[0]
                if item_length > 1:
                    if user_interactions == 0:
                        weight = 0.5
                    elif user_interactions < item_length * 0.4:
                        weight = 0.75
                    # elif user_interactions < item_length * 0.7:
                    else:
                        weight = 1
                # films has only one element
                elif item_length == 1:
                    if user_interactions == 0:
                        weight = 0.5
                    elif user_interactions == 1:
                        weight = 0.75
                    else:
                        weight = 1
                # it is not known if this item is a film or a tv series
                else:
                    if user_interactions == 0:
                        weight = 0.5
                    else:
                        weight = 1

            item_id = utilities.item_original_id_to_mapped_id[item_id]
            URM[user_mapped_id, item_id] = weight

        utilities.pretty_print_progress(
            user_mapped_id, num_users, "Calculating URM")

    URM = sparse.csr_matrix(URM)
    path = os.path.join(utilities.base_path, "urm.npz")
    sparse.save_npz(path, URM)
