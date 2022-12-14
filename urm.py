#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
from scipy import sparse

import utilities


def calculate_URM():
    use_custom_interactions_weights = True
    use_custom_impressions_weights = False

    df = utilities.interactions_df.copy()

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
            item_original_id = user_items.loc[i, 'item_id']

            if use_custom_interactions_weights:
                user_interactions = user_items.loc[i, 'data']
                weight = custom_interactions_weights(
                    item_original_id, user_interactions)
            else:
                weight = 1

            item_mapped_id = utilities.item_original_id_to_mapped_id[item_original_id]
            URM[user_mapped_id, item_mapped_id] = weight

        utilities.pretty_print_progress(
            user_mapped_id, num_users, "Calculating URM")

    if use_custom_interactions_weights:
        utilities.save_sparse_matrix(URM, 'urm_custom_interactions.npz')
    else:
        utilities.save_sparse_matrix(URM, 'urm_base.npz')

    if use_custom_impressions_weights:
        URM = custom_impressions_weights(URM)
        utilities.save_sparse_matrix(URM, 'urm_custom_impressions_weights.npz')

    utilities.save_sparse_matrix(URM, 'urm.npz')


def custom_interactions_weights(item_id, user_interactions):
    weight = 0
    df = utilities.items_length_df.copy()

    try:
        item_id = utilities.item_original_id_to_mapped_id[item_id]
        item_length = df.loc[item_id, 'data']
    except KeyError:
        item_length = 0

    # tv series have more than one element
    if item_length > 1:
        if user_interactions == 0:
            weight = 0.5
        elif user_interactions < item_length * 0.4:
            weight = 0.75
        # elif user_interactions < item_length * 0.7:
        else:
            weight = 1
    # films have only one element
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

    return weight


def custom_impressions_weights(URM):
    df = utilities.interactions_df.copy()
    for id in df.index:
        row = df.iloc[id]
        user_id = row['user_id']
        impressions = row['impressions'].split(',')

        for item_id in impressions:
            if item_id == '':
                continue
            item_id = int(item_id)
            item_id = utilities.item_original_id_to_mapped_id[item_id]
            user_id = utilities.user_original_id_to_mapped_id[user_id]
            if URM[user_id, item_id] == 0:
                URM[user_id, item_id] = 0.5

        utilities.pretty_print_progress(
            id, df.shape[0], "Calculating custom impressions weights", interval=1000)
    return URM
