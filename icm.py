#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
from scipy import sparse

import utilities


def calculate_ICM():
    df = utilities.items_types_df

    num_items = utilities.num_items
    num_features = utilities.num_features

    ICM = np.zeros((num_items, num_features + 2), dtype=np.int8)

    for i in df.index:

        item_id = df.loc[i, 'item_id']
        feature_id = df.loc[i, 'feature_id']
        item_id = utilities.item_original_id_to_mapped_id[item_id]
        feature_id = utilities.feature_original_id_to_mapped_id[feature_id]
        ICM[item_id, feature_id] = 1

        utilities.pretty_print_progress(
            i, df.shape[0], "Calculating ICM with types")

    df = utilities.items_length_df

    for i in df.index:

        item_id = df.loc[i, 'item_id']
        length = df.loc[i, 'data']
        item_id = utilities.item_original_id_to_mapped_id[item_id]

        if length == 0:
            continue
        elif length == 1:
            feature_id = num_features
        else:
            feature_id = num_features + 1

        ICM[item_id, feature_id] = 1

        utilities.pretty_print_progress(
            i, df.shape[0], "Calculating ICM with items length")

    utilities.save_sparse_matrix(ICM, "icm.npz")
