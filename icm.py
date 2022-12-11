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

    ICM = np.zeros((num_items, num_features), dtype=np.int8)

    for i in df.index:

        item_id = df.loc[i, 'item_id']
        feature_id = df.loc[i, 'feature_id']
        item_id = utilities.item_original_id_to_mapped_id[item_id]
        feature_id = utilities.feature_original_id_to_mapped_id[feature_id]
        ICM[item_id, feature_id] = 1

        utilities.pretty_print_progress(
            i, df.shape[0], "Calculating ICM")

    ICM = sparse.csr_matrix(ICM)
    path = os.path.join(utilities.base_path, "icm.npz")
    sparse.save_npz(path, ICM)
