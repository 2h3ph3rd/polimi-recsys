#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
from scipy import sparse

from Recommenders.GraphBased.RP3betaCBFRecommender import RP3betaCBFRecommender

import utilities


def recommend():

    path = os.path.join(utilities.base_path, "urm.npz")
    URM = sparse.load_npz(path)

    path = os.path.join(utilities.base_path, "icm.npz")
    ICM = sparse.load_npz(path)

    ICM_combined = sparse.hstack((URM.T, ICM), format='csr')

    rec = RP3betaCBFRecommender(
        URM_train=URM,
        ICM_train=ICM_combined,
        verbose=True
    )

    rec.fit(topK=10, alpha=0.5, implicit=True)

    return rec
