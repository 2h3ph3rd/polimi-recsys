#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
from scipy import sparse

from Recommenders.GraphBased.RP3betaCBFRecommender import RP3betaCBFRecommender

import utilities


def recommend():

    URM = utilities.load_sparse_matrix("urm.npz")
    ICM = utilities.load_sparse_matrix("icm.npz")

    ICM_combined = sparse.hstack((URM.T, ICM), format='csr')

    rec = RP3betaCBFRecommender(
        URM_train=URM,
        ICM_train=ICM_combined,
        verbose=True,
    )

    rec.fit(topK=10, alpha=0.7, beta=0.3)

    return rec
