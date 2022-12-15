#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
from scipy import sparse

from Recommenders.Implicit.FeatureCombinedImplicitALSRecommender import FeatureCombinedImplicitALSRecommender
from Recommenders.GraphBased.RP3betaCBFRecommender import RP3betaCBFRecommender
from Recommenders.GraphBased.P3alphaCBFRecommender import P3alphaCBFRecommender
from Recommenders.Hybrid.GeneralizedMergedHybridRecommender import GeneralizedMergedHybridRecommender
from Recommenders.SLIM_ElasticNet.SLIMElasticNetRecommender import MultiThreadSLIM_ElasticNet

import utilities


def recommend():

    URM = utilities.load_sparse_matrix("urm.npz")
    ICM = utilities.load_sparse_matrix("icm.npz")

    ICM_combined = sparse.hstack((URM.T, ICM), format='csr')

    recommenders = []

    # recommenders.append(ials_recommender(URM, ICM_combined))
    recommenders.append(rp3beta_recommender(URM, ICM_combined))
    recommenders.append(slim_recommender(URM, ICM_combined))
    # recommenders.append(p3alpha_recommender(URM, ICM_combined))

    rec = GeneralizedMergedHybridRecommender(
        URM_train=URM,
        recommenders=recommenders,
        verbose=True
    )

    rec.fit(
        alphas=[
            0.7,
            0.3,
            # 0.3
        ]
    )

    return rec


def ials_recommender(URM, ICM_combined):
    rec = FeatureCombinedImplicitALSRecommender(
        URM_train=URM,
        ICM_train=ICM_combined,
        verbose=True
    )

    rec.fit(
        factors=398,
        regularization=0.01,
        use_gpu=False,
        iterations=100,
        num_threads=6,
        confidence_scaling=utilities.linear_scaling_confidence,
        **{
            'URM': {"alpha": 42.07374324671451},
            'ICM': {"alpha": 41.72067133975204}
        }
    )

    return rec


def p3alpha_recommender(URM, ICM_combined):
    rec = P3alphaCBFRecommender(
        URM_train=URM,
        ICM_train=ICM_combined,
        verbose=True,
    )
    rec.fit(topK=100, alpha=0.7)
    return rec


def rp3beta_recommender(URM, ICM_combined):
    rec = RP3betaCBFRecommender(
        URM_train=URM,
        ICM_train=ICM_combined,
        verbose=True,
    )
    rec.fit(topK=100, alpha=0.7, beta=0.3)
    return rec


def slim_recommender(URM, ICM_combined):
    rec = SLIM_recommender = MultiThreadSLIM_ElasticNet(
        URM_train=ICM_combined.T,
        verbose=True
    )
    SLIM_recommender.fit(
        alpha=1.0,
        l1_ratio=0.1,
        topK=100,
        workers=6
    )
    SLIM_recommender.URM_train = URM
    return rec
