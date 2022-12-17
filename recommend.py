#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
from scipy import sparse

from Recommenders.Implicit.FeatureCombinedImplicitALSRecommender import FeatureCombinedImplicitALSRecommender
from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
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

    recommenders.append(P3alpha(URM))
    recommenders.append(P3alphaCBF(URM, ICM_combined))
    recommenders.append(RP3betaCBF(URM, ICM_combined))

    # recommenders.append(ials_recommender(URM, ICM_combined))
    # recommenders.append(slim_recommender(URM, ICM_combined))
    # recommenders.append(p3alpha_recommender(URM, ICM_combined))

    rec = GeneralizedMergedHybridRecommender(
        URM_train=URM,
        recommenders=recommenders,
        verbose=True
    )
    rec.fit(alphas=[0.2, 0.4, 0.7])
    return rec


def ials_recommender(URM, ICM):
    rec = FeatureCombinedImplicitALSRecommender(
        URM_train=URM,
        ICM_train=ICM,
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


def P3alpha(URM):
    rec = P3alphaRecommender(
        URM_train=URM,
        verbose=True,
    )
    rec.fit(topK=100, alpha=0.7)
    return rec


def P3alphaCBF(URM, ICM_combined):
    rec = P3alphaCBFRecommender(
        URM_train=URM,
        ICM_train=ICM_combined,
        verbose=True,
    )
    rec.fit(topK=100, alpha=0.7)
    return rec


def RP3betaCBF(URM, ICM_combined):
    rec = RP3betaCBFRecommender(
        URM_train=URM,
        ICM_train=ICM_combined,
        verbose=True,
    )
    rec.fit(topK=100, alpha=0.7, beta=0.3)
    return rec


def slim_recommender(URM, ICM_combined):
    rec = MultiThreadSLIM_ElasticNet(
        URM_train=ICM_combined.T,
        verbose=True
    )
    rec.fit(
        alpha=0.7,
        l1_ratio=0.1,
        topK=100,
        workers=6
    )
    rec.URM_train = URM
    return rec
