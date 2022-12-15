#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Cesare Bernardis
"""

import numpy as np
import scipy.sparse as sps

from sklearn.preprocessing import normalize
from Recommenders.Recommender_utils import check_matrix, similarityMatrixTopK
from Utils.seconds_to_biggest_unit import seconds_to_biggest_unit

from Recommenders.BaseRecommender import BaseRecommender


class HybridRecommender(BaseRecommender):
    """ Hybrid recommender """

    RECOMMENDER_NAME = "HybridRecommender"

    def __init__(self, URM_train, recommenders, verbose=True):
        super(BaseRecommender, self).__init__(URM_train, verbose=verbose)
        self.recommenders = recommenders

    def __str__(self):
        return "P3alpha(alpha={}, min_rating={}, topk={}, implicit={}, normalize_similarity={})".format(self.alpha,
                                                                                                        self.min_rating, self.topK, self.implicit,
                                                                                                        self.normalize_similarity)

    def fit(self, alphas):

         for index in range(1, len(alphas)):

            recommender = ItemKNNSimilarityHybridRecommender(
                URM_train=self.URM_train,
                Similarity_1=recommender.W_sparse,
                Similarity_2=self.similarityRecommenders[index+1].W_sparse,
                verbose=self.verbose
            )

            recommender.fit(
                topKs[index],
                alphas[index]
            )

        self.W_sparse = recommender.W_sparse

