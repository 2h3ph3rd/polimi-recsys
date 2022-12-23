#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 15/04/19

@author: Maurizio Ferrari Dacrema
"""


######################################################################
##########                                                  ##########
##########                  NON PERSONALIZED                ##########
##########                                                  ##########
######################################################################
from src.Recommenders.NonPersonalizedRecommender import TopPop, Random, GlobalEffects


######################################################################
##########                                                  ##########
##########                  PURE COLLABORATIVE              ##########
##########                                                  ##########
######################################################################
from src.Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from src.Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from src.Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from src.Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender, MultiThreadSLIM_SLIMElasticNetRecommender
from src.Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
from src.Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from src.Recommenders.MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython, MatrixFactorization_FunkSVD_Cython, MatrixFactorization_AsySVD_Cython
from src.Recommenders.MatrixFactorization.PureSVDRecommender import PureSVDRecommender
from src.Recommenders.MatrixFactorization.IALSRecommender import IALSRecommender
from src.Recommenders.MatrixFactorization.NMFRecommender import NMFRecommender
from src.Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender
from src.Recommenders.FactorizationMachines.LightFMRecommender import LightFMCFRecommender
from src.Recommenders.Neural.MultVAERecommender import MultVAERecommender_OptimizerMask as MultVAERecommender


######################################################################
##########                                                  ##########
##########                  PURE CONTENT BASED              ##########
##########                                                  ##########
######################################################################
from src.Recommenders.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from src.Recommenders.KNN.UserKNNCBFRecommender import UserKNNCBFRecommender


######################################################################
##########                                                  ##########
##########                       HYBRID                     ##########
##########                                                  ##########
######################################################################
from src.Recommenders.KNN.ItemKNN_CFCBF_Hybrid_Recommender import ItemKNN_CFCBF_Hybrid_Recommender
from src.Recommenders.KNN.UserKNN_CFCBF_Hybrid_Recommender import UserKNN_CFCBF_Hybrid_Recommender
from src.Recommenders.FactorizationMachines.LightFMRecommender import LightFMUserHybridRecommender, LightFMItemHybridRecommender
