#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os
import sys

import urm
import icm
import recommend
import utilities

if "urm" in sys.argv:
    urm.calculate_URM()
    exit()
elif "icm" in sys.argv:
    icm.calculate_ICM()
    exit()


rec = recommend.recommend()

utilities.save_recommendations(rec)
