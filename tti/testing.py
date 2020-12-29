import os

import numpy as np
import pandas as pd
from tqdm.notebook import trange
from tqdm import tqdm

from tti_explorer import config, utils
from tti_explorer.case import simulate_case, CaseFactors
from tti_explorer.contacts import EmpiricalContactsSimulator
from tti_explorer.strategies import TTIFlowModel, RETURN_KEYS

def print_doc(func):
    print(func.__doc__)

rng = np.random.RandomState(0)
case_config = config.get_case_config("delve")
print(case_config)