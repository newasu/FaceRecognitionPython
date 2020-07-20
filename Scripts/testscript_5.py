
# Add project path to sys
import sys
import pathlib
my_current_path = pathlib.Path(__file__).parent.absolute()
my_root_path = my_current_path.parent
sys.path.insert(0, str(my_root_path))

# Import lib
import pandas as pd
import numpy as np

# Import my own lib
import others.utilities as my_util


fn = '/Users/Wasu/Downloads/exp_2_alg_selmDist_run_0/kf_0_dtf_euclidean_hdn_0d7000000000000001_rc_1e-05_rd_0.npy'

my_model = np.load(fn, encoding='latin1', allow_pickle=True)

my_model = my_util.load_numpy_file(fn)