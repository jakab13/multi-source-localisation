from labplatform.config import get_config
import os
from experiment import h5Tables
from glob import glob

# get absolute filepath to cohort files
data_root = get_config("DATA_ROOT")
fp = os.path.join(data_root, "example\\example_control_Mouse_C1")

# define filters and fields for h5Tables.extract files
filters = (('_v_name', 'trial_log'), ('<+nyu_id', lambda id: id != 0),)
fields = (('<+identifier', 'id'), ('<+sex', 'sex'))

# get all files from cohort directory
files = glob(os.path.join(fp, '*.h5'))

# extract files
data = h5Tables.extract_data(files, filters, fields)
