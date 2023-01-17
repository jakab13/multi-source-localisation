from labplatform.config import get_config
import os
from experiment import h5Tables
from glob import glob
from labplatform.core import Subject
from labplatform.core.Subject import load_cohort

# TODO: ask chao about subject and data modules

# get absolute filepath to cohort files
data_root = get_config("DATA_ROOT")
fp = os.path.join(data_root, "Foo_Test\\_control_Human_Foo")

# define filters and fields for h5Tables.extract files
filters = (('_v_name', 'trial_log'), ('nyu_id', lambda id: id != 0))
fields = (('identifier', 'id'), ('sex', 'sex'))

# get all files from data directory
files = glob(os.path.join(fp, '*.h5'))
# extract data files
data = h5Tables.extract_data(files, filters, fields)

