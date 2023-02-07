from labplatform.config import get_config
import os
from tables import *
from experiment.h5Tables import *
from glob import glob
from labplatform.core import Subject
from labplatform.core.Subject import load_cohort

# TODO: ask chao about subject and data modules

# get absolute filepath to cohort files
data_root = get_config("DATA_ROOT")
fp = os.path.join(data_root, "Foo_Test", "FooExp_Pilot_Human_Foo")
files = glob(os.path.join(fp, '*.h5'))
filters = (('_v_name', 'trial_log'), ('<+nyu_id', lambda id: id != 0))
fields = (('<+identifier', 'id'), ('<+sex', 'sex'))
data = extract_data(files, filters, fields)

# define filters and fields for h5Tables.extract files
# filters = (('_v_name', 'trial_log'), ('<+nyu_id', lambda id: id != 0))
# fields = (('<+identifier', 'id'), ('<+sex', 'sex'))
# files = glob(os.path.join(fp, '*.h5'))
# data = h5Tables.extract_data(files, filters, fields)


