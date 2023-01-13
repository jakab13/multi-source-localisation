from labplatform.config import get_config
import os
from experiment import h5Tables
import tables


data_root = get_config
h5path = os.path.join(get_config('DATA_ROOT'), 'example.h5')
fh = tables.open_file(h5path, 'r')

explorer = DataExplorer()
explorer.configure_traits()

