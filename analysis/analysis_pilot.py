import os
import pathlib
import tables
import logging

DIR = pathlib.Path(os.getcwd())

file = DIR / 'analysis' / 'pilot_data' / '_Pilot_Human_Foo' / 'LocaAccu_0000.h5'

log = logging.getLogger(__name__)

fh = tables.open_file(file, 'r')

