from labplatform.core.Subject import Subject, SubjectList
from labplatform.config import get_config
import os
import pathlib


name = "Test"
group = "Pilot"
sex = "M"
cohort = "Vertical"
subject = Subject(name=f"sub{name}",
                  group=group,
                  species="Human",
                  sex=sex,
                  cohort=cohort)
cohort_fp = pathlib.Path(os.path.join(get_config("SUBJECT_ROOT"), cohort))
if not pathlib.Path.is_dir(cohort_fp):
    os.mkdir(path=os.path.join(get_config("SUBJECT_ROOT"), cohort))
h5fp = os.path.join(get_config("SUBJECT_ROOT"), cohort, name)
subject.add_subject_to_h5file(file=h5fp)