from labplatform.core.Subject import Subject, SubjectList
from labplatform.config import get_config
import os
import logging
from experiment.Numerosity_Judgement import NumerosityJudgementExperiment
from experiment.Spatial_Unmasking import SpatialUnmaskingExperiment
from experiment.Localization_Accuracy import LocalizationAccuracyExperiment


log = logging.getLogger(__name__)
log.setLevel(logging.WARNING)
# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.WARNING)
# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# add formatter to ch
ch.setFormatter(formatter)
# add ch to logger
log.addHandler(ch)


def setup_experiment():
    exp_type = input("la, su or nm?").lower()
    subject = input("Enter subject id: ").lower()
    group = input("Vertical or horizontal? Insert v or h").lower()
    sex = input("m or f? ").upper()
    cohort = input("Pilot or Test cohort? Insert p or t").lower()
    experimenter = input("Enter your name: ").lower()

    #NumerosityJudgementExperiment.set_plane(plane=group)
    #SpatialUnmaskingExperiment.set_plane(plane=group)
    #LocalizationAccuracyExperiment.set_plane(plane=group)

    try:
        subject = Subject(name=f"sub{subject}",
                          group=group,
                          species="Human",
                          sex=sex,
                          cohort=cohort)
        subject.data_path = os.path.join(get_config("DATA_ROOT"), f"sub{subject}.h5")
        subject.add_subject_to_h5file(os.path.join(get_config("SUBJECT_ROOT"), f"sub{subject}.h5"))
    except ValueError:
        # read the subject information
        sl = SubjectList(file_path=os.path.join(get_config("SUBJECT_ROOT"), f"sub{subject}.h5"))
        sl.read_from_h5file()
        subject = sl.subjects[0]
        subject.data_path = os.path.join(get_config("DATA_ROOT"), f"sub{subject}.h5")

    if exp_type == "su":
        exp = SpatialUnmaskingExperiment(subject=subject, experimenter=experimenter, plane=group)
        exp.plane = group
        return exp
    elif exp_type == "nm":
        exp = NumerosityJudgementExperiment(subject=subject, experimenter=experimenter)
        exp.plane = group
        return exp
    elif exp_type == "la":
        exp = LocalizationAccuracyExperiment(subject=subject, experimenter=experimenter)
        exp.plane = group
        return exp
    else:
        log.warning("Paradigm not found, aborting ...")


def run_experiment(experiment, n_blocks):
    for _ in range(n_blocks):
        #input("Enter any key to start experiment")
        experiment.start()

