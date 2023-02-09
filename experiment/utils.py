from labplatform.core.Subject import Subject, SubjectList
from labplatform.config import get_config
import os
import logging
from experiment.Numerosity_Judgement import NumerosityJudgementExperiment
from experiment.Spatial_Unmasking import SpatialUnmaskingExperiment
from experiment.Localization_Accuracy import LocalizationAccuracyExperiment
from experiment.exp_examples import LocalizationAccuracyExperiment_exmp, SpatialUnmaskingExperiment_exmp, \
    NumerosityJudgementExperiment_exmp


log = logging.getLogger(__name__)


def setup_experiment():
    exp_type = input("What is the paradigm? Enter la, su or nm ").lower()
    name = input("Enter subject id: ").lower()
    group = input("Vertical or horizontal? Insert v or h").lower()
    sex = input("m or f? ").upper()
    cohort = input("Pilot or Test cohort? Insert p or t").lower()
    experimenter = input("Enter your name: ").lower()
    is_example = input("Example? y or n").lower()

    try:
        if is_example == "y":
            name = "99"
            subject = Subject(name=name,
                              group=group,
                              species="Human",
                              sex=sex,
                              cohort=cohort)
        elif is_example == "n":
            subject = Subject(name=f"sub{name}",
                              group=group,
                              species="Human",
                              sex=sex,
                              cohort=cohort)
        subject.add_subject_to_h5file(os.path.join(get_config("SUBJECT_ROOT"), f"sub{name}.h5"))
    except ValueError:
        # read the subject information
        subject = Subject(name=f"sub{name}",
                          group=group,
                          species="Human",
                          sex=sex,
                          cohort=cohort)
        subject.read_from_h5file()
    subject.data_path = os.path.join(get_config("DATA_ROOT"), f"sub{name}.h5")
    if is_example == "n":
        if exp_type == "su":
            exp = SpatialUnmaskingExperiment(subject=name, experimenter=experimenter)
            exp.plane = group
            return exp
        elif exp_type == "nm":
            exp = NumerosityJudgementExperiment(subject=name, experimenter=experimenter)
            exp.plane = group
            return exp
        elif exp_type == "la":
            exp = LocalizationAccuracyExperiment(subject=name, experimenter=experimenter)
            exp.plane = group
            return exp
        else:
            log.info("Paradigm not found, aborting ...")

    elif is_example == "y":
        if exp_type == "su":
            exp = SpatialUnmaskingExperiment_exmp(subject=name, experimenter=experimenter)
            exp.plane = group
            return exp
        elif exp_type == "nm":
            exp = NumerosityJudgementExperiment_exmp(subject=name, experimenter=experimenter)
            exp.plane = group
            return exp
        elif exp_type == "la":
            exp = LocalizationAccuracyExperiment_exmp(subject=name, experimenter=experimenter)
            exp.plane = group
            return exp
        else:
            log.info("Paradigm not found, aborting ...")


def run_experiment(experiment, n_blocks):
    for _ in range(n_blocks):
        #input("Enter any key to start experiment")
        experiment.start()

