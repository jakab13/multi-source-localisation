from labplatform.core.Subject import Subject
from labplatform.config import get_config
import os
import logging
from experiment.Numerosity_Judgement import NumerosityJudgementExperiment
from experiment.Spatial_Unmasking import SpatialUnmaskingExperiment
from experiment.Localization_Accuracy import LocalizationAccuracyExperiment
from experiment.exp_examples import LocalizationAccuracyExperiment_exmp, SpatialUnmaskingExperiment_exmp, \
    NumerosityJudgementExperiment_exmp

# TODO: make set_logger()
log = logging.getLogger(__name__)


def setup_experiment():
    exp_type = input("What is the paradigm? Enter la, su or nm ").lower()
    name = input("Enter subject id: ").lower()
    group = input("Vertical or horizontal? Insert v or h").lower()
    sex = input("m or f? ").upper()
    cohort = input("Pilot or Test cohort? Insert p or t").lower()
    experimenter = input("Enter your name: ").lower()
    is_example = input("Example? y or n").lower()

    # _get_data_path() method call possible if instance calls itself
    #TODO
    # Verzweigung "= None:"-> add_subject...() else: read_from...()
    try:
        if is_example == "y":
            name = 99
            subject = Subject(name=f"sub_{name}",
                              group=group,
                              species="Human",
                              sex=sex,
                              cohort=cohort)
        elif is_example == "n":
            subject = Subject(name=f"sub_{name}",
                              group=group,
                              species="Human",
                              sex=sex,
                              cohort=cohort)
        subject.add_subject_to_h5file(os.path.join(get_config("SUBJECT_ROOT"), f"sub_{name}.h5"))
    except ValueError:
        # read the subject information
        subject = Subject(name=f"sub_{name}",
                          group=group,
                          species="Human",
                          sex=sex,
                          cohort=cohort)
        subject.read_info_from_h5file(file=os.path.join(get_config("SUBJECT_ROOT"), f"sub_{name}.h5"))
    subject.data_path = subject._get_data_path()
    if is_example == "n":
        if exp_type == "su":
            exp = SpatialUnmaskingExperiment(subject=subject, experimenter=experimenter)
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
            log.info("Paradigm not found, aborting ...")

    elif is_example == "y":
        if exp_type == "su":
            exp = SpatialUnmaskingExperiment_exmp(subject=subject, experimenter=experimenter)
            exp.plane = group
            return exp
        elif exp_type == "nm":
            exp = NumerosityJudgementExperiment_exmp(subject=subject, experimenter=experimenter)
            exp.plane = group
            return exp
        elif exp_type == "la":
            exp = LocalizationAccuracyExperiment_exmp(subject=subject, experimenter=experimenter)
            exp.plane = group
            return exp
        else:
            log.info("Paradigm not found, aborting ...")


def run_experiment(experiment, n_blocks):
    for _ in range(n_blocks):
        #input("Enter any key to start experiment")
        experiment.start()


def set_logger(level, report=True):
    """
    Set the logger to a specific level.
    Parameters:
        level: logging level. Only events of this level and above will be tracked. Can be 'DEBUG', 'INFO', 'WARNING',
         'ERROR' or 'CRITICAL'. Set level to '
    """
    try:
        logger = logging.getLogger()
        eval(f"logger.setLevel(logging.{level.upper()})")
        if report:
            print(f"logger.setLevel(logging.{level.upper()})")
    except AttributeError:
        raise AttributeError("Choose from 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'")