from labplatform.core import Subject, SubjectList
from labplatform.config import get_config
import os
import datetime
from experiment.Numerosity_Judgement import NumerosityJudgementExperiment
from experiment.Spatial_Unmasking import SpatialUnmaskingExperiment
from experiment.Localization_Accuracy import LocalizationAccuracyExperiment
import pickle


# Create subject
try:
    subject = Subject(name="Foo",
                      group="Pilot",
                      birth=datetime.date(1996, 11, 18),
                      species="Human",
                      sex="M")
    subject.data_path = os.path.join(get_config("DATA_ROOT"), "Foo_test.h5")
    subject.add_subject_to_h5file(os.path.join(get_config("SUBJECT_ROOT"), "Foo_test.h5"))
    # test_subject.file_path
except ValueError:
    # read the subject information
    sl = SubjectList(file_path=os.path.join(get_config("SUBJECT_ROOT"), "Foo_test.h5"))
    sl.read_from_h5file()
    subject = sl.subjects[0]
    subject.data_path = os.path.join(get_config("DATA_ROOT"), "Foo_test.h5")
# subject.file_path
experimenter = "Max"
exp = LocalizationAccuracyExperiment(subject=subject, experimenter=experimenter)
# exp.load_pinknoise()
exp.load_babble()

# save file
dict_data = exp.signals

file = os.path.join(os.getcwd(), "analysis", "locaaccu_babble_noise.pkl")


with open(file, 'wb') as fp:
    pickle.dump(dict_data, fp)
    print('dictionary saved successfully to file')

# test load data
with open(file, 'rb') as fp:
    data = pickle.load(fp)
