# INITIALISE EXPERIMENT ========================================
from experiment.utils import setup_experiment, set_logger
set_logger("WARNING")

# EXPERIMENT PARAMETERS ========================================
subject_id = "test0000"
plane = "h"

# LOCALISATION ACCURACY ========================================
exp = setup_experiment(subject_id, plane, "LocaAccu", mode="sn")
exp.calibrate_camera()
exp.start()

exp = setup_experiment(subject_id, plane, "LocaAccu", mode="b")
exp.calibrate_camera()
exp.start()

# SPATIAL UNMASKING ============================================
exp = setup_experiment(subject_id, plane, "su")
exp.calibrate_camera()
exp.start()

# NUMEROSITY JUDGEMENT =========================================
exp = setup_experiment(subject_id, plane, "nm", mode="forward")
exp.calibrate_camera()
exp.start()

exp = setup_experiment(subject_id, plane, "nm", mode="forward")
exp.calibrate_camera()
exp.start()

exp = setup_experiment(subject_id, plane, "nm", mode="reversed")
exp.calibrate_camera()
exp.start()

exp = setup_experiment(subject_id, plane, "nm", mode="reversed")
exp.calibrate_camera()
exp.start()

