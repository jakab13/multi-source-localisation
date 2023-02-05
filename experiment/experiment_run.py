from experiment.utils import setup_experiment, run_experiment


# STEP 1: set up experiment settings
exp = setup_experiment()

# STEP 2: calibrate camera
exp.calibrate_camera()

# STEP 3: start experiment (LocalizationTest, NumerosityJudgement, SpatialUnmasking)
# run_experiment(experiment=exp, n_blocks=1)
exp.start()


