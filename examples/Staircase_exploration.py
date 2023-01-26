from slab import Staircase

"""
Staircases are means which give rise to psychometric measures of the perceivable stimulus detection threshold. The slab 
staircase implementation allows the user to keep control of several attributes that may or may not be important for the
accurate investigation of detection thresholds (in this case hearing thresholds). Since staircase trials can be really
tedious and exhausting, researchers want to keep track of specific attributes in their experiment in order to make the 
procedure as convenient for the participant as possible. For example: setting the n_up/n_down rate to 1:1 allows the
researcher to estimate 50 % hearing thresholds (the stimulus level where humans usually report every other trial
correctly) which is an accurate measure of hearing capability, but the stimulus level will really soon reach a level
at which the participant has to put a lot of effort into stimulus detection. Increasing the n_down parameter will there-
fore lead to a higher detection percentage and less exhaustive listening. On the other hand, increasing the n_down para-
meter will lead to a higher amount of total trials in order to estimate hearing thresholds, probably leading to exhaust-
ion in the end as well. Setting the n_down parameter accordingly is therefore based on a trade-off and will be probably
estimated by trial-and-error.
"""

stairs = Staircase(start_val=50,
                   n_reversals=10,
                   step_sizes=[4, 1],
                   step_up_factor=1,
                   n_pretrials=0,
                   n_up=1,
                   n_down=3,  # the higher this value, the longer the staircase duration
                   step_type="lin")

for level in stairs:
    response = stairs.simulate_response(30)
    stairs.add_response(response)
    print(f'reversals: {stairs.reversal_intensities}')
    stairs.plot()

stairs.close_plot()
print(f"mean detection threshold: {stairs.threshold()}")
print(f"total trials: {stairs.this_trial_n}")
