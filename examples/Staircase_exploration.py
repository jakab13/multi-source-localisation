from slab import Staircase
import matplotlib.pyplot as plt
import scienceplots
plt.style.use("science")
plt.ion()


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
estimated by trial-and-error. Also, setting the step_up_factor to n_down*(1-Xthresh) allows for weighted staircases,
which in theory will yield a more efficient staircase with less error and trials. However, this is not the truth with 
simulated responses.
"""

start_val = 75
stairs = Staircase(start_val=start_val,  # starting dB value
                   n_reversals=16,  # number of reversals
                   step_sizes=[4, 2, 1],  # step sizes (in this case, go 4 dB until first reversal point, then do 1 steps)
                   step_up_factor=1,  # Sdown * (1-Xthresh)
                   n_pretrials=0,  # number of pretrials before the staircase begins
                   n_up=1,  # amount of correct responses before stimulus is lowered in dB
                   n_down=1,  # amount of correct responses before stimulus is lowered in dB
                   step_type="db")

simulated_hearing_threshold = start_val - 30
for level in stairs:
    response = stairs.simulate_response(simulated_hearing_threshold)
    stairs.add_response(response)
    stairs.plot(show=False)

plt.title("")
plt.ylabel("Target Sound Intensity [dB]")
plt.show()
print(f"mean detection threshold: {stairs.threshold()}")
print(f"deviation from true threshold: {simulated_hearing_threshold-stairs.threshold()}")
print(f"total trials: {stairs.this_trial_n}")

plt.savefig("/home/max/labplatform/plots/MA_thesis/materials_methods/staircase_example.png",
            dpi=800)
