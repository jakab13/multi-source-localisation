import slab
import matplotlib.pyplot as plt
import scienceplots
# plt.style.use("science")
plt.ion()


hrtf = slab.HRTF.kemar()
sourceidx = hrtf.cone_sources(0)
hrtf.plot_tf(sourceidx, ear='left', kind="image")
plt.tight_layout()
plt.show()
