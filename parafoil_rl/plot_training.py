#plot_training.py
import glob
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.results_plotter import load_results, ts2xy

# Find all monitor log folders
monitor_dirs = glob.glob("checkpoints/*/")
for d in monitor_dirs:
    try:
        x, y = ts2xy(load_results(d), "timesteps")
        if len(x) > 0:
            plt.plot(x, y, label=d.split("\\")[-2])
    except:
        pass

plt.xlabel("Timesteps")
plt.ylabel("Episode Reward")
plt.title("Training Progress")
plt.legend()
plt.grid(True)
plt.show()