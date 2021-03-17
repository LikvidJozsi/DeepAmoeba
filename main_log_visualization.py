import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from AmoebaPlayGround.AmoebaTrainer import logs_folder

list_of_files = glob.glob(os.path.join(logs_folder, '*.log'))
latest_file = max(list_of_files, key=os.path.getctime)

df = pd.read_csv(latest_file, sep="\t")

plt.plot(np.arange(len(df)), df['loss'], 'r')
plt.title("Loss")
plt.show()

plt.plot(np.arange(len(df)), df['average_game_length'], 'r')
plt.title("average_game_length")
plt.show()
