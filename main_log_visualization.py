import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from AmoebaPlayGround.Training.Logger import logs_folder

list_of_files = glob.glob(os.path.join(logs_folder, '*.csv'))
latest_file = max(list_of_files, key=os.path.getctime)
df = pd.read_csv(latest_file, sep=",")

plt.plot(np.arange(len(df)), df['max_search_depth'], 'r', label="max_search_depth")
plt.plot(np.arange(len(df)), df['avg_search_depth'], 'b', label="avg_search_depth")
plt.title("search depths")
plt.legend()
plt.show()

plt.plot(np.arange(len(df)), df['avg_game_length'], 'r', label="self play")
plt.plot(np.arange(len(df)), df['random_agent_game_length'], 'g', label="random_agent_game_length")
plt.plot(np.arange(len(df)), df['hand_written_agent_game_length'], 'b', label="hand_written_agent_game_length")
plt.title("avg_game_length")
plt.legend()
plt.show()

plt.plot(np.arange(len(df)), df['loss'], 'r')
plt.title("loss")
plt.legend()
plt.show()

plt.plot(np.arange(len(df)), df['agent_rating'], 'r', label="agent_rating")
plt.title("agent_rating")
plt.legend()
plt.show()

plt.plot(np.arange(len(df)), df['level_1_puzzle_policy_score'], 'r', label="level_1_puzzle_policy_score")
plt.plot(np.arange(len(df)), df['level_1_puzzle_search_score'], 'g', label="level_1_puzzle_search_score")
plt.plot(np.arange(len(df)), df['level_2_puzzle_policy_score'], 'b', label="level_2_puzzle_policy_score")
plt.plot(np.arange(len(df)), df['level_2_puzzle_search_score'], 'y', label="level_2_puzzle_search_score")
plt.plot(np.arange(len(df)), df['level_3_puzzle_policy_score'], 'o', label="level_3_puzzle_policy_score")
plt.plot(np.arange(len(df)), df['level_3_puzzle_search_score'], 'm', label="level_3_puzzle_search_score")
plt.title("puzzle performance")
plt.legend()
plt.show()

plt.plot(np.arange(len(df)), df['level_1_puzzle_value_error'], 'r', label="level_1_puzzle_value_error")
plt.plot(np.arange(len(df)), df['level_2_puzzle_value_error'], 'g', label="level_2_puzzle_value_error")
plt.plot(np.arange(len(df)), df['level_3_puzzle_value_error'], 'b', label="level_3_puzzle_value_error")
plt.plot(np.arange(len(df)), df['level_4_puzzle_value_error'], 'y', label="level_4_puzzle_value_error")
plt.plot(np.arange(len(df)), df['level_5_puzzle_value_error'], 'm', label="level_5_puzzle_value_error")
plt.title("puzzle value performance")
plt.legend()
plt.show()

plt.plot(np.arange(len(df)), df['level_4_puzzle_policy_score'], 'b', label="level_4_puzzle_policy_score")
plt.plot(np.arange(len(df)), df['level_4_puzzle_search_score'], 'y', label="level_4_puzzle_search_score")
plt.plot(np.arange(len(df)), df['level_5_puzzle_policy_score'], 'r', label="level_5_puzzle_policy_score")
plt.plot(np.arange(len(df)), df['level_5_puzzle_search_score'], 'g', label="level_5_puzzle_search_score")
plt.title("hard puzzle performance")
plt.legend()
plt.show()
