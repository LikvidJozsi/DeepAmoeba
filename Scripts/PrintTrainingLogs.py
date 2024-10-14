import glob
import os
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from AmoebaPlayGround.Training.Logger import logs_folder

parser = argparse.ArgumentParser("PrintTrainingLogs")
parser.add_argument("--log-file", help="Name of the logfile (within the logs folder) to extract data from, latest is taken when not given",
                    type=str)
args = parser.parse_args()


if args.log_file is not None:
    log_file_name = os.path.join(logs_folder, args.log_file)
else:
    list_of_files = glob.glob(os.path.join(logs_folder, '*.csv'))
    latest_file = max(list_of_files, key=os.path.getctime)
    log_file_name = latest_file

df = pd.read_csv(log_file_name, sep=",")

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

plt.plot(np.arange(len(df)), df['loss'], 'r', label="combined_loss")
plt.plot(np.arange(len(df)), df['policy_loss'], 'g', label="policy_loss")
plt.plot(np.arange(len(df)), df['value_loss'], 'b', label="value_loss")
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
plt.title("puzzle performance")
plt.legend()
plt.show()

plt.plot(np.arange(len(df)), df['level_1_puzzle_value_error'], 'r', label="level_1_puzzle_value_error")
plt.plot(np.arange(len(df)), df['level_2_puzzle_value_error'], 'g', label="level_2_puzzle_value_error")
plt.title("puzzle value performance")
plt.legend()
plt.show()

# plt.plot(np.arange(len(df)), df['level_4_puzzle_policy_score'], 'b', label="level_4_puzzle_policy_score")
# plt.plot(np.arange(len(df)), df['level_4_puzzle_search_score'], 'y', label="level_4_puzzle_search_score")
# plt.plot(np.arange(len(df)), df['level_5_puzzle_policy_score'], 'r', label="level_5_puzzle_policy_score")
# plt.plot(np.arange(len(df)), df['level_5_puzzle_search_score'], 'g', label="level_5_puzzle_search_score")
# plt.title("hard puzzle performance")
# plt.legend()
# plt.show()

plt.plot(np.arange(len(df)), df['avg_fraction_not_visited'], 'b', label="not_visited")
plt.plot(np.arange(len(df)), df['avg_fraction_visited_once'], 'y', label="visited_once")
plt.plot(np.arange(len(df)), df['avg_fraction_visited_twice'], 'r', label="visited_twice")
plt.plot(np.arange(len(df)), df['avg_fraction_visited_at_least_10_times'], 'g', label="visited_at_least_10_times")
plt.plot(np.arange(len(df)), df['fraction_won_by_player_1'], 'm', label="won_by_player_1")
plt.title("self play_stats")
plt.legend()
plt.show()

plt.plot(np.arange(len(df)), df['hand_written_agent_score'], 'r', label="hand written score")
plt.plot(np.arange(len(df)), df['fraction_draw'], 'b', label="self-play draws")
plt.title("hand_written_agent winrate")
plt.legend()
plt.show()

plt.plot(np.arange(len(df)), df['top_1_move_average_search_count'], 'r', label="top_1_move_average_search_count")
plt.title("top 1 move average search count")
plt.legend()
plt.show()
