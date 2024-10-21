import glob
import os
import pickle
import argparse


import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser("PrintTrainingLogs")
parser.add_argument("--log-file", help="Name of the logfile (within the logs folder) to extract data from, latest is taken when not given",
                    type=str)
args = parser.parse_args()

if args.log_file is None:
    list_of_files = glob.glob(os.path.join("../PreTrainingLogs/", '*.p'))
    training_name = max(list_of_files, key=os.path.getctime)
else:
    training_name = "../PreTrainingLogs/" + args.log_file + ".p"

with open(training_name, 'rb') as file:
    metrics = pickle.load(file)
    epochs = len(metrics["combined_loss"])

plt.plot(np.arange(epochs), metrics['combined_loss'], 'r', label="train")
plt.plot(np.arange(epochs), metrics['validation_combined_loss'], 'g', label="val")
plt.title("combined losses")
plt.legend()
plt.show()

plt.plot(np.arange(epochs), metrics['policy_loss'], 'r', label="train")
plt.plot(np.arange(epochs), metrics['validation_policy_loss'], 'g', label="val")
plt.title("policy losses")
plt.legend()
plt.show()

plt.plot(np.arange(epochs), metrics['value_loss'], 'r', label="train")
plt.plot(np.arange(epochs), metrics['validation_value_loss'], 'g', label="val")
plt.title("value losses")
plt.legend()
plt.show()

plt.plot(np.arange(epochs+1), metrics['level_1_puzzle_policy_score'], 'r', label="level_1_policy")
plt.plot(np.arange(epochs+1), metrics['level_1_puzzle_search_score'], 'g', label="level_1_search")
plt.plot(np.arange(epochs+1), metrics['level_2_puzzle_policy_score'], 'b', label="level_2_policy")
plt.plot(np.arange(epochs+1), metrics['level_2_puzzle_search_score'], 'y', label="level_2_search")
plt.title("puzzle scores")
plt.legend()
plt.show()

plt.plot(np.arange(epochs+1), metrics['level_1_puzzle_value_error'], 'r', label="level_1")
plt.plot(np.arange(epochs+1), metrics['level_2_puzzle_value_error'], 'g', label="level_2")
plt.title("puzzle value loss")
plt.legend()
plt.show()

plt.plot(np.arange(epochs+1), metrics['random_agent_game_length'], 'r')
plt.title("Random agent game length")
plt.show()

plt.plot(np.arange(epochs+1), metrics['random_agent_score'], 'r')
plt.title("Random agent score")
plt.show()