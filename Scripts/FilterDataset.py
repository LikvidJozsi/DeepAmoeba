import math
import pickle

from tqdm import tqdm

from AmoebaPlayGround.Training.TrainingSampleGenerator import TrainingSampleCollection


def calculate_entropy(sample):
    entropy = 0
    for i in sample.flatten():
        if i == 0:
            continue
        entropy -= i * math.log(i, 2)
    return entropy


def filter_dataset(dataset):
    low_entropy_samples = TrainingSampleCollection()
    progress_bar = tqdm(total=len(dataset.board_states))
    for index, move_probability in enumerate(dataset.move_probabilities):
        progress_bar.update(1)
        if calculate_entropy(move_probability) < 6:
            low_entropy_samples.add_sample(dataset.board_states[index],
                                           move_probability, dataset.rewards[index])

    print(len(dataset.board_states))
    print(len(low_entropy_samples.board_states))
    return low_entropy_samples


if __name__ == '__main__':
    with open("../Datasets/large_dataset.p", 'rb') as train_file, open("../Datasets/evaluation_dataset.p",
                                                                       'rb') as eval_file:
        dataset = pickle.load(train_file)
    filtered = filter_dataset(dataset)
    with open("../Datasets/filtered_dataset.p", 'wb') as file:
        pickle.dump(filtered, file)
