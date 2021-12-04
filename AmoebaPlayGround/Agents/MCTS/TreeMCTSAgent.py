from AmoebaPlayGround.Agents.MCTS.BatchMCTSAgent import BatchMCTSAgent
from AmoebaPlayGround.Agents.MCTS.MCTSTree import MCTSTree
from AmoebaPlayGround.Agents.TensorflowModels import NetworkModel, ResNetLike


class TreeMCTSAgent(BatchMCTSAgent):

    def __init__(self, model_name=None, load_latest_model=False,
                 model_type: NetworkModel = ResNetLike(6), search_count=100, exploration_rate=1.4,
                 batch_size=20, training_epochs=10, dirichlet_ratio=0.1, map_size=(8, 8),
                 max_intra_game_parallelism=4, neural_network_evaluator=None,
                 virtual_loss=1, training_dataset_max_size=600000):
        super().__init__(model_name, load_latest_model, model_type, search_count, exploration_rate, batch_size,
                         training_epochs, dirichlet_ratio, map_size, MCTSTree, max_intra_game_parallelism,
                         neural_network_evaluator,
                         virtual_loss, training_dataset_max_size)

    def get_copy(self):
        new_instance = self.__class__(model_type=self.model_type, search_count=self.search_count,
                                      exploration_rate=self.exploration_rate, training_epochs=self.training_epochs,
                                      dirichlet_ratio=self.dirichlet_ratio,
                                      batch_size=self.batch_size, map_size=self.map_size,
                                      max_intra_game_parallelism=self.max_intra_game_parallelism,
                                      virtual_loss=self.virtual_loss,
                                      training_dataset_max_size=self.training_dataset_max_size)
        new_instance.set_weights(self.get_weights())
        return new_instance

    def set_policies(self, nodes, policies, paths):
        for path, node in zip(paths, nodes):
            if len(path) > 0:
                (node, move) = path[-1]
                node.child_policy_calculation_ended(move)

        for node, policy in zip(nodes, policies):
            node.set_policy(policy)

    def node_selected(self, selected_node, path):
        if len(path) > 0:
            (previous_node, move) = path[-1]
            previous_node.child_policy_calculation_started(move)
