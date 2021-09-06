from AmoebaPlayGround.Agents.MCTS.BatchMCTSAgent import BatchMCTSAgent
from AmoebaPlayGround.Agents.MCTS.MCTSTree import MCTSTree
from AmoebaPlayGround.Agents.NetworkModels import NetworkModel, ResNetLike


class TreeMCTSAgent(BatchMCTSAgent):

    def __init__(self, model_name=None, load_latest_model=False,
                 model_type: NetworkModel = ResNetLike(6), search_count=100, exploration_rate=1.4,
                 batch_size=20, training_epochs=10, dirichlet_ratio=0.25, map_size=(8, 8)):
        super().__init__(model_name, load_latest_model, model_type, search_count, exploration_rate, batch_size,
                         training_epochs, dirichlet_ratio, map_size, MCTSTree)

    def set_policies(self, nodes, policies, paths):
        for path, node in zip(paths, nodes):
            if len(path) > 0:
                (node, move) = path[-1]
                node.child_policy_calculation_ended(move)
            else:
                node.policy_calculation_ended()

        for node, policy in zip(nodes, policies):
            node.set_policy(policy)

    def node_selected(self, selected_node, path):
        if len(path) > 0:
            (previous_node, move) = path[-1]
            previous_node.child_policy_calculation_started(move)
        else:
            selected_node.policy_calculation_started()
