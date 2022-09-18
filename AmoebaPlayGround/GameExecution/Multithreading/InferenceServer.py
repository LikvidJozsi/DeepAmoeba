import asyncio
import math
from typing import Dict

import ray

from AmoebaPlayGround.Agents.TensorflowModels import NeuralNetworkModel


class NetworkPredictor:
    def predict(self, boards, players):
        pass


class InferenceServerWrapper(NetworkPredictor):

    def __init__(self, model_name, inference_server):
        self.model_name = model_name
        self.inference_server = inference_server

    def predict(self, boards, players):
        return ray.get(self.inference_server.predict.remote(self.model_name, boards, players))

    def worker_finished(self):
        self.inference_server.worker_finished.remote()


@ray.remote
class InferenceServer:
    def __init__(self, models: Dict[str, NeuralNetworkModel]):
        self.worker_count = None
        self.prediction_completion_condition = asyncio.Condition()
        self.results = {}
        for agent in models.values():
            agent.unpack()
        self.models = models
        self.requests = {}
        for model_name in self.models:
            self.requests[model_name] = {}
        self.batch_size = None
        self.request_id_counter = 0
        self.processing = False

    def set_agent_weights(self, agent_name, weights):
        self.models[agent_name].set_weights(weights)

    async def game_group_started(self, worker_count):
        self.update_worker_count(worker_count)

    def update_worker_count(self, worker_count):
        self.worker_count = worker_count
        self.batch_size = max(1, math.floor(self.worker_count / 2))

    def try_start_prediction(self):
        if self.enough_requests_accumulated(self.batch_size) and not self.processing:
            self.processing = True
            asyncio.create_task(self.do_predict())

    async def worker_finished(self):
        self.update_worker_count(self.worker_count - 1)
        self.try_start_prediction()

    async def predict(self, model, boards, players):
        request_id = self.request_id_counter
        self.request_id_counter += 1
        self.requests[model][request_id] = (boards, players)
        self.try_start_prediction()

        while True:
            await self.prediction_completion_condition.acquire()
            await self.prediction_completion_condition.wait()
            if request_id in self.results.keys():
                result = self.results.pop(request_id)
                self.prediction_completion_condition.release()
                return result
            self.prediction_completion_condition.release()

    def enough_requests_accumulated(self, minimum_threshold):
        for requests in self.requests.values():
            if len(requests) >= minimum_threshold:
                return True
        return False

    def get_model_to_predict(self, minimum_threshold):
        for model, requests in self.requests.items():
            if len(requests) >= minimum_threshold:
                return model
        raise Exception("wtf bro just wat")

    async def do_predict(self):
        await self.prediction_completion_condition.acquire()
        model_name = self.get_model_to_predict(self.batch_size)
        requests_to_process = self.get_earliest_requests(model_name, self.batch_size)
        model = self.models[model_name]
        input_boards, players, batch_sizes = self.concatenate_requests(list(requests_to_process.values()))
        output_2d, values = model.predict(input_boards, players)
        cumulative_index = 0
        for batch_size, id in zip(batch_sizes, requests_to_process.keys()):
            output_for_worker = output_2d[cumulative_index:cumulative_index + batch_size]
            values_for_worker = values[cumulative_index:cumulative_index + batch_size]
            self.results[id] = (output_for_worker, values_for_worker)
            cumulative_index += batch_size
        self.prediction_completion_condition.notify_all()
        if self.enough_requests_accumulated(self.batch_size):
            asyncio.create_task(self.do_predict())
        else:
            self.processing = False
        self.prediction_completion_condition.release()

    def concatenate_requests(self, requests):
        batch_sizes = []
        concatenated_boards = []
        concatenated_players = []
        for boards, players in requests:
            batch_sizes.append(len(boards))
            concatenated_boards.extend(boards)
            concatenated_players.extend(players)
        return concatenated_boards, concatenated_players, batch_sizes

    def get_earliest_requests(self, model, count):
        first_request_ids = sorted(list(self.requests[model].keys()))[0:count]
        first_requests = {}
        for id in first_request_ids:
            request = self.requests[model].pop(id)
            first_requests[id] = request
        return first_requests
