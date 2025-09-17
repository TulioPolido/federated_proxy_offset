import torch
import random
import json
import os
import time
from collections import OrderedDict
from typing import Dict, List, Tuple, Union

import FedRecurrenceDataset

import flautim as fl
from flautim.pytorch.common import run_federated
import flwr
from flwr.common import ndarrays_to_parameters, Context, Metrics, NDArrays, ParametersRecord, Scalar
from flwr.server import ServerConfig, ServerAppComponents
from flwr.client import NumPyClient

from utils import get_create_model_fn, get_server_strategy, get_model_manager_class
from models import CNNCifar10ModelManager, CNNCifar100ModelManager
from constants import FEDREP_HEAD_STATE

DATASET = "PAMAP"
NUM_CHANNELS = 6
BATCH_SIZE = 32
NUM_ROUNDS = 175
NUM_EPOCHS = 10

RESULTS_FILE = "result-{}.json"

##############
# SERVER
##############

def config_json_file(context: Context) -> None:
    """Initialize the json file and write the run configurations."""
    # Initialize the execution results directory.
    res_save_path = "./results"
    if not os.path.exists(res_save_path):
        os.makedirs(res_save_path)
    res_save_name = time.strftime("%Y-%m-%d-%H-%M-%S")
    # Set the date and full path of the file to save the results.
    global RESULTS_FILE  # pylint: disable=global-statement
    RESULTS_FILE = RESULTS_FILE.format(res_save_name)
    RESULTS_FILE = f"{res_save_path}/{RESULTS_FILE}"
    data = {
        "run_config": dict(context.run_config.items()),
        "round_res": [],
    }
    with open(RESULTS_FILE, "w+", encoding="UTF-8") as fout:
        json.dump(data, fout, indent=4)

def evaluate_metrics_aggregation_fn(eval_metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Weighted metrics evaluation."""
    weights, accuracies, losses = [], [], []
    for num_examples, metric in eval_metrics:
        weights.append(num_examples)
        accuracies.append(float(metric["accuracy"]) * num_examples)
        losses.append(float(metric["loss"]) * num_examples)
    accuracy = sum(accuracies) / sum(weights)
    loss = sum(losses) / sum(weights)
    # write_res({"accuracy": accuracy, "loss": loss})
    return {"accuracy": accuracy}

def generate_server_fn(context, eval_fn=evaluate_metrics_aggregation_fn):
    def server_fn(context) -> ServerAppComponents:
        """Construct components that set the ServerApp behaviour."""
        config_json_file(context)
        # Read from config

        num_rounds = NUM_ROUNDS

        # Initialize model parameters
        create_model_fn, split_class = get_create_model_fn(context)
        net = split_class(create_model_fn())
        parameters = ndarrays_to_parameters(net.get_parameters())

        # Define strategy
        strategy = get_server_strategy(
            context=context, params=parameters, eval_fn=eval_fn
        )
        config = ServerConfig(num_rounds=int(num_rounds))

        return ServerAppComponents(strategy=strategy, config=config)

    return server_fn

###################
# CLIENT
###################

class BaseClient(NumPyClient):
    """Implementation of Federated Averaging (FedAvg) Client."""

    # pylint: disable=R0913
    def __init__(
        self, model_manager: Union[CNNCifar10ModelManager, CNNCifar100ModelManager]
    ):
        """Initialize client attributes.

        Args:
            model_manager: the model manager object
        """
        super().__init__()
        self.model_manager = model_manager

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Return the current local model parameters."""
        return self.model_manager.model.get_parameters()

    def set_parameters(self, parameters: NDArrays, evaluate: bool = False) -> None:
        """Set the local model parameters to the received parameters.

        Args:
            parameters: parameters to set the model to.
            evaluate: whether to evaluate or not.
        """
        _ = evaluate
        model_keys = [
            k
            for k in self.model_manager.model.state_dict().keys()
            if k.startswith("_body") or k.startswith("_head")
        ]
        params_dict = zip(model_keys, parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model_manager.model.set_parameters(state_dict)

    def perform_train(self) -> Dict[str, Union[List[Dict[str, float]], int, float]]:
        """Perform local training to the whole model.

        Returns
        -------
            Dict with the train metrics.
        """
        self.model_manager.model.enable_body()
        self.model_manager.model.enable_head()

        return self.model_manager.train(self.device)

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Union[bool, bytes, float, int, str]]]:
        """Train the provided parameters using the locally held dataset.

        Args:
            parameters: The current (global) model parameters.
            config: configuration parameters for training sent by the server.

        Returns
        -------
            Tuple containing the locally updated model parameters, \
                the number of examples used for training and \
                the training metrics.
        """
        self.set_parameters(parameters)
        self.perform_train()

        return self.get_parameters(config), self.model_manager.train_dataset_size(), {}

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Union[bool, bytes, float, int, str]]]:
        """Evaluate the provided global parameters using the locally held dataset.

        Args:
            parameters: The current (global) model parameters.
            config: configuration parameters for training sent by the server.

        Returns
        -------
        Tuple containing the test loss, \
                the number of examples used for evaluation and \
                the evaluation metrics.
        """
        self.set_parameters(parameters, evaluate=True)

        # Test the model
        test_results = self.model_manager.test(self.device)

        return (
            test_results.get("loss", 0.0),
            self.model_manager.test_dataset_size(),
            {k: v for k, v in test_results.items() if not isinstance(v, (dict, list))},
        )


class FedRepClient(BaseClient):
    """Implementation of Federated Personalization (FedRep) Client."""

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Return the current local body parameters."""
        return [
            val.cpu().numpy()
            for val in self.model_manager.model.body.state_dict().values()
        ]

    def set_parameters(self, parameters: NDArrays, evaluate: bool = False) -> None:
        """Set the local body parameters to the received parameters.

        Args:
            parameters: parameters to set the body to.
            evaluate: whether the client is evaluating or not.
        """
        model_keys = [
            k
            for k in self.model_manager.model.state_dict().keys()
            if k.startswith("_body")
        ]

        if not evaluate:
            # Only update client's local head if it hasn't trained yet
            model_keys.extend(
                [
                    k
                    for k in self.model_manager.model.state_dict().keys()
                    if k.startswith("_head")
                ]
            )

        state_dict = OrderedDict(
            (k, torch.from_numpy(v)) for k, v in zip(model_keys, parameters)
        )

        self.model_manager.model.set_parameters(state_dict)

def generate_client_fn(context, files):
    def create_client_fn(context_flwr):
        try:
            model_manager_class = get_model_manager_class(context)
            cid = int(context_flwr.node_config["partition-id"])
            file = files[cid]

            dataset = FedRecurrenceDataset.RecurrenceDataset(
                file,
                augment_multiplier=2,
                use_crop=False,
                use_erase=False,
                use_jitter=False,
                use_rotation=True,
                test_percentage=0.2
            )

            trainloader = dataset.dataloader()
            valloader = dataset.dataloader(validation=True)

            # This state variable will only be used by the FedRep algorithm.
            # We only need to initialize once, since client_fn will be called
            # again at every invocation of the ClientApp.
            if FEDREP_HEAD_STATE not in context.state.parameters_records:
                context.state.parameters_records[FEDREP_HEAD_STATE] = ParametersRecord()
            client_class = FedRepClient

            model_manager_obj = model_manager_class(
                context=context, trainloader=trainloader, testloader=valloader
            )

            # Return client object.
            client = client_class(model_manager_obj).to_client()
            return client
        except Exception as e:
            print(f"Error in create_client_fn: {e}")

    return create_client_fn

###########
# RUN
###########

if __name__ == "__main__":
    try:
        context = fl.init()
        # Generate a random seed
        seed = random.randint(0, 2**32 - 1)
        random.seed(seed)
        fl.log(f"Using random seed: {seed}")

        if DATASET == "PAMAP":
            # Number of training clients
            TOTAL_USERS = 8
            NUM_CLIENTS = 7
            TEST_USERS = TOTAL_USERS - NUM_CLIENTS
            NUM_CLASSES = 7
            WINDOW_SIZE = 1024
            STRIDE = 256
            LEARNING_RATE = 1e-4
            LEARNING_RATE_STEP = 3
            LEARNING_RATE_GAMMA = 0.5
            WEIGHT_DECAY = 6.7e-5

            all_files = [i for i in range(1, TOTAL_USERS + 1)]

            # Shuffle and split
            test_files = [8]
            train_files = [i for i in all_files if i not in test_files]
            # test_files = random.sample(all_files, TEST_USERS)
            # train_files = [f for f in all_files if f not in test_files]
            fl.log(f"Train users: {train_files}")
            fl.log(f"Test users: {test_files}")

            # Load training client files
            files = [
                torch.load(
                    f"./data/PAMAP_processed_user_absFalse_{WINDOW_SIZE}_{STRIDE}_{i}_rgb_withGyro.pt"
                )
                for i in train_files
            ]

            # Load and combine test files
            test_files = [
                torch.load(
                    f"./data/PAMAP_processed_user_absFalse_{WINDOW_SIZE}_{STRIDE}_{i}_rgb_withGyro.pt"
                )
                for i in test_files
            ]

        elif DATASET == "WISDM":
            # Number of training clients
            TOTAL_USERS = 51
            NUM_CLIENTS = 41
            TEST_USERS = TOTAL_USERS - NUM_CLIENTS
            NUM_CLASSES = 5
            WINDOW_SIZE = 256
            STRIDE = 64
            LEARNING_RATE = 6.9e-4
            LEARNING_RATE_STEP = 13
            LEARNING_RATE_GAMMA = 0.53
            WEIGHT_DECAY = 1.1e-5

            all_files = [i for i in range(1, TOTAL_USERS + 1)]

            # Shuffle and split
            test_files = random.sample(all_files, TEST_USERS)
            train_files = [f for f in all_files if f not in test_files]
            fl.log(f"Train users: {train_files}")
            fl.log(f"Test users: {test_files}")

            # Load training client files
            files = [
                torch.load(
                    f"./data/WISDM_processed_user_absFalse_{WINDOW_SIZE}_{STRIDE}_{i}_rgb_withGyro.pt"
                )
                for i in train_files
            ]

            # Load and combine test files
            test_files = [
                torch.load(
                    f"./data/WISDM_processed_user_absFalse_{WINDOW_SIZE}_{STRIDE}_{i}_rgb_withGyro.pt"
                )
                for i in test_files
            ]

        combined_test_data = {
            "data": torch.cat([f["data"] for f in test_files], dim=0),
            "labels": torch.cat([f["labels"] for f in test_files], dim=0),
        }

        client_fn_callback = generate_client_fn(context, files)
        server_fn_callback = generate_server_fn(context, eval_fn=None)

        # Run federated learning
        run_federated(
            client_fn=client_fn_callback,
            server_fn=server_fn_callback,
            num_clients=NUM_CLIENTS,
            num_rounds=NUM_ROUNDS,
        )
    except Exception as e:
        fl.log(f"Error in run.py: {e}")
