import random

import flautim as fl
import RecurrenceModel
from fedmeta_recurrence_experiment import RecurrenceExperiment
from fedmeta_recurrence_dataset import RecurrenceDataset
from fedmeta_strategy_v1 import FedMeta
from flautim.pytorch.common import run_federated, weighted_average
from flwr.common import ndarrays_to_parameters
from flwr.server import ServerAppComponents, ServerConfig

from dataset import load_datasets

DATASET = "FEMINIST"
NUM_CHANNELS = 1
BATCH_SIZE = 32
NUM_ROUNDS = 10
NUM_EPOCHS = 1
EMB_DIM = 64


#########################
# SIMULATION
#########################


def fit_config(server_round):
    return {"server_round": server_round}


def generate_server_fn(context, eval_fn=None):
    def create_server_fn(context_flwr):

        net = RecurrenceModel.RecurrenceModel(
            context, num_classes=NUM_CLASSES, num_channels=NUM_CHANNELS, suffix=0
        )
        params = ndarrays_to_parameters(net.get_parameters())
        

        strategy = FedMeta(
            model=net,
            lr_outer=1e-4,
            initial_parameters=params,
            evaluate_metrics_aggregation_fn=weighted_average,
            fraction_fit=0.00001,
            fraction_evaluate=0.00001,
            evaluate_fn=eval_fn,
            on_fit_config_fn=fit_config,
            on_evaluate_config_fn=fit_config,
        )

        config = ServerConfig(num_rounds=NUM_ROUNDS)
        return ServerAppComponents(config=config, strategy=strategy)

    return create_server_fn


def generate_client_fn(context, trainloaders, valloaders):
    def create_client_fn(context_flwr):

        cid = int(context_flwr.node_config["partition-id"])

        model = RecurrenceModel.RecurrenceModel(
            context, num_classes=NUM_CLASSES, num_channels=NUM_CHANNELS, suffix=cid
        )

        dataset = RecurrenceDataset(
            num_workers=0
        )

        return RecurrenceExperiment(
            context,
            model,
            dataset, # Não usado. Placeholder.
            trainloaders,
            valloaders,
            cid=cid
        ).to_client()

    return create_client_fn


def evaluate_fn(context):
    def fn(server_round, parameters, config):
        pass

    return fn


if __name__ == "__main__":
    try:
        context = fl.init()
        # Generate a random seed
        seed = random.randint(0, 2**32 - 1)
        seed = 2270238713
        random.seed(seed)
        fl.log(f"Using random seed: {seed}")

        if DATASET == "FEMINIST":
            NUM_CLASSES = 62

            # partition dataset and get dataloaders
            trainloaders, valloaders, _ = load_datasets(path="./data/")
            NUM_CLIENTS = len(trainloaders["sup"])

        client_fn_callback = generate_client_fn(context, trainloaders, valloaders)
        server_fn_callback = generate_server_fn(context, eval_fn=None)

        clients_resources = {"num_cpus": 1, "num_gpus": 0.15}

        fl.log("Starting Federated Experiment")
        # Run federated learning
        run_federated(
            client_fn=client_fn_callback,
            server_fn=server_fn_callback,
            num_clients=NUM_CLIENTS,
            num_rounds=NUM_ROUNDS,
            clients_resources=clients_resources,
        )
    except Exception as e:
        fl.log(f"Error in run.py: {e}")
