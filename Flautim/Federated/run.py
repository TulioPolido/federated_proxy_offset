from flautim.pytorch.common import run_federated, weighted_average
import RecurrenceModel
import FedRecurrenceDataset
import FedRecurrenceExperiment
import flwr
from flwr.common import ndarrays_to_parameters
from flwr.server import ServerConfig, ServerAppComponents
import flautim as fl
import torch
import random


DATASET = "PAMAP"
NUM_CHANNELS = 6
NUM_CLIENTS = 6
BATCH_SIZE = 32
NUM_EPOCHS = 5
NUM_ROUNDS = 150


def fit_config(server_round):
    return {"server_round": server_round}


def generate_server_fn(context, eval_fn):
    def create_server_fn(context_flwr):
        try:
            net = RecurrenceModel.RecurrenceModel(
                context, num_classes=NUM_CLASSES, num_channels=NUM_CHANNELS, suffix=0
            )
            params = ndarrays_to_parameters(net.get_parameters())

            strategy = flwr.server.strategy.FedAvg(
                initial_parameters=params,
                evaluate_metrics_aggregation_fn=weighted_average,
                fraction_fit=1.0,
                fraction_evaluate=1.0,
                evaluate_fn=eval_fn,
                on_fit_config_fn=fit_config,
                on_evaluate_config_fn=fit_config,
            )

            config = ServerConfig(num_rounds=NUM_ROUNDS)
            return ServerAppComponents(config=config, strategy=strategy)
        except Exception as e:
            print(f"Error in create_server_fn: {e}")

    return create_server_fn


def generate_client_fn(context, files):
    def create_client_fn(context_flwr):
        try:
            cid = int(context_flwr.node_config["partition-id"])
            file = files[cid]
            model = RecurrenceModel.RecurrenceModel(
                context, num_classes=NUM_CLASSES, num_channels=NUM_CHANNELS, suffix=cid
            )
            dataset = FedRecurrenceDataset.RecurrenceDataset(
                file, batch_size=BATCH_SIZE, shuffle=True
            )
            return FedRecurrenceExperiment.RecurrenceExperiment(
                model,
                dataset,
                context,
                lr=LEARNING_RATE,
                lr_step=LEARNING_RATE_STEP,
                lr_gamma=LEARNING_RATE_GAMMA,
                epochs=NUM_EPOCHS,
                weight_decay=WEIGHT_DECAY,
            ).to_client()
        except Exception as e:
            print(f"Error in create_client_fn: {e}")

    return create_client_fn


def evaluate_fn(context, test_data):
    def fn(server_round, parameters, config):
        try:
            model = RecurrenceModel.RecurrenceModel(
                context, num_classes=NUM_CLASSES, num_channels=NUM_CHANNELS
            )
            model.set_parameters(parameters)
            dataset = FedRecurrenceDataset.RecurrenceDataset(
                test_data, batch_size=BATCH_SIZE, shuffle=False
            )
            experiment = FedRecurrenceExperiment.RecurrenceExperiment(
                model,
                dataset,
                context,
                lr=LEARNING_RATE,
                lr_step=LEARNING_RATE_STEP,
                lr_gamma=LEARNING_RATE_GAMMA,
                epochs=NUM_EPOCHS,
                weight_decay=WEIGHT_DECAY,
            )

            config["server_round"] = server_round
            loss, _, return_dic = experiment.evaluate(parameters, config)
            return loss, return_dic
        except Exception as e:
            print(f"Error in evaluate_fn: {e}")

    return fn


if __name__ == "__main__":
    try:
        context = fl.init()
        # Generate a random seed
        seed = random.randint(0, 2**32 - 1)
        random.seed(seed)
        fl.log(f"Using random seed: {seed}")

        if DATASET == "PAMAP":
            LEARNING_RATE = 4.7e-3
            LEARNING_RATE_STEP = 6
            LEARNING_RATE_GAMMA = 0.808
            WEIGHT_DECAY = 6.7e-5
            WINDOW_SIZE = 1024
            STRIDE = 256
            NUM_CLASSES = 7
            # All available file indices
            all_clients = [i for i in range(1, 9)]

            # Shuffle and split
            train_clients = random.sample(all_clients, NUM_CLIENTS)
            test_clients = [f for f in all_clients if f not in train_clients]

            fl.log(f"Train clients: {train_clients}")
            fl.log(f"Test clients: {test_clients}")

            # Load training client files
            files = [
                torch.load(
                    f"./data/PAMAP_processed_user_absFalse_{WINDOW_SIZE}_{STRIDE}_{client}_rgb_withGyro.pt"
                )
                for client in train_clients
            ]

            # Load and combine test files
            test_files = [
                torch.load(
                    f"./data/PAMAP_processed_user_absFalse_{WINDOW_SIZE}_{STRIDE}_{client}_rgb_withGyro.pt"
                )
                for client in test_clients
            ]

        elif DATASET == "WISDM":
            LEARNING_RATE = 6.9e-4
            LEARNING_RATE_STEP = 13
            LEARNING_RATE_GAMMA = 0.53
            WEIGHT_DECAY = 1.1e-5
            WINDOW_SIZE = 256
            STRIDE = 64
            NUM_CLASSES = 5
            # All available file indices
            all_clients = [i for i in range(1, 52)]

            # Shuffle and split
            train_clients = random.sample(all_clients, NUM_CLIENTS)
            test_clients = [f for f in all_clients if f not in train_clients]

            fl.log(f"Train clients: {train_clients}")
            fl.log(f"Test clients: {test_clients}")

            # Load training client files
            files = [
                torch.load(
                    f"./data/WISDM_processed_user_absFalse_{WINDOW_SIZE}_{STRIDE}_{client}_rgb_withGyro.pt"
                )
                for client in train_clients
            ]

            # Load and combine test files
            test_files = [
                torch.load(
                    f"./data/WISDM_processed_user_absFalse_{WINDOW_SIZE}_{STRIDE}_{client}_rgb_withGyro.pt"
                )
                for client in test_clients
            ]

        combined_test_data = {
            "data": torch.cat([f["data"] for f in test_files], dim=0),
            "labels": torch.cat([f["labels"] for f in test_files], dim=0),
        }

        client_fn_callback = generate_client_fn(context, files)
        evaluate_fn_callback = evaluate_fn(context, combined_test_data)
        server_fn_callback = generate_server_fn(context, eval_fn=evaluate_fn_callback)

        # Run federated learning
        run_federated(
            client_fn=client_fn_callback,
            server_fn=server_fn_callback,
            num_clients=NUM_CLIENTS,
            num_rounds=NUM_ROUNDS,
        )
    except Exception as e:
        fl.log(f"Error in run.py: {e}")
