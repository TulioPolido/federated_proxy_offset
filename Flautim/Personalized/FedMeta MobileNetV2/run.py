import random

import flautim as fl
import torch
from fedmeta_recurrence_dataset import RecurrenceDatasetLOSO, RecurrenceDataset
from fedmeta_recurrence_experiment import RecurrenceExperiment
from fedmeta_strategy_v1 import FedMeta
from flautim.pytorch.common import run_federated, weighted_average
from flwr.common import ndarrays_to_parameters
from flwr.server import ServerAppComponents, ServerConfig

MODEL = "MobileNetV2WithoutBatchNorm"
if MODEL == "MobileNetV2WithoutBatchNorm":
    from fedmeta_mobilenet_model import MobileNetV2WithoutBatchNorm as Model
elif MODEL == "MobileNetV2":
    from RecurrenceModel import RecurrenceModel as Model

DATASET = "WISDM"
NUM_CHANNELS = 6
BATCH_SIZE = 16
NUM_ROUNDS = 3000
NUM_EPOCHS = 1
GRADIENT_STEP = 2
EMB_DIM = 64
BETA = 1e-5
SUPPORT_RATIO = 0.3
SMOOTHING_WEIGHT = 0.95
USE_EMA = False
USE_LOSO = True


#########################
# SIMULATION
#########################


def fit_config(server_round):
    return {"server_round": server_round}


def generate_server_fn(context, eval_fn=None):
    def create_server_fn(context_flwr):

        net = Model(
            context, num_classes=NUM_CLASSES, num_channels=NUM_CHANNELS, suffix=0
        )
        params = ndarrays_to_parameters(net.get_parameters())

        strategy = FedMeta(
            model=net,
            lr_outer=BETA,
            initial_parameters=params,
            evaluate_metrics_aggregation_fn=weighted_average,
            fraction_fit=FRACTION_FIT,
            fraction_evaluate=FRACTION_EVALUATE,
            evaluate_fn=eval_fn,
            on_fit_config_fn=fit_config,
            on_evaluate_config_fn=fit_config,
            use_ema=USE_EMA,
            smoothing_weight=SMOOTHING_WEIGHT
        )

        config = ServerConfig(num_rounds=NUM_ROUNDS)
        return ServerAppComponents(config=config, strategy=strategy)

    return create_server_fn


def generate_client_fn(context, files, test_file):
    def create_client_fn(context_flwr):

        cid = int(context_flwr.node_config["partition-id"])
        file = files[cid]

        model = Model(
            context, num_classes=NUM_CLASSES, num_channels=NUM_CHANNELS, suffix=cid
        )

        if USE_LOSO:
            dataset = RecurrenceDatasetLOSO(
                train_file=file,
                test_file=test_file,
                augment_multiplier=2,
                use_rotation=True,
                support_ratio=SUPPORT_RATIO,
                num_workers=0,
                batch_size=BATCH_SIZE,
            )
        else:
            dataset = RecurrenceDataset(
                dataset_file=file,
                augment_multiplier=2,
                use_rotation=True,
                support_ratio=SUPPORT_RATIO,
                num_workers=0,
            )

        return RecurrenceExperiment(
            context,
            model,
            dataset,
            cid=cid,
            lr=LEARNING_RATE,
            epochs=NUM_EPOCHS,
            weight_decay=WEIGHT_DECAY,
            gradient_step=GRADIENT_STEP,
            num_classes=NUM_CLASSES,
        ).to_client()

    return create_client_fn


def evaluate_fn(context, test_data):
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

        if DATASET == "PAMAP":
            # Number of training clients
            TOTAL_USERS = 8
            NUM_CLIENTS = 7
            TEST_USERS = TOTAL_USERS - NUM_CLIENTS
            NUM_CLASSES = 7
            WINDOW_SIZE = 1024
            STRIDE = 256
            LEARNING_RATE = 1e-3
            WEIGHT_DECAY = 6.7e-5
            FRACTION_FIT = 1.0
            FRACTION_EVALUATE = 1.0

            all_files = [i for i in range(1, TOTAL_USERS + 1)]

            # Shuffle and split
            test_files = [7]
            train_files = [i for i in all_files if i not in test_files]
            # test_files = random.sample(all_files, TEST_USERS)
            # train_files = [f for f in all_files if f not in test_files]
            fl.log(f"Train users: {train_files}")
            fl.log(f"Test users: {test_files}")

            # Load training client files
            train_files = [
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
            LEARNING_RATE = 5e-3
            WEIGHT_DECAY = 1.1e-5
            FRACTION_FIT = 0.1
            FRACTION_EVALUATE = 0.4

            all_files = [i for i in range(1, TOTAL_USERS + 1)]

            # Shuffle and split
            test_files = random.sample(all_files, TEST_USERS)
            train_files = [f for f in all_files if f not in test_files]
            fl.log(f"Train users: {train_files}")
            fl.log(f"Test users: {test_files}")

            # Load training client files
            train_files = [
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

        client_fn_callback = generate_client_fn(context, train_files, combined_test_data)
        evaluate_fn_callback = evaluate_fn(context, combined_test_data)
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
