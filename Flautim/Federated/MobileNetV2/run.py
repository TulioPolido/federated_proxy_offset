from flautim.pytorch.common import run_federated, weighted_average
import RecurrenceModel
import FedMetaRecurrenceDataset
import FedRecurrenceExperiment
import flwr
from flwr.common import ndarrays_to_parameters
from flwr.server import ServerConfig, ServerAppComponents
import flautim as fl
import torch
import random

DATASET = "WISDM"
NUM_CHANNELS = 6
BATCH_SIZE = 32
NUM_ROUNDS = 3000
NUM_EPOCHS = 1


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
                fraction_fit=FRACTION_FIT,
                fraction_evaluate=FRACTION_EVALUATE,
                evaluate_fn=eval_fn,
                on_fit_config_fn=fit_config,
                on_evaluate_config_fn=fit_config,
            )

            config = ServerConfig(num_rounds=NUM_ROUNDS)
            return ServerAppComponents(config=config, strategy=strategy)
        except Exception as e:
            print(f"Error in create_server_fn: {e}")

    return create_server_fn


def generate_client_fn(context, files, test_file):
    def create_client_fn(context_flwr):
        try:
            cid = int(context_flwr.node_config["partition-id"])
            file = files[cid]

            model = RecurrenceModel.MobileNetV2WithoutBatchNorm(
                context, num_classes=NUM_CLASSES, num_channels=NUM_CHANNELS, suffix=cid
            )

            dataset = FedMetaRecurrenceDataset.RecurrenceDataset(
                train_file=file,
                test_file=test_file,
                augment_multiplier=AUGMENTATION,
                use_crop=False,
                use_erase=False,
                use_jitter=False,
                use_rotation=True,
            )

            return FedRecurrenceExperiment.RecurrenceExperiment(
                model,
                dataset,
                context,
                lr=LEARNING_RATE,
                epochs=NUM_EPOCHS,
                weight_decay=WEIGHT_DECAY,
                lr_step=LEARNING_RATE_STEP,
                lr_gamma=LEARNING_RATE_GAMMA,
            ).to_client()
        except Exception as e:
            print(f"Error in create_client_fn: {e}")

    return create_client_fn


# def evaluate_fn(context, test_data):
#     def fn(server_round, parameters, config):
#         try:
#             model = RecurrenceModel.RecurrenceModel(
#                 context, num_classes=NUM_CLASSES, num_channels=NUM_CHANNELS
#             )
#             model.set_parameters(parameters)
#             dataset = FedRecurrenceDataset.RecurrenceDataset(
#                 test_data, batch_size=BATCH_SIZE, shuffle=False, test_percentage=1
#             )
#             experiment = FedRecurrenceExperiment.RecurrenceExperiment(
#                 model,
#                 dataset,
#                 context,
#                 lr=LEARNING_RATE,
#                 epochs=NUM_EPOCHS,
#                 weight_decay=WEIGHT_DECAY,
#             )

#             config["server_round"] = server_round
#             loss, _, return_dic = experiment.evaluate(parameters, config)
#             return loss, return_dic
#         except Exception as e:
#             print(f"Error in evaluate_fn: {e}")

#     return fn


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
            FRACTION_FIT = 0.5
            FRACTION_EVALUATE = 1.0
            AUGMENTATION = 2

            all_files = [i for i in range(1, TOTAL_USERS + 1)]

            # Shuffle and split
            test_files = [7]
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
            FRACTION_FIT = 0.1
            FRACTION_EVALUATE = 0.4
            AUGMENTATION = 0

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

        client_fn_callback = generate_client_fn(context, files, combined_test_data)
        # evaluate_fn_callback = evaluate_fn(context, combined_test_data)
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
