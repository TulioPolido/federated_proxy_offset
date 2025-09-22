from flautim.pytorch.common import run_federated, weighted_average
import SmellRecurrenceModel
import SmellRecurrenceDataset
import SmellRecurrenceExperiment
import flwr
from flwr.common import ndarrays_to_parameters
from flwr.server import ServerConfig, ServerAppComponents
import flautim as fl
import torch
import random

DATASET = "PAMAP"
NUM_CHANNELS = 6
BATCH_SIZE = 32
NUM_ROUNDS = 50
NUM_EPOCHS = 2
EMB_DIM = 64


def fit_config(server_round):
    return {"server_round": server_round}


def generate_server_fn(context, eval_fn=None):
    def create_server_fn(context_flwr):
       
            net = SmellRecurrenceModel.RecurrenceModel(
                context, emb_dim=EMB_DIM, num_channels=NUM_CHANNELS, suffix=0
            )
            params = ndarrays_to_parameters(net.get_parameters())

            strategy = flwr.server.strategy.FedAvg(
                initial_parameters=params,
                evaluate_metrics_aggregation_fn=weighted_average,
                fraction_fit=0.5,
                fraction_evaluate=1.0,
                evaluate_fn=eval_fn,
                on_fit_config_fn=fit_config,
                on_evaluate_config_fn=fit_config,
            )

            config = ServerConfig(num_rounds=NUM_ROUNDS)
            return ServerAppComponents(config=config, strategy=strategy)
        

    return create_server_fn


def generate_client_fn(context, files):
    def create_client_fn(context_flwr):
      
            cid = int(context_flwr.node_config["partition-id"])
            file = files[cid]

            data = file["data"]
            labels = file["labels"]

            # shuffle indices
            n_samples = len(data)
            indices = torch.randperm(n_samples)

            split = int(0.8 * n_samples)  # 80% train, 20% test
            train_idx, test_idx = indices[:split], indices[split:]

            train_file = {
                "data": data[train_idx],
                "labels": labels[train_idx]
            }
            test_file = {
                "data": data[test_idx],
                "labels": labels[test_idx]
            }

            model = SmellRecurrenceModel.RecurrenceModel(
                context, emb_dim=EMB_DIM, num_channels=NUM_CHANNELS, suffix=cid
            )

            dataset = SmellRecurrenceDataset.RecurrenceDataset(
                train_file=train_file,
                test_file=test_file,
                augment_multiplier=2,
                use_crop=False,
                use_erase=False,
                use_jitter=False,
                use_rotation=True,
            )

            return SmellRecurrenceExperiment.RecurrenceExperiment(
                model,
                dataset,
                context,
                lr=LEARNING_RATE,
                pre_train_lr=PRE_TRAIN_LEARNING_RATE,
                epochs=NUM_EPOCHS,
                weight_decay=WEIGHT_DECAY,
                lr_step=LEARNING_RATE_STEP,
                lr_gamma=LEARNING_RATE_GAMMA,
                embedding_size=EMB_DIM
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
            PRE_TRAIN_LEARNING_RATE = 1e-4
            LEARNING_RATE = 1e-5
            LEARNING_RATE_STEP = 3
            LEARNING_RATE_GAMMA = 0.5
            WEIGHT_DECAY = 6.7e-5

            all_files = [i for i in range(1, TOTAL_USERS + 1)]

            # Load training client files
            files = [
                torch.load(
                    f"./data/PAMAP_processed_user_absFalse_{WINDOW_SIZE}_{STRIDE}_{i}_rgb_withGyro.pt"
                )
                for i in all_files
            ]

        elif DATASET == "WISDM":
            # Number of training clients
            TOTAL_USERS = 51
            NUM_CLIENTS = 41
            TEST_USERS = TOTAL_USERS - NUM_CLIENTS
            NUM_CLASSES = 5
            WINDOW_SIZE = 256
            STRIDE = 64
            PRE_TRAIN_LEARNING_RATE = 1e-4
            LEARNING_RATE = 1e-5
            LEARNING_RATE_STEP = 3
            LEARNING_RATE_GAMMA = 0.5
            WEIGHT_DECAY = 1.1e-5

            all_files = [i for i in range(1, TOTAL_USERS + 1)]

            # Load training client files
            files = [
                torch.load(
                    f"./data/WISDM_processed_user_absFalse_{WINDOW_SIZE}_{STRIDE}_{i}_rgb_withGyro.pt"
                )
                for i in all_files
            ]


        client_fn_callback = generate_client_fn(context, files)
        evaluate_fn_callback = evaluate_fn(context)
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
