from typing import List, Tuple
from collections import OrderedDict
from flautim.pytorch.common import run_federated, weighted_average
import SmellRecurrenceModel
import SmellRecurrenceDataset
import SmellRecurrenceExperiment
import flwr
import torch
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays,NDArrays
from flwr.server import ServerConfig, ServerAppComponents
from flwr.server.strategy.aggregate import aggregate
import flautim as fl
import random

DATASET = "PAMAP"
NUM_CHANNELS = 6
BATCH_SIZE = 8
NUM_ROUNDS = 5
NUM_EPOCHS = 1
EMB_DIM = 64

def fedmeta_update_maml(
    net: torch.nn.Module,
    beta: float,
    weights_results: NDArrays,
    gradients_aggregated: NDArrays,
    weight_decay: float,
) -> NDArrays:
    params_dict = zip(net.state_dict().keys(), weights_results)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
    optimizer = torch.optim.Adam(
        list(net.parameters()), lr=beta, weight_decay=weight_decay
    )
    for params, grad_ins in zip(net.parameters(), gradients_aggregated):
        params.grad = torch.tensor(grad_ins).to(params.dtype)
    optimizer.step()
    optimizer.zero_grad()
    weights_prime = [val.cpu().numpy() for _, val in net.state_dict().items()]

    return weights_prime



class FedMeta(flwr.server.strategy.FedAvg):
    """Federated Meta-Learning Strategy (MAML-style)."""

    def __init__(self, model, lr_outer=0.001, **kwargs):
        super().__init__(**kwargs)  # forward other FedAvg args
        self.global_model = model
        self.lr_outer = lr_outer
        
    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        if not self.accept_failures and failures:
            return None, {}
        
                # Convert results
        weights_results: List[Tuple[NDArrays, int]] = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]


        grads_results: List[Tuple[NDArrays, int]] = [
            (fit_res.metrics["grads"], fit_res.num_examples)  # type: ignore
            for _, fit_res in results
        ]
        gradients_aggregated = aggregate(grads_results)
        weights_prime = fedmeta_update_maml(
            net=self.global_model,
            beta=self.lr_outer,
            weights_results=weights_results[0][0],
            gradients_aggregated=gradients_aggregated,
            weight_decay=0.0001,
        )
        parameters_aggregated = weights_prime

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        return ndarrays_to_parameters(parameters_aggregated), metrics_aggregated



def fit_config(server_round):
    return {"server_round": server_round}


def generate_server_fn(context, eval_fn=None):
    def create_server_fn(context_flwr):

        net = SmellRecurrenceModel.RecurrenceModel(
            context, emb_dim=EMB_DIM, num_channels=NUM_CHANNELS, suffix=0
        )
        params = ndarrays_to_parameters(net.get_parameters())

        strategy = FedMeta(
            model=net,
            lr_outer=1e-4,
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


def generate_client_fn(context, files, test_file):
    def create_client_fn(context_flwr):

        cid = int(context_flwr.node_config["partition-id"])
        file = files[cid]

        model = SmellRecurrenceModel.RecurrenceModel(
            context, emb_dim=EMB_DIM, num_channels=NUM_CHANNELS, suffix=cid
        )

        dataset = SmellRecurrenceDataset.RecurrenceDataset(
            train_file=file,
            test_file=test_file,
            augment_multiplier=2,
            use_crop=False,
            use_erase=False,
            use_jitter=False,
            use_rotation=True
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
            # pre_train=True
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
            PRE_TRAIN_LEARNING_RATE = 5e-5
            LEARNING_RATE = 1e-5
            LEARNING_RATE_STEP = 3
            LEARNING_RATE_GAMMA = 0.5
            WEIGHT_DECAY = 1.1e-5

            all_files = [i for i in range(1, TOTAL_USERS + 1)]

            # Shuffle and split
            test_files = [random.sample(all_files, TEST_USERS)]
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
        evaluate_fn_callback = evaluate_fn(context, combined_test_data)
        server_fn_callback = generate_server_fn(context, eval_fn=None)

        fl.log("Starting Federated Experiment")
        # Run federated learning
        run_federated(
            client_fn=client_fn_callback,
            server_fn=server_fn_callback,
            num_clients=NUM_CLIENTS,
            num_rounds=NUM_ROUNDS,
        )
    except Exception as e:
        fl.log(f"Error in run.py: {e}")
