from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union
from flautim.pytorch.common import run_federated, weighted_average
import numpy as np
import SmellRecurrenceModel
import SmellRecurrenceDataset
import SmellRecurrenceExperiment
import flwr
import torch
from logging import INFO
from flwr.common.logger import log
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Metrics,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server import ServerConfig, ServerAppComponents
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
import flautim as fl
import random

DATASET = "PAMAP"
NUM_CHANNELS = 6
BATCH_SIZE = 16
NUM_ROUNDS = 5
NUM_EPOCHS = 1
EMB_DIM = 64

######################################
# FEDMETA
######################################


def update_ema(
    prev_ema: float,
    current_value: float,
    smoothing_weight: float,
) -> float:
    if prev_ema is None:
        return current_value
    return (1 - smoothing_weight) * current_value + smoothing_weight * prev_ema


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

    def __init__(self, model, lr_inner=1e-4, lr_outer=1e-4, **kwargs):
        super().__init__(**kwargs)  # forward other FedAvg args
        self.net = model
        self.beta = lr_outer
        self.alpha = lr_inner
        # self.alpha = torch.nn.ParameterList(
        #     [
        #         torch.nn.Parameter(torch.full_like(p, lr_inner))
        #         for p in self.net.parameters()
        #     ]
        # )

    # def configure_fit(
    #     self, server_round: int, parameters: Parameters, client_manager: ClientManager
    # ) -> List[Tuple[ClientProxy, FitIns]]:
    #     """Configure the next round of training."""
    #     if self.on_fit_config_fn is not None:
    #         # Custom fit config function provided
    #         config = self.on_fit_config_fn(server_round)
    #     config["alpha"] = self.alpha
    #     fit_ins = FitIns(parameters, config)

    #     # Sample clients
    #     sample_size, min_num_clients = self.num_fit_clients(
    #         client_manager.num_available()
    #     )
    #     clients = client_manager.sample(
    #         num_clients=sample_size, min_num_clients=min_num_clients
    #     )

    #     # Return client/config pairs
    #     return [(client, fit_ins) for client in clients]

    def aggregate_fit(self, server_round, results, failures):
        for failure in failures:
            print(f"Fit Failure: {failure}")

        if not results:
            return None, {}

        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        weights_results: List[Tuple[NDArrays, int]] = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        # Load grads from the files
        grads_results: List[Tuple[NDArrays, int]] = []
        for _, fit_res in results:
            grads_file = fit_res.metrics.get("grads_file")
            if grads_file is None:
                raise ValueError("No grads_file in metrics for a client!")
            # Load the .npz file
            log(INFO, f"Reading file for: {grads_file}")
            file_path = f"gradients_{grads_file}.npz"
            with np.load(file_path) as data:
                grads = [data[key] for key in data]
            grads_results.append((grads, fit_res.num_examples))

        # Aggregate gradients
        gradients_aggregated = aggregate(grads_results)

        weights_prime = fedmeta_update_maml(
            net=self.net,
            beta=self.beta,
            weights_results=weights_results[0][0],
            gradients_aggregated=gradients_aggregated,
            weight_decay=1e-4,
        )
        parameters_aggregated = ndarrays_to_parameters(weights_prime)

        # Metrics
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        return parameters_aggregated, metrics_aggregated

    # def configure_evaluate(
    #     self, server_round: int, parameters: Parameters, client_manager: ClientManager
    # ) -> List[Tuple[ClientProxy, EvaluateIns]]:
    #     """Configure the next round of evaluation."""
    #     # Do not configure federated evaluation if fraction eval is 0.
    #     if self.fraction_evaluate == 0.0:
    #         return []

    #     # Parameters and config
    #     config = {}
    #     if self.on_evaluate_config_fn is not None:
    #         # Custom evaluation config function provided
    #         config = self.on_evaluate_config_fn(server_round)
    #     config["alpha"] = self.alpha
    #     evaluate_ins = EvaluateIns(parameters, config)

    #     # Sample clients
    #     sample_size, min_num_clients = self.num_evaluation_clients(
    #         client_manager.num_available()
    #     )
    #     clients = client_manager.sample(
    #         num_clients=sample_size, min_num_clients=min_num_clients
    #     )

    #     # Return client/config pairs
    #     return [(client, evaluate_ins) for client in clients]

    # def aggregate_evaluate(
    #     self,
    #     server_round: int,
    #     results: List[Tuple[ClientProxy, EvaluateRes]],
    #     failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    # ) -> Tuple[Optional[float], Dict[str, Scalar]]:
    #     """Aggregate evaluation losses using weighted average."""
    #     for failure in failures:
    #         print(f"Evaluate Failure: {failure}")
    #     if not results:
    #         return None, {}
    #     # Do not aggregate if there are failures and failures are not accepted
    #     if not self.accept_failures and failures:
    #         return None, {}

    #     # Aggregate loss
    #     loss_aggregated = weighted_loss_avg(
    #         [
    #             (evaluate_res.num_examples, evaluate_res.loss)
    #             for _, evaluate_res in results
    #         ]
    #     )

    #     smoothing_weight = 0.7
    #     self.ema_loss = update_ema(self.ema_loss, loss_aggregated, smoothing_weight)
    #     loss_aggregated = self.ema_loss

    #     # Aggregate custom metrics if aggregation fn was provided
    #     metrics_aggregated = {}
    #     if self.evaluate_metrics_aggregation_fn:
    #         eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
    #         metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
    #         self.ema_acc = update_ema(
    #             self.ema_acc,
    #             round(float(metrics_aggregated["accuracy"] * 100), 3),
    #             smoothing_weight,
    #         )
    #         metrics_aggregated["accuracy"] = self.ema_acc

    #     return loss_aggregated, metrics_aggregated


#########################
# SIMULATION
#########################


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
            use_rotation=True,
            # num_workers=0,
        )

        return SmellRecurrenceExperiment.RecurrenceExperiment(
            model,
            dataset,
            context,
            client_name=cid,
            lr=LEARNING_RATE,
            pre_train_lr=PRE_TRAIN_LEARNING_RATE,
            epochs=NUM_EPOCHS,
            weight_decay=WEIGHT_DECAY,
            lr_step=LEARNING_RATE_STEP,
            lr_gamma=LEARNING_RATE_GAMMA,
            embedding_size=EMB_DIM,
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
