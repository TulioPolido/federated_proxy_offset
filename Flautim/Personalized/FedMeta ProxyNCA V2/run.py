from __future__ import annotations

import random
import torch
import flautim as fl

from flwr.server import ServerConfig, ServerAppComponents
from flwr.common import ndarrays_to_parameters

from flautim.pytorch.common import run_federated, weighted_average

import RecurrenceModel
from fedmeta_recurrence_dataset import RecurrenceDataset

from meta_metric_model import MetaMetricModel
from hf_meta_metric_experiment_recurrence import (
    HFMetaMetricRecurrenceExperiment,
    HFMetaMetricConfig,
)
from hf_meta_metric_strategy import HFMetaMetricStrategy


DATASET = "PAMAP"
NUM_CHANNELS = 6
BATCH_SIZE = 32
NUM_ROUNDS = 100
INNER_EPOCHS = 2
ALPHA = 5e-4
BETA = 1e-4


def fit_config(server_round: int):
    return {"server_round": server_round}


def generate_server_fn(context, eval_fn=None):
    def create_server_fn(context_flwr):
        # backbone = seu RecurrenceModel (MobileNetV2 modificado)
        backbone = RecurrenceModel.RecurrenceModel(
            context,
            num_classes=NUM_CLASSES,  # usado só para classifier (não é o embedding)
            num_channels=NUM_CHANNELS,
            suffix=0,
        )

        # Emb dim = 1280 no MobileNetV2
        emb_dim = int(backbone.base_model.last_channel)

        net = MetaMetricModel(
            context,
            backbone=backbone,
            num_classes=NUM_CLASSES,
            emb_dim=emb_dim,
            suffix=0,
        )
        params = ndarrays_to_parameters(net.get_parameters())

        strategy = HFMetaMetricStrategy(
            model=net,
            lr_outer=BETA,
            initial_parameters=params,
            evaluate_metrics_aggregation_fn=weighted_average,
            fraction_fit=0.4,
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
        train_file = files[cid]

        backbone = RecurrenceModel.RecurrenceModel(
            context,
            num_classes=NUM_CLASSES,
            num_channels=NUM_CHANNELS,
            suffix=cid,
        )
        emb_dim = int(backbone.base_model.last_channel)

        model = MetaMetricModel(
            context,
            backbone=backbone,
            num_classes=NUM_CLASSES,
            emb_dim=emb_dim,
            suffix=cid,
        )

        dataset = RecurrenceDataset(
            train_file=train_file,
            test_file=test_file,
            augment_multiplier=2,
            use_rotation=True,
            support_ratio=0.5,
            num_workers=0,
            batch_size=BATCH_SIZE,
        )

        cfg = HFMetaMetricConfig(
            alpha=ALPHA,
            inner_epochs=INNER_EPOCHS,
            lambda_offset=1e-3,
            temperature=0.1,
            normalize_embeddings=True,
            normalize_proxies=True,
            use_amp=True,
        )

        return HFMetaMetricRecurrenceExperiment(
            model=model,
            dataset=dataset,
            context=context,
            client_name=cid,
            cfg=cfg,
        ).to_client()

    return create_client_fn


if __name__ == "__main__":
    context = fl.init()
    seed = 2270238713
    random.seed(seed)
    fl.log(f"Using random seed: {seed}")

    if DATASET == "WISDM":
        TOTAL_USERS = 10
        NUM_CLIENTS = 8
        TEST_USERS = TOTAL_USERS - NUM_CLIENTS
        NUM_CLASSES = 5
        WINDOW_SIZE = 256
        STRIDE = 64

        all_files = [i for i in range(1, TOTAL_USERS + 1) if i not in [38, 41, 44]]
        test_users = random.sample(all_files, TEST_USERS)
        train_users = [f for f in all_files if f not in test_users]

        fl.log(f"Train users: {train_users}")
        fl.log(f"Test users: {test_users}")

        files = [
            torch.load(
                f"./data/WISDM_processed_user_absFalse_{WINDOW_SIZE}_{STRIDE}_{i}_rgb_withGyro.pt"
            )
            for i in train_users
        ]
        test_files = [
            torch.load(
                f"./data/WISDM_processed_user_absFalse_{WINDOW_SIZE}_{STRIDE}_{i}_rgb_withGyro.pt"
            )
            for i in test_users
        ]

    elif DATASET == "PAMAP":
        TOTAL_USERS = 8
        NUM_CLIENTS = 7
        TEST_USERS = TOTAL_USERS - NUM_CLIENTS
        NUM_CLASSES = 7
        WINDOW_SIZE = 1024
        STRIDE = 256

        all_files = [i for i in range(1, TOTAL_USERS + 1)]
        test_users = [7]
        train_users = [i for i in all_files if i not in test_users]

        fl.log(f"Train users: {train_users}")
        fl.log(f"Test users: {test_users}")

        files = [
            torch.load(
                f"./data/PAMAP_processed_user_absFalse_{WINDOW_SIZE}_{STRIDE}_{i}_rgb_withGyro.pt"
            )
            for i in train_users
        ]
        test_files = [
            torch.load(
                f"./data/PAMAP_processed_user_absFalse_{WINDOW_SIZE}_{STRIDE}_{i}_rgb_withGyro.pt"
            )
            for i in test_users
        ]

    combined_test_data = {
        "data": torch.cat([f["data"] for f in test_files], dim=0),
        "labels": torch.cat([f["labels"] for f in test_files], dim=0),
    }

    client_fn_callback = generate_client_fn(context, files, combined_test_data)
    server_fn_callback = generate_server_fn(context, eval_fn=None)

    clients_resources = {"num_cpus": 1, "num_gpus": 0.15}

    fl.log("Starting HFMetaMetric Federated Experiment")
    run_federated(
        client_fn=client_fn_callback,
        server_fn=server_fn_callback,
        num_clients=NUM_CLIENTS,
        num_rounds=NUM_ROUNDS,
        clients_resources=clients_resources,
    )
