from __future__ import annotations

import numpy as np
import random
import torch
import flautim as fl
import flautim.metrics as flm
import RecurrenceModel
import FedMetaRecurrenceDatasetnew
from meta_metric_model import MetaMetricModel
from ProxyNCAExperiment import (
    HFMetaMetricRecurrenceExperiment,
    HFMetaMetricConfig,
)

DATASET = "PAMAP"
NUM_CHANNELS = 6
BATCH_SIZE = 32
EPOCHS = 100


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

    combined_train_data = {
        "data": torch.cat([f["data"] for f in files], dim=0),
        "labels": torch.cat([f["labels"] for f in files], dim=0),
    }

    combined_test_data = {
        "data": torch.cat([f["data"] for f in test_files], dim=0),
        "labels": torch.cat([f["labels"] for f in test_files], dim=0),
    }

    dataset = FedMetaRecurrenceDatasetnew.RecurrenceDatasetnew(
        train_file=combined_train_data,
        test_file=combined_test_data,
        augment_multiplier=2,
        use_crop=False,
        use_erase=False,
        use_jitter=False,
        use_rotation=True,
        batch_size=BATCH_SIZE,
    )
    fl.log("Dataset created")

    backbone = RecurrenceModel.RecurrenceModel(
        context,
        num_classes=NUM_CLASSES,
        num_channels=NUM_CHANNELS,
        suffix=0,
    )
    emb_dim = int(backbone.base_model.last_channel)

    model = MetaMetricModel(
        context,
        backbone=backbone,
        num_classes=NUM_CLASSES,
        emb_dim=emb_dim,
        suffix=0,
    )

    fl.log("Model created")

    cfg = HFMetaMetricConfig(
        alpha=1e-4,
        inner_epochs=1,
        lambda_offset=1e-3,
        temperature=0.1,
        normalize_embeddings=True,
        normalize_proxies=True,
        use_amp=True,
    )

    experiment = HFMetaMetricRecurrenceExperiment(
        model=model,
        dataset=dataset,
        context=context,
        epochs=EPOCHS,
        client_name=0,
        cfg=cfg,
    )

    fl.log("Experiment created")

    def balanced_accuracy(y_true, y_pred):
        # Flatten and convert to numpy array of scalars
        y_true = np.array([int(y.item() if hasattr(y, "item") else y) for y in y_true])
        y_pred = np.array([int(y.item() if hasattr(y, "item") else y) for y in y_pred])

        classes = np.unique(y_true)
        recalls = []

        for cls in classes:
            tp = np.sum((y_pred == cls) & (y_true == cls))
            fn = np.sum((y_pred != cls) & (y_true == cls))
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            recalls.append(recall)

        return np.mean(recalls)

    flm.Metrics.balanced_accuracy = balanced_accuracy

    experiment.run(metrics={"ACCURACY": flm.Metrics.balanced_accuracy})

    fl.log("Starting HFMetaMetric Centralized Experiment")
