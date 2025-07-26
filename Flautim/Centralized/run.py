import flautim as fl
import RecurrenceModel
import CentRecurrenceDataset
import CentRecurrenceExperiment
import numpy as np
import pandas as pd
import flautim.metrics as flm

import torch

LEARNING_RATE = 4.7e-3
LEARNING_RATE_STEP = 6
LEARNING_RATE_GAMMA = 0.808
NUM_EPOCHS = 100
NUM_CLASSES = 7
NUM_CHANNELS = 6
WINDOW_SIZE = 1024
STRIDE = 256
BATCH_SIZE = 32
WEIGHT_DECAY = 6.7e-5


if __name__ == "__main__":
    context = fl.init()
    fl.log(f"Flautim inicializado!!!")
    train_files = [8, 7, 6, 5, 4, 1]
    test_files = [2, 3]

    def load_client_file(n: int):
        path = f"./data/PAMAP_processed_user_absFalse_{WINDOW_SIZE}_{STRIDE}_{n}_rgb_withGyro.pt"
        return torch.load(path)

    # Load subsets
    train_dicts = [load_client_file(n) for n in train_files]
    test_dicts = [load_client_file(n) for n in test_files]

    # Concatenate
    train_data = {
        "data": torch.cat([d["data"] for d in train_dicts], dim=0),
        "labels": torch.cat([d["labels"] for d in train_dicts], dim=0),
    }

    test_data = {
        "data": torch.cat([d["data"] for d in test_dicts], dim=0),
        "labels": torch.cat([d["labels"] for d in test_dicts], dim=0),
    }

    dataset = CentRecurrenceDataset.RecurrenceDataset(
        train_data, test_data, batch_size=BATCH_SIZE
    )

    model = RecurrenceModel.RecurrenceModel(
        context, num_classes=NUM_CLASSES, num_channels=NUM_CHANNELS
    )
    experiment = CentRecurrenceExperiment.RecurrenceExperiment(
        model,
        dataset,
        context,
        lr=LEARNING_RATE,
        epochs=NUM_EPOCHS,
        lr_step=LEARNING_RATE_STEP,
        lr_gamma=LEARNING_RATE_GAMMA,
        weight_decay=WEIGHT_DECAY,
    )

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
