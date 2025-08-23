import random

import torch
import numpy as np
import flautim as fl
import flautim.metrics as flm
import RecurrenceModel
import CentRecurrenceDataset
import CentRecurrenceExperiment

DATASET = "WISDM"
NUM_EPOCHS = 100
NUM_CHANNELS = 6
BATCH_SIZE = 32

if __name__ == "__main__":
    try:
        context = fl.init()
        fl.log("Flautim initalized!")

        # Generate a random seed
        seed = random.randint(0, 2**32 - 1)
        random.seed(seed)
        fl.log(f"Using random seed: {seed}")

        # All available file indices
        if DATASET == "PAMAP":
            LEARNING_RATE = 4.7e-3
            LEARNING_RATE_STEP = 6
            LEARNING_RATE_GAMMA = 0.808
            WEIGHT_DECAY = 6.7e-5
            WINDOW_SIZE = 1024
            STRIDE = 256
            NUM_CLASSES = 7

            all_files = [i for i in range(1, 9)]

            # Shuffle and split
            test_files = random.sample(all_files, 2)
            train_files = [f for f in all_files if f not in test_files]

            def load_client_file(n: int):
                path = f"./data/PAMAP_processed_user_absFalse_{WINDOW_SIZE}_{STRIDE}_{n}_rgb_withGyro.pt"
                return torch.load(path)

        elif DATASET == "WISDM":
            LEARNING_RATE = 6.9e-4
            LEARNING_RATE_STEP = 13
            LEARNING_RATE_GAMMA = 0.53
            WEIGHT_DECAY = 1.1e-5
            WINDOW_SIZE = 256
            STRIDE = 64
            NUM_CLASSES = 5

            all_files = [i for i in range(1, 52)]

            # Shuffle and split
            test_files = random.sample(all_files, 10)
            train_files = [f for f in all_files if f not in test_files]

            def load_client_file(n: int):
                path = f"./data/WISDM_processed_user_absFalse_{WINDOW_SIZE}_{STRIDE}_{n}_rgb_withGyro.pt"
                return torch.load(path)

        # Load subsets
        train_dicts = [load_client_file(n) for n in train_files]
        test_dicts = [load_client_file(n) for n in test_files]

        fl.log("Files loaded")
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
            train_data,
            test_data,
            batch_size=BATCH_SIZE,
            augment_multiplier=0,
            use_crop=False,
            use_erase=False,
            use_jitter=False,
            use_rotation=True,
        )

        fl.log("Dataset created")
        model = RecurrenceModel.RecurrenceModel(
            context, num_classes=NUM_CLASSES, num_channels=NUM_CHANNELS
        )

        fl.log("Model created")
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
        fl.log("Experiment created")

        def balanced_accuracy(y_true, y_pred):
            # Flatten and convert to numpy array of scalars
            y_true = np.array(
                [int(y.item() if hasattr(y, "item") else y) for y in y_true]
            )
            y_pred = np.array(
                [int(y.item() if hasattr(y, "item") else y) for y in y_pred]
            )

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
    except Exception as e:
        fl.log(f"Error in run.py: {e}")
