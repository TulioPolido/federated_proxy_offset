import SmellRecurrenceModel
import SmellRecurrenceDataset
import SmellRecurrenceExperiment
import flwr
import flautim as fl
import flautim.metrics as flm
import torch
import numpy as np
import random

DATASET = "PAMAP"
NUM_CHANNELS = 6
BATCH_SIZE = 32
NUM_ROUNDS = 175
NUM_EPOCHS = 10
EMB_DIM = 64


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

        fl.log("Files loaded")
        # Concatenate
        train_data = {
            "data": torch.cat([d["data"] for d in train_files], dim=0),
            "labels": torch.cat([d["labels"] for d in train_files], dim=0),
        }

        test_data = {
            "data": torch.cat([d["data"] for d in test_files], dim=0),
            "labels": torch.cat([d["labels"] for d in test_files], dim=0),
        }

        dataset = SmellRecurrenceDataset.RecurrenceDataset(
            train_data,
            test_data,
            batch_size=BATCH_SIZE,
            augment_multiplier=2,
            use_crop=False,
            use_erase=False,
            use_jitter=False,
            use_rotation=True,
        )

        fl.log("Dataset created")
        model = SmellRecurrenceModel.RecurrenceModel(
            context, emb_dim=EMB_DIM, num_channels=NUM_CHANNELS, suffix=0
        )

        # Run federated learning
        experiment = SmellRecurrenceExperiment.RecurrenceExperiment(
            model,
            dataset,
            context,
            lr=LEARNING_RATE,
            epochs=NUM_EPOCHS,
            weight_decay=WEIGHT_DECAY,
            lr_step=LEARNING_RATE_STEP,
            lr_gamma=LEARNING_RATE_GAMMA,
            embedding_size=EMB_DIM
        )

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
