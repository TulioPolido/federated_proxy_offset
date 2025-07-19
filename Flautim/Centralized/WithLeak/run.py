
import flautim as fl
import RecurrenceModel
import CentRecurrenceDatasetWithLeak
import CentRecurrenceExperiment
import numpy as np
import pandas as pd
import flautim.metrics as flm

import torch

BATCH_SIZE = 64
LEARNING_RATE = 5e-4
LEARNING_RATE_STEP=10
LEARNING_RATE_GAMMA=0.9
NUM_EPOCHS = 100

if __name__ == "__main__":
    context = fl.init()
    fl.log(f"Flautim inicializado!!!")
    def load_client_file(n: int):
        path = f"./data/PAMAP_processed_user_absFalse_512_256_{n}_rgb_withGyro.pt"
        return torch.load(path)

    dicts = [load_client_file(n+1) for n in range(8)]

    data = {
        "data":   torch.cat([d["data"]   for d in dicts], dim=0),
        "labels": torch.cat([d["labels"] for d in dicts], dim=0),
    }

    dataset = CentRecurrenceDatasetWithLeak.RecurrenceDataset(
                data,
                batch_size=BATCH_SIZE
            )
    
    model = RecurrenceModel.RecurrenceModel(context)
    experiment = CentRecurrenceExperiment.RecurrenceExperiment(
        model,
        dataset,
        context,
        lr=LEARNING_RATE,
        epochs=NUM_EPOCHS,
        lr_step=LEARNING_RATE_STEP,
        lr_gamma=LEARNING_RATE_GAMMA
        )
    
    def balanced_accuracy(y_true, y_pred):
        # Flatten and convert to numpy array of scalars
        y_true = np.array([int(y.item() if hasattr(y, 'item') else y) for y in y_true])
        y_pred = np.array([int(y.item() if hasattr(y, 'item') else y) for y in y_pred])

        classes = np.unique(y_true)
        recalls = []

        for cls in classes:
            tp = np.sum((y_pred == cls) & (y_true == cls))
            fn = np.sum((y_pred != cls) & (y_true == cls))
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            recalls.append(recall)

        return np.mean(recalls)
    
    flm.Metrics.balanced_accuracy = balanced_accuracy

    experiment.run(metrics = {'ACCURACY': flm.Metrics.balanced_accuracy})