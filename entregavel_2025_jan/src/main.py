import time
import os

from datetime import datetime
import json

from collections import OrderedDict
from typing import List, Tuple

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from torchsummary import summary
from torchvision import models

from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, confusion_matrix, ConfusionMatrixDisplay

import flwr
from flwr.client import Client, NumPyClient
from flwr.server import ServerConfig
from flwr.common import Metrics
from flwr.server.strategy import FedAvg
from flwr.simulation import start_simulation

DEVICE = torch.device("cpu")  # Try "cuda" to train on GPU
print(f"Training on {DEVICE}")
print(f"Flower {flwr.__version__} / PyTorch {torch.__version__}")

# Define the path to the dataset and the sequence length
PATH = '../datasets/'
NUM_CLIENTS_TRAIN = 27
NUM_CLIENTS_TEST = 9
NUM_CLASSES = 4
LEARNING_RATE = 5e-6
NUM_ROUNDS = 30
NUM_EPOCHS = 10
WINDOW_SIZE = 128
STRIDE = 48
BATCH_SIZE = 32

# Function to normalize columns
def normalize_group(group):
    group[['x', 'y', 'z']] = (group[['x', 'y', 'z']] - group[['x', 'y', 'z']].mean()) / group[['x', 'y', 'z']].std()
    return group

# Load data from file
df = pd.read_csv(f"{PATH}/WISDM_ar_v1.1_balanced.csv", header=None, names=['user_id', 'activity', 'timestamp', 'x', 'y', 'z'])
if(NUM_CLASSES == 4):
    # Filter for specific activities
    df = df.loc[
        (df['activity'] == 'Walking') | 
        (df['activity'] == 'Jogging') | 
        (df['activity'] == 'Sitting') | 
        (df['activity'] == 'Standing') 
    ].reset_index(drop=True)

df = df.groupby('user_id').apply(normalize_group)
    
# Map activities to integers if not already done
activity_mapping = {activity: idx for idx, activity in enumerate(df['activity'].unique())}
df['activity'] = df['activity'].map(activity_mapping)

# Split the dataset by user_id
user_ids = df['user_id'].unique()

def recurrence_plot_vector(data, eps):
    """
    Generate a recurrence plot for a vector sequence using a specified threshold.
    
    Parameters:
    - data: torch.Tensor, shape (N, D), where N is the number of vectors and D is their dimensionality.
    - eps: float, the threshold for recurrence.

    Returns:
    - recurrence_matrix: torch.Tensor, shape (N, N), binary recurrence matrix.
    """
    # Ensure the input is a tensor on the correct device
    if not isinstance(data, torch.Tensor):
        raise ValueError("Input data must be a PyTorch tensor.")
    
    # Compute pairwise Euclidean distances using PyTorch broadcasting
    data = data.to(torch.float32)  # Ensure correct dtype for computation
    distances = torch.cdist(data, data, p=2)  # Compute pairwise distances

    # Apply the threshold
    recurrence_matrix = (distances <= eps).int()

    return recurrence_matrix


def preprocess_and_save_recurrence_plots(df, thresholds, sequence_length=50, stride=1, save_path="processed_data.pt"):
    """
    Preprocess the dataset to generate recurrence plots and save them.

    Parameters:
    - df: DataFrame containing the data.
    - thresholds: list of floats, thresholds for recurrence plots.
    - sequence_length: int, length of each sequence.
    - stride: int, step size between sequences.
    - save_path: str, path to save the preprocessed data.

    Returns:
    - None
    """
    try:
        processed_data = []
        labels = []

        if os.path.exists(save_path):
            print(f"File {save_path} already exists. Skipping preprocessing.")
            return

        for idx in range(0, len(df) - sequence_length + 1, stride):
            # Select a window of data points
            window = df.iloc[idx:idx + sequence_length]

            # Extract x, y, z values
            sequence = torch.tensor(window[['x', 'y', 'z']].values, dtype=torch.float32)

            # Generate multi-channel recurrence plots
            recurrence_plots = torch.stack(
                [recurrence_plot_vector(sequence, threshold) for threshold in thresholds], dim=-1
            )

            # The label is the activity of the last element in the sequence
            label = int(window.iloc[-1]['activity'])

            processed_data.append(recurrence_plots)
            labels.append(label)

        # Convert to tensors
        processed_data = torch.stack(processed_data)  # Shape: (num_sequences, sequence_length, sequence_length, num_channels)
        labels = torch.tensor(labels, dtype=torch.long)  # Shape: (num_sequences,)

        # Save to disk
        torch.save({'data': processed_data, 'labels': labels}, save_path)
        print(f"Preprocessed data saved to {save_path}")

    except Exception as e:
        print(f"An error occurred during preprocessing: {e}")



THRESHOLDS = [
    1.0000, 
    0.0187, 
    0.0113, 
    1.0000
]  # List of thresholds

for user_id in df['user_id'].unique():
    user_df = df[df['user_id'] == user_id]
    preprocess_and_save_recurrence_plots(
        user_df,
        thresholds=THRESHOLDS,
        sequence_length=WINDOW_SIZE,
        stride=STRIDE,
        save_path=f"{PATH}/processed_user_{user_id}.pt"
    )


class WISDMDataset(Dataset):
    def __init__(self, processed_data_path):
        """
        Initialize the dataset with precomputed recurrence plots.

        Parameters:
        - processed_data_path: str, path to the preprocessed data file.
        """
        # Load the preprocessed data
        data = torch.load(processed_data_path)
        self.data = data['data'].float().permute(0, 3, 1, 2)
        self.labels = data['labels'].long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {'data': self.data[idx], 'label': self.labels[idx]}


def load_client_data(partition_id: int, batch_size=32, split=True):
    try:
        if not isinstance(partition_id, int):  # Ensure partition ID is an integer
            partition_id = int(partition_id)

        # Load preprocessed data for this client
        processed_data_path = f"{PATH}/processed_user_{partition_id}.pt"
        dataset = WISDMDataset(processed_data_path)

        if(not split):
            return DataLoader(dataset, batch_size=batch_size, shuffle=True), None
            
        # Split the dataset into train and validation sets (80% train, 20% validation)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        # Create DataLoader objects
        trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        return trainloader, valloader
    except Exception as e:
        print(f"An error occurred (load_client_data): {e}")
        return None, None


train_loader, val_loader = load_client_data(partition_id=8)
batch = next(iter(train_loader))
data, labels = batch["data"], batch["label"]
x_shape = data.shape
y_shape = labels.shape

def load_test_data():
    """
    Load test data from preprocessed recurrence plot files.

    Returns:
    - testloader: DataLoader for the test data.
    """
    try:
        test_clients = user_ids[-NUM_CLIENTS_TEST:]
        print(f"Test clients: {test_clients}")
        
        datasets = []

        for client_id in test_clients:
            # Construct the file path for the preprocessed data
            file_path = f"{PATH}processed_user_{client_id}.pt"
            try:
                # Create a dataset object for the client's preprocessed data
                dataset = WISDMDataset(processed_data_path=file_path)
                datasets.append(dataset)
            except FileNotFoundError:
                print(f"Preprocessed file not found for client {client_id}: {file_path}")
                continue

        if not datasets:
            raise ValueError("No preprocessed data found for any test client.")

        # Concatenate all client datasets into one
        combined_dataset = torch.utils.data.ConcatDataset(datasets)

        # Create DataLoader
        testloader = DataLoader(combined_dataset, batch_size=BATCH_SIZE, shuffle=False)

        return testloader

    except Exception as e:
        print(f"An error occurred in load_test_data: {e}")
        return None


test_loader = load_test_data()
batch = next(iter(test_loader))
data, labels = batch["data"], batch["label"]
x_shape = data.shape
y_shape = labels.shape
x_shape, y_shape

class ModifiedMobileNetV2(nn.Module):
    def __init__(self, num_classes):
        super(ModifiedMobileNetV2, self).__init__()
        # Load MobileNetV2 with pretrained weights
        self.base_model = models.mobilenet_v2(weights=None)
        self.base_model.load_state_dict(torch.load(f"{PATH}/mobilenet_v2-b0353104.pth"))
        
        # Modify the input layer to handle 4 channels instead of 3
        self.base_model.features[0][0] = nn.Conv2d(
            in_channels=num_classes, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False
        )
        
        # Replace the final layer to match the number of classes
        self.base_model.classifier[1] = nn.Linear(self.base_model.last_channel, num_classes)
        
    def forward(self, x):
        return self.base_model(x)

def test(net, testloader):
    """Evaluate the PEN network on the entire test set and return various metrics."""
    try:
        criterion = torch.nn.CrossEntropyLoss()
        correct, total, loss = 0, 0, 0.0
        net.eval()
        
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for batch in testloader:
                data, labels = batch["data"].to(DEVICE), batch["label"].to(DEVICE)
                outputs = net(data)

                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        # Calculate metrics
        loss /= len(testloader.dataset)
        balanced_acc = balanced_accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
        conf_matrix = confusion_matrix(all_labels, all_predictions, normalize='true').round(2)

        return loss, balanced_acc, f1, precision, conf_matrix

    except Exception as e:
        print(f"An error occurred (test): {e}")
        return None, None, None, None, None


def train(net, trainloader, epochs: int, verbose=False):
    """
    Train the model with early stopping and evaluate it at each epoch.

    Parameters:
    - net: PyTorch model to be trained.
    - trainloader: DataLoader object containing the training data.
    - epochs: Number of epochs to train the model.
    - criterion: Loss function. If None, defaults to CrossEntropyLoss.
    - optimizer: Optimization algorithm. If None, defaults to Adam optimizer.
    - verbose: Boolean indicating whether to print detailed information during training.

    Returns:
    - train_loss: List of training losses for each epoch.
    """
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    
    net.train()
    train_losses = []

    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        start_epoch_time = time.time()

        for batch in trainloader:
            data, labels = batch["data"].to(DEVICE), batch["label"].to(DEVICE)
            optimizer.zero_grad()
            outputs = net(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()

        avg_loss = epoch_loss / len(trainloader)
        train_losses.append(avg_loss)

        end_epoch_time = time.time()
        epoch_time = end_epoch_time - start_epoch_time

        if verbose:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Time: {epoch_time:.2f} seconds')

    return train_losses

def freeze_and_clean_layers(model, unfrozen_layers):
    frozen = 0
    
    # Freeze all layers in the model initially
    for param in model.base_model.parameters():
        param.requires_grad = False
    
    # Unfreeze and clean the classifier layers first
    for layer in reversed(list(model.base_model.classifier.children())):
        if frozen < unfrozen_layers:
            for param in layer.parameters():
                param.requires_grad = True
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()  # Clean weights
            frozen += 1

    # Unfreeze and clean the feature layers next
    for layer in reversed(list(model.base_model.features.children())):
        if frozen < unfrozen_layers:
            for param in layer.parameters():
                param.requires_grad = True
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()  # Clean weights
            frozen += 1


model = ModifiedMobileNetV2(NUM_CLASSES).to(DEVICE)
UNFROZEN_LAYERS = len(list(model.base_model.parameters())) * 0.5
freeze_and_clean_layers(model, UNFROZEN_LAYERS)

train_losses = train(
    net=model,
    trainloader=train_loader,
    epochs=NUM_EPOCHS,
    verbose=True
)

print("Training complete.")

_, balanced_acc, precision, f1, conf_matrix = test(model, test_loader)
print(f"  Balanced Accuracy: {balanced_acc:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  F1 Score: {f1:.4f}")
print(f"  Confusion Matrix: \n{conf_matrix.round(2)}")

def set_parameters(net, parameters: List[np.ndarray]):
    try:
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.from_numpy(np.copy(v)) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)
    except Exception as e:
        print(f"An error occurred (set_parameters): {e}")


def get_parameters(net) -> List[np.ndarray]:
    try:
        return [val.cpu().numpy() for _, val in net.state_dict().items()]
    except Exception as e:
        print(f"An error occurred (get_parameters): {e}")
        return None

class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        train_losses = train(
            net=self.net,
            trainloader=self.trainloader,
            epochs=NUM_EPOCHS,
            verbose=False
        )
        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)

        # If valloader and trainloader are the same, then this client will
        # be tested with the global test dataset
        if(self.valloader == self.trainloader):
            loss, balanced_acc, f1, precision, conf_matrix = test(self.net, test_loader)
        else:
            loss, balanced_acc, f1, precision, conf_matrix = test(self.net, self.valloader)
            
        return float(loss), len(self.valloader), {"accuracy": float(balanced_acc), "f1score": float(f1), "precision": float(precision)}

def client_fn(context) -> Client:
    """Create a Flower client representing a single user."""
    try:
        # Load model
        net = ModifiedMobileNetV2(NUM_CLASSES).to(DEVICE)
        freeze_and_clean_layers(net, UNFROZEN_LAYERS)
        
        partition_id = int(context)
        client_id = int(partition_id) + 1

        # Split = False doesn't create a validation dataset from this client
        # Then all data is used to train
        trainloader, valloader = load_client_data(partition_id=client_id, split=False)

        # If valloader is None, then this client will not use
        # it to test its local model. It will use the global test dataset
        if(valloader == None):
            valloader = trainloader
        
        return FlowerClient(net, trainloader, trainloader).to_client()
    except Exception as e:
        print(f"An error occurred (client_fn): {e}")
        return None

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    try:
        # Multiply accuracy of each client by number of examples used
        accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
        f1s = [num_examples * m["f1score"] for num_examples, m in metrics]
        precisions = [num_examples * m["precision"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]

        # Aggregate and return custom metric (weighted average)
        return {"accuracy": sum(accuracies) / sum(examples), "f1score": sum(f1s) / sum(examples), "precision": sum(precisions) / sum(examples)}
    except Exception as e:
        print(f"An error occurred (weighted_average): {e}")
        return None

def server_evaluate_fn(server_round:int, parameters, config):
    try:
        # Load model
        model = ModifiedMobileNetV2(NUM_CLASSES).to(DEVICE)
        freeze_and_clean_layers(model, UNFROZEN_LAYERS)

        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.from_numpy(np.copy(v)) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        
        loss, balanced_acc, f1, precision, conf_matrix = test(model, test_loader)

        return float(loss), {"accuracy": float(balanced_acc), "f1score": float(f1), "precision": float(precision), "conf_matrix": conf_matrix}
    except Exception as e:
        print(f"An error occurred (server_evaluate_fn): {e}")
        return None, None


# Create FedAvg strategy
strategy = FedAvg(
    fraction_fit=0.2,  # Sample % of available clients for training
    fraction_evaluate=0.1,  # Sample % of available clients for evaluation
    min_fit_clients=1,  # sample  clients for training
    min_evaluate_clients=1,  # sample clients for evaluation
    min_available_clients=1,  # Wait until all clients are available
    evaluate_metrics_aggregation_fn=weighted_average,
    evaluate_fn=server_evaluate_fn,
)

# Configure the server for 5 rounds of training
config = ServerConfig(num_rounds=NUM_ROUNDS)

hist = start_simulation(
    num_clients=NUM_CLIENTS_TRAIN,
    client_fn=client_fn,
    strategy=strategy,
    config=config,
    client_resources={'num_cpus': 8, 'num_gpus': 0.0}
)

best_tuple = max(hist.metrics_centralized['accuracy'], key=lambda x: x[1])  # Compare by the second element

best_index, best_value = best_tuple
print(f"Accuracy: {hist.metrics_centralized['accuracy'][best_index][1]}")
print(f"F1 Score: {hist.metrics_centralized['f1score'][best_index][1]}")
print(f"Precision: {hist.metrics_centralized['precision'][best_index][1]}")

def plot_metrics_by_round(data, metric, legend, is_loss=False):
    df = pd.DataFrame(data, columns=["Round", "Accuracy"])
    # Set the style
    # Set the style
    sns.set_theme(style="dark")

    # Create the line plot with customizations
    plt.figure(figsize=(10, 6))
    sns.lineplot(x="Round", y="Accuracy", data=df, marker="o", color="royalblue", linewidth=2.5)

    # Customize the title and labels
    plt.title(f"{metric} per Round ({legend})", fontsize=16, fontweight='bold')
    plt.xlabel("Round", fontsize=14)
    plt.ylabel(metric, fontsize=14)

    # Customize the grid
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Customize the ticks
    plt.xticks(ticks=np.arange(df["Round"].min()-1, df["Round"].max() +1 , 10), fontsize=12)  # Ensure only integer rounds are savefign on x-axis
    plt.yticks(fontsize=12)

    # Add a range for the y-axis for better visibility
    if(not is_loss):
        plt.ylim(0, 1)

    # Display the plot
    plt.savefig(f'{metric}.png')

plot_metrics_by_round(hist.losses_distributed, 'Loss', 'Server', True)

def plot_multiple_metrics_by_round(data1, data2, data3, legend):
    # Convert data to DataFrames
    df1 = pd.DataFrame(data1, columns=["Round", "Metric"])
    df2 = pd.DataFrame(data2, columns=["Round", "Metric"])
    df3 = pd.DataFrame(data3, columns=["Round", "Metric"])
    
    # Set the style
    sns.set_theme(style="dark")
    
    # Create the main plot
    plt.figure(figsize=(10, 6))
    
    # Plot each metric with different styles
    sns.lineplot(x="Round", y="Metric", data=df1, marker="o", color="royalblue", linewidth=2.5, label="Accuracy")
    sns.lineplot(x="Round", y="Metric", data=df2, marker="s", color="forestgreen", linewidth=2.5, label="F1 Score")
    sns.lineplot(x="Round", y="Metric", data=df3, marker="^", color="darkorange", linewidth=2.5, label="Precision")
    
    # Customize the title and labels
    plt.title(f"Metrics per Round ({legend})", fontsize=16, fontweight='bold')
    plt.xlabel("Round", fontsize=14)
    plt.ylabel("Value", fontsize=14)
    
    # Customize the grid
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Calculate and set x-axis ticks
    round_ticks = range(0, max(df1["Round"].max(), df2["Round"].max(), df3["Round"].max()) + 10, 10)
    plt.xticks(ticks=round_ticks, fontsize=12)  
    
    # Customize the y-axis ticks
    plt.yticks(fontsize=12)
    
    # Set y-axis limits
    plt.ylim(0.2, 1.0)
    
    # Add a legend
    plt.legend(title="Metrics", title_fontsize='13', fontsize='12')
    
    # Inset plot for the last 5 values
    inset_axes = plt.gca().inset_axes([0.6, 0.1, 0.35, 0.35])  # Position inset: [left, bottom, width, height]
    sns.lineplot(x="Round", y="Metric", data=df1.tail(5), marker="o", color="royalblue", linewidth=2.0, ax=inset_axes)
    sns.lineplot(x="Round", y="Metric", data=df2.tail(5), marker="s", color="forestgreen", linewidth=2.0, ax=inset_axes)
    sns.lineplot(x="Round", y="Metric", data=df3.tail(5), marker="^", color="darkorange", linewidth=2.0, ax=inset_axes)
    
    # Annotate the last point in the inset
    for df, color, shift in zip([df1, df2, df3], ["royalblue", "forestgreen", "darkorange"], [0, 0.5, 1]):
        last_point = df.tail(5).iloc[-1]
        inset_axes.text(last_point["Round"] + shift, last_point["Metric"], 
                        f"{last_point['Metric']:.2f}", 
                        fontsize=10, fontweight='bold', color=color,
                        ha='right', va='bottom')
    
    # Customize the inset plot
    inset_axes.set_xlim(df1["Round"].max() - 4, df1["Round"].max())  # Set x-axis range for last 5 rounds
    #inset_axes.set_ylim(0.2, 1.0)  # Match y-axis limits to the main plot
    inset_axes.set_title("Last 5 Rounds", fontsize=12)
    inset_axes.set_ylabel("Value")
    inset_axes.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Display the plot
    plt.savefig('metrics.png')

plot_multiple_metrics_by_round(hist.metrics_centralized['accuracy'], hist.metrics_centralized['f1score'], hist.metrics_centralized['precision'], 'Server')

disp = ConfusionMatrixDisplay(confusion_matrix=hist.metrics_centralized['conf_matrix'][best_index][1], display_labels=list(activity_mapping.keys()))
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.xticks(rotation=45)
plt.savefig('confusion_matrix.png')

# Define the folder structure and ensure it exists
os.makedirs(PATH, exist_ok=True)  # Create folder if it doesn't exist

# Function to manually serialize the flwr.server.History object
def serialize_history(hist):
    # Prepare a dictionary to hold the serialized data
    serialized_data = {}

    # Serialize loss data
    serialized_data['loss'] = {
        'distributed': [v for k, v in hist.losses_distributed],
        'centralized': [v for k, v in hist.losses_centralized]
    }

    # Serialize metrics data
    serialized_data['metrics'] = {
        'distributed': {
            'accuracy': [v for k, v in hist.metrics_distributed['accuracy']],
            'f1score': [v for k, v in hist.metrics_distributed['f1score']],
            'precision': [v for k, v in hist.metrics_distributed['precision']]
        },
        'centralized': {
            'accuracy': [v for k, v in hist.metrics_centralized['accuracy']],
            'f1score': [v for k, v in hist.metrics_centralized['f1score']],
            'precision': [v for k, v in hist.metrics_centralized['precision']],
            'conf_matrix': hist.metrics_centralized['conf_matrix'][best_index][1].tolist()
        }
    }

    # Return the dictionary instead of converting it to JSON here
    return serialized_data

# Save the serialized history to a JSON file
def save_history_as_json(hist):
    # Generate a filename with the given format
    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{PATH}/recurrence_mobilenet_wisdm_{datetime_str}.json"
    
    # Serialize the history object
    serialized_hist = serialize_history(hist)
    
    # Save to JSON
    with open(filename, "w") as f:
        # Dump the dictionary directly to the JSON file
        json.dump(serialized_hist, f, indent=4)
    
    print(f"History saved to {filename}")

save_history_as_json(hist) # Ensure hist is a valid object you want to serialize.