from flautim.pytorch.common import run_federated, weighted_average
import RecurrenceModel
import RecurrenceDataset
import RecurrenceExperiment
import flwr
from flwr.common import ndarrays_to_parameters
from flwr.server import ServerConfig, ServerAppComponents
import flautim as fl
import torch

NUM_CLASSES = 7
NUM_CHANNELS = 6
LEARNING_RATE = 5e-4
BATCH_SIZE = 16
EPOCHS = 5
ROUNDS = 40


def fit_config(server_round):
    return {"server_round": server_round}


def generate_server_fn(context, eval_fn):
    def create_server_fn(context_flwr):
        try:
            net = RecurrenceModel.RecurrenceModel(
                context,
                num_classes=NUM_CLASSES,
                num_channels=NUM_CHANNELS,
                suffix = 0
            )
            params = ndarrays_to_parameters(net.get_parameters())

            strategy = flwr.server.strategy.FedAvg(
                initial_parameters=params,
                evaluate_metrics_aggregation_fn=weighted_average,
                fraction_fit=1.0,
                fraction_evaluate=1.0,
                evaluate_fn=eval_fn,
                on_fit_config_fn=fit_config,
                on_evaluate_config_fn=fit_config
            )

            config = ServerConfig(num_rounds=ROUNDS)
            return ServerAppComponents(config=config, strategy=strategy)
        except Exception as e:
            print(f"Error in create_server_fn: {e}")

    return create_server_fn


def generate_client_fn(context, files):
    def create_client_fn(context_flwr):
        try:
            cid = int(context_flwr.node_config["partition-id"])
            file = files[cid]
            model = RecurrenceModel.RecurrenceModel(
                context,
                num_classes=NUM_CLASSES,
                num_channels=NUM_CHANNELS,
                suffix = cid
            )
            dataset = RecurrenceDataset.RecurrenceDataset(
                file,
                batch_size=BATCH_SIZE,
                shuffle=True
            )
            return RecurrenceExperiment.RecurrenceExperiment(
                model,
                dataset,
                context,
                lr=LEARNING_RATE,
                epochs=EPOCHS
            ).to_client()
        except Exception as e:
            print(f"Error in create_client_fn: {e}")

    return create_client_fn


def evaluate_fn(context, test_data):
    def fn(server_round, parameters, config):
        try:
            model = RecurrenceModel.RecurrenceModel(
                context,
                num_classes=NUM_CLASSES,
                num_channels=NUM_CHANNELS
            )
            model.set_parameters(parameters)
            dataset = RecurrenceDataset.RecurrenceDataset(
                test_data,
                batch_size=BATCH_SIZE,
                shuffle=False
            )
            experiment = RecurrenceExperiment.RecurrenceExperiment(
                model,
                dataset,
                context,
                lr=LEARNING_RATE,
                epochs=EPOCHS
            )

            config["server_round"] = server_round
            loss, _, return_dic = experiment.evaluate(parameters, config)
            return loss, return_dic
        except Exception as e:
            print(f"Error in evaluate_fn: {e}")
    return fn


if __name__ == "__main__":
    context = fl.init()

    # Number of training clients
    num_clients = 6

    # Load training client files
    files = [
        torch.load(
            f"./data/PAMAP_processed_user_absFalse_512_256_{i+1}_rgb_withGyro.pt"
        )
        for i in range(num_clients)
    ]

    # Load and combine test files
    test_files = [
        torch.load(
            f"./data/PAMAP_processed_user_absFalse_512_256_{i+1}_rgb_withGyro.pt"
        )
        for i in range(num_clients, num_clients + 2)
    ]

    combined_test_data = {
        'data': torch.cat([f['data'] for f in test_files], dim=0),
        'labels': torch.cat([f['labels'] for f in test_files], dim=0),
    }
    client_fn_callback = generate_client_fn(context, files)
    evaluate_fn_callback=evaluate_fn(context, combined_test_data)
    server_fn_callback=generate_server_fn(context, eval_fn=evaluate_fn_callback)
    # Run federated learning
    run_federated(
        client_fn=client_fn_callback,
        server_fn=server_fn_callback, num_clients = num_clients, num_rounds = 40
    )
