from collections import OrderedDict
from logging import ERROR, INFO, WARNING
from typing import List, Optional, Tuple, Union, Dict

import flwr
import numpy as np
import torch
from flwr.common import (
    EvaluateRes,
    NDArrays,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg


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

    # ---- ASSIGN GRADS ----
    for (name, param), grad_ins in zip(net.named_parameters(), gradients_aggregated):
        param.grad = torch.tensor(grad_ins, dtype=param.dtype)

    # ---- OPTIMIZER STEP ----
    optimizer.step()
    optimizer.zero_grad()

    weights_prime = [val.cpu().numpy() for _, val in net.state_dict().items()]
    return weights_prime


class FedMeta(flwr.server.strategy.FedAvg):
    """Federated Meta-Learning Strategy (MAML-style)."""


    def __init__(self, model, lr_inner=1e-4, lr_outer=1e-4, smoothing_weight=0.95, weight_decay=0.001, use_ema=False, **kwargs):
        super().__init__(**kwargs)  # forward other FedAvg args
        self.net = model
        self.beta = lr_outer
        self.alpha = lr_inner
        self.smoothing_weight = smoothing_weight
        self.weight_decay = weight_decay
        self.use_ema = use_ema
        self.ema_loss = None
        self.ema_metrics = {}

    def aggregate_fit(self, server_round, results, failures):
        for failure in failures:
            log(ERROR, f"Fit Failure: {failure}")

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

            try:
                with np.load(file_path) as data:
                    grads = [data[key] for key in data]
            except FileNotFoundError:
                log(ERROR, f"File not found: {file_path}")
                raise

            grads_results.append((grads, fit_res.num_examples))

        # Aggregate gradients
        gradients_aggregated = aggregate(grads_results)

        weights_prime = fedmeta_update_maml(
            net=self.net,
            beta=self.beta,
            weights_results=weights_results[0][0],
            gradients_aggregated=gradients_aggregated,
            weight_decay=self.weight_decay,
        )
        parameters_aggregated = ndarrays_to_parameters(weights_prime)

        # Metrics
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate loss
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )

        self.ema_loss = self.update_ema(self.ema_loss, loss_aggregated, self.smoothing_weight)
        loss_aggregated = self.ema_loss

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
            
            if self.use_ema:
                for metric, value in metrics_aggregated.items():
                    if metric not in self.ema_metrics:
                        self.ema_metrics[metric] = None

                    value = float(value) * 100

                    self.ema_metrics[metric] = self.update_ema(
                        self.ema_metrics[metric],
                        value,
                        self.smoothing_weight,
                    )

                    metrics_aggregated[metric] = round(self.ema_metrics[metric], 3)

        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        return loss_aggregated, metrics_aggregated
    

    def update_ema(self,
        prev_ema: Optional[float],
        current_value: float,
        smoothing_weight: float,
    ) -> float:
        """We use EMA to visually enhance the learning trend for each round.

        Parameters
        ----------
        prev_ema : float
            The list of metrics to aggregate.
        current_value : float
            The list of metrics to aggregate.
        smoothing_weight : float
            The list of metrics to aggregate.


        Returns
        -------
        EMA_Loss or EMA_ACC
            The weighted average metric.
        """
        if prev_ema is None:
            return current_value
        return (1 - smoothing_weight) * current_value + smoothing_weight * prev_ema
