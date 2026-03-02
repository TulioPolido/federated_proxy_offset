from __future__ import annotations

from collections import OrderedDict
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import flwr as flwr

from logging import INFO, ERROR
from flwr.common.logger import log
from flwr.common import EvaluateRes, NDArrays, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg

def aggregate_grads_by_name(grads_results):
    # grads_results: List[Tuple[dict[str, np.ndarray], int]]
    total = sum(n for _, n in grads_results)
    out = {}
    for grads_dict, n in grads_results:
        w = n / total
        for k, g in grads_dict.items():
            if k not in out:
                out[k] = g * w
            else:
                out[k] += g * w
    return out



def server_meta_update_adam(
    net: torch.nn.Module,
    beta: float,
    base_weights,
    gradients_aggregated: dict[str, "np.ndarray"],
    weight_decay: float = 0.0,
):
    params_dict = zip(net.state_dict().keys(), base_weights)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

    opt = torch.optim.Adam(net.parameters(), lr=float(beta), weight_decay=float(weight_decay))

    for name, param in net.named_parameters():
        g = gradients_aggregated.get(name, None)
        if g is None:
            param.grad = None
        else:
            param.grad = torch.tensor(g, dtype=param.dtype, device=param.device)

    opt.step()
    opt.zero_grad()

    return [v.detach().cpu().numpy() for _, v in net.state_dict().items()]



class HFMetaMetricStrategy(flwr.server.strategy.FedAvg):
    def __init__(self, model: torch.nn.Module, lr_outer: float = 1e-4, **kwargs):
        super().__init__(**kwargs)
        self.net = model
        self.beta = float(lr_outer)

    def aggregate_fit(self, server_round, results, failures):
        for failure in failures:
            log(ERROR, f"Fit Failure: {failure}")

        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        # weights returned by clients (base global) - use first
        weights_results: List[Tuple[NDArrays, int]] = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        grads_results: List[Tuple[NDArrays, int]] = []
        for _, fit_res in results:
            grads_file = fit_res.metrics.get("grads_file")
            if grads_file is None:
                raise ValueError("No grads_file in metrics for a client!")

            file_path = f"gradients_{grads_file}.npz"
            log(INFO, f"Reading gradients file: {file_path}")
            with np.load(file_path) as data:
                grads_dict = {k: data[k] for k in data.files}

            grads_results.append((grads_dict, fit_res.num_examples))

        gradients_aggregated = aggregate_grads_by_name(grads_results)


        updated = server_meta_update_adam(
            net=self.net,
            beta=self.beta,
            base_weights=weights_results[0][0],
            gradients_aggregated=gradients_aggregated,
            weight_decay=1e-4,
        )

        return ndarrays_to_parameters(updated), {}

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[Union[tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> tuple[Optional[float], dict[str, Scalar]]:
        for failure in failures:
            log(ERROR, f"Evaluate Failure: {failure}")
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        loss_agg = weighted_loss_avg([(r.num_examples, r.loss) for _, r in results])
        return loss_agg, {}

