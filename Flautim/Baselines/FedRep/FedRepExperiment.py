import torch
import flautim as fl
from collections import OrderedDict

class RecurrenceExperiment(fl.Experiment):
    def __init__(self, context, model_manager, trainloader, valloader):
        super().__init__(context=context,
                         model_manager=model_manager,
                         trainloader=trainloader,
                         valloader=valloader)

    def training_loop(self, device: torch.device):
        """Override this instead of fit."""
        self.model_manager.model.enable_body()
        self.model_manager.model.enable_head()
        return self.model_manager.train(device)

    def validation_loop(self, device: torch.device):
        """Override this instead of evaluate."""
        test_results = self.model_manager.test(device)
        return {
            "loss": test_results.get("loss", 0.0),
            "accuracy": test_results.get("accuracy", 0.0),
            "num_examples": self.model_manager.test_dataset_size(),
        }

    def get_parameters(self, config):
        return [
            val.cpu().numpy()
            for val in self.model_manager.model.body.state_dict().values()
        ]

    def set_parameters(self, parameters, evaluate: bool = False):
        model_keys = [
            k
            for k in self.model_manager.model.state_dict().keys()
            if k.startswith("_body")
        ]

        if not evaluate:
            model_keys.extend(
                [k for k in self.model_manager.model.state_dict().keys() if k.startswith("_head")]
            )

        state_dict = OrderedDict(
            (k, torch.from_numpy(v)) for k, v in zip(model_keys, parameters)
        )
        self.model_manager.model.set_parameters(state_dict)
