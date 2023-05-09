"""
Helper classes and functions
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import logsumexp
from torch import Tensor


def make_relu_mlp(n_linear_layers, n_input_features, n_hidden_features,
                  n_output_features, final_linear_layer_bias=True):
    if n_linear_layers == 1:
        layers = \
            [nn.Linear(in_features=n_input_features,
                       out_features=n_output_features,
                       bias=final_linear_layer_bias)]
    elif n_linear_layers == 2:
        layers = \
            [nn.Linear(in_features=n_input_features,
                       out_features=n_hidden_features),
             nn.ReLU(),
             nn.Linear(in_features=n_hidden_features,
                       out_features=n_output_features,
                       bias=final_linear_layer_bias)]
    else:
        layers = \
            [nn.Linear(in_features=n_input_features,
                       out_features=n_hidden_features),
             nn.ReLU()]
        for idx in range(n_linear_layers - 2):
            layers += \
                [nn.Linear(in_features=n_hidden_features,
                           out_features=n_hidden_features),
                 nn.ReLU()]
        layers.append(
            nn.Linear(in_features=n_hidden_features,
                      out_features=n_output_features,
                      bias=final_linear_layer_bias))
    return layers


class Hypersphere(nn.Module):
    def __init__(self) -> None:
        super(Hypersphere, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        return F.normalize(input, dim=1)


def apply_network(torch_model, feature_vectors_as_np_array, batch_size, device):
    torch_model.eval()
    with torch.no_grad():
        loader = \
            torch.utils.data.DataLoader(
                list(torch.tensor(feature_vectors_as_np_array,
                                  dtype=torch.float32)),
                batch_size=batch_size,
                shuffle=False)
        outputs = []
        for batch_inputs in loader:
            batch_inputs = batch_inputs.to(device)
            batch_outputs = torch_model(batch_inputs)
            outputs.append(batch_outputs.detach().cpu().numpy())
        return np.vstack(outputs)
    torch_model.train()


def breslow_estimate(y_train, train_risk_scores):
    event_mask = (y_train[:, 1] == 1)
    event_times = y_train[:, 0][event_mask]

    sorted_unique_times = np.unique(event_times)
    baseline_hazards = np.zeros(len(sorted_unique_times))
    for idx, time in enumerate(sorted_unique_times):
        mask = (y_train[:, 0] >= time)
        baseline_hazards[idx] = \
            np.exp(np.log((event_times == time).sum())
                   - logsumexp(train_risk_scores[mask]))
    return sorted_unique_times, np.cumsum(baseline_hazards)
