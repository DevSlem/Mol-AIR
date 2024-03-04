from abc import ABC

import torch
import torch.nn as nn

# Reference: https://github.com/thomashirtz/noisy-networks/blob/main/noisynetworks.py

class AbstractNoisyLayer(nn.Module, ABC):
    def __init__(
            self,
            input_features: int,
            output_features: int,
            sigma: float,
    ):
        super().__init__()

        self.sigma = sigma
        self.input_features = input_features
        self.output_features = output_features

        self.mu_bias = nn.Parameter(torch.FloatTensor(output_features))
        self.sigma_bias = nn.Parameter(torch.FloatTensor(output_features))
        self.mu_weight = nn.Parameter(torch.FloatTensor(output_features, input_features))
        self.sigma_weight = nn.Parameter(torch.FloatTensor(output_features, input_features))

        self.register_buffer('epsilon_input', torch.FloatTensor(input_features))
        self.register_buffer('epsilon_output', torch.FloatTensor(output_features))

    def forward(
            self,
            x: torch.Tensor,
            sample_noise: bool = True
    ) -> torch.Tensor:
        if not self.training:
            return nn.functional.linear(x, weight=self.mu_weight, bias=self.mu_bias)

        if sample_noise:
            self.sample_noise()

        return nn.functional.linear(x, weight=self.weight, bias=self.bias)

    @property
    def weight(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def bias(self) -> torch.Tensor:
        raise NotImplementedError

    def sample_noise(self) -> None:
        raise NotImplementedError

    def parameter_initialization(self) -> None:
        raise NotImplementedError

    def get_noise_tensor(self, features: int) -> torch.Tensor:
        noise = torch.FloatTensor(features).uniform_(-self.bound, self.bound).to(self.mu_bias.device)
        return torch.sign(noise) * torch.sqrt(torch.abs(noise))


class IndependentNoisyLayer(AbstractNoisyLayer):
    def __init__(
            self,
            input_features: int,
            output_features: int,
            sigma: float = 0.017,
    ):
        super(AbstractNoisyLayer).__init__(
            input_features=input_features,
            output_features=output_features,
            sigma=sigma
        )

        self.bound = (3 / input_features) ** 0.5
        self.parameter_initialization()
        self.sample_noise()

    @property
    def weight(self) -> torch.Tensor:
        return self.sigma_weight * self.epsilon_weight + self.mu_weight

    @property
    def bias(self) -> torch.Tensor:
        return self.sigma_bias * self.epsilon_bias + self.mu_bias

    def sample_noise(self) -> None:
        self.epsilon_bias = self.get_noise_tensor((self.output_features,))
        self.epsilon_weight = self.get_noise_tensor((self.output_features, self.input_features))

    def parameter_initialization(self) -> None:
        self.sigma_bias.data.fill_(self.sigma)
        self.sigma_weight.data.fill_(self.sigma)
        self.mu_bias.data.uniform_(-self.bound, self.bound)
        self.mu_weight.data.uniform_(-self.bound, self.bound)


class FactorisedNoisyLayer(AbstractNoisyLayer):
    def __init__(
            self,
            input_features: int,
            output_features: int,
            sigma: float = 0.5,
    ):
        super().__init__(
            input_features=input_features,
            output_features=output_features,
            sigma=sigma
        )

        self.bound = input_features**(-0.5)
        self.parameter_initialization()
        self.sample_noise()

    @property
    def weight(self) -> torch.Tensor:
        return self.sigma_weight * torch.ger(self.epsilon_output, self.epsilon_input) + self.mu_weight

    @property
    def bias(self) -> torch.Tensor:
        return self.sigma_bias * self.epsilon_output + self.mu_bias

    def sample_noise(self) -> None:
        self.epsilon_input = self.get_noise_tensor(self.input_features)
        self.epsilon_output = self.get_noise_tensor(self.output_features)

    def parameter_initialization(self) -> None:
        self.mu_bias.data.uniform_(-self.bound, self.bound)
        self.sigma_bias.data.fill_(self.sigma * self.bound)
        self.mu_weight.data.uniform_(-self.bound, self.bound)
        self.sigma_weight.data.fill_(self.sigma * self.bound)