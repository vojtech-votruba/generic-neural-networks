import torch


class PositiveLinear(torch.nn.Linear):
    """
        A positive layer that we use to enforce convexity and concavity
    """

    def __init__(self, *args, epsilon=5.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon

    def forward(self, input):
        W = self.weight
        eps_tensor = torch.tensor(self.epsilon, device=W.device, dtype=W.dtype)

        positive_W = W + torch.exp(-eps_tensor)
        negative_W = torch.exp(W - eps_tensor) 
        reparam_W = torch.where(W >= 0, positive_W, negative_W) 

        return torch.nn.functional.linear(input, reparam_W, self.bias)