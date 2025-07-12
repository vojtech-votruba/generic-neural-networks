from torch import nn


class HamiltonianNewtork(nn.Module):
    """
        Standard feeedforward neural network that calculates the Hamiltonian function
    """
    def __init__(self, dimension, generator):
        super().__init__()
        self.dimension = dimension
        self.generator = generator

        self.input_layer = nn.Linear(self.dimension, 8)

        self.prop_layer1 = nn.Linear(8, 8)
        self.prop_layer2 = nn.Linear(8, 8)
        self.prop_layer3 = nn.Linear(8, 8)
        self.prop_layer4 = nn.Linear(8, 8)

        self.output_layer = nn.Linear(8, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, generator=self.generator)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x0):
        x = nn.Softplus()(self.input_layer(x0))

        x = nn.Softplus()(self.prop_layer1(x))
        x = nn.Softplus()(self.prop_layer2(x))
        x = nn.Softplus()(self.prop_layer3(x))
        x = nn.Softplus()(self.prop_layer4(x))

        H = nn.Softplus()(self.output_layer(x))
        return H
    