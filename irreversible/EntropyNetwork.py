from torch import nn


from .PositiveLinear import PositiveLinear


class EntropyNetwork(nn.Module):
    """
        For the entropy network we are using a fully input concave neural network achitecture,
        it's a simple alteration of FICNN - fully input convex neural nets
    """
    def __init__(self, dimension, generator):
        super().__init__()
        self.dimension = dimension
        self.generator = generator

        self.input_layer = nn.Linear(dimension, 8)

        self.prop_layer1 = PositiveLinear(8, 8)
        self.lateral_layer1 = nn.Linear(dimension, 8)

        self.prop_layer2 = PositiveLinear(8, 8)
        self.lateral_layer2 = nn.Linear(dimension, 8)

        self.prop_layer3 = PositiveLinear(8, 8)
        self.lateral_layer3 = nn.Linear(dimension, 8)

        self.prop_layer4 = PositiveLinear(8, 8)
        self.lateral_layer4 = nn.Linear(dimension, 8)

        self.output_layer = PositiveLinear(8, 1,)
        self.lateral_layer_out = nn.Linear(dimension, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, generator=self.generator)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x0):
        x = nn.Softplus()(self.input_layer(x0))

        x = nn.Softplus()(self.prop_layer1(x) + self.lateral_layer1(x0))

        x = nn.Softplus()(self.prop_layer2(x) + self.lateral_layer2(x0))

        x = nn.Softplus()(self.prop_layer3(x) + self.lateral_layer3(x0))

        x = nn.Softplus()(self.prop_layer4(x) + self.lateral_layer4(x0))

        S_out = nn.Softplus()(self.output_layer(x) + self.lateral_layer_out(x0))

        return -S_out