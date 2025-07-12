from torch import nn
from torch import autograd
from torch import mul, zeros_like, ones_like


from .PositiveLinear import PositiveLinear


class DissipationNetwork(nn.Module):
    """
        For this network we are using a more complex architecture to ensure 
        only a partial convexity of the output with respect to some inputs.
        Specifically: PICNN, source: https://arxiv.org/pdf/1609.07152
    """
    def __init__(self, dimension, generator):
        super().__init__()
        self.dimension = dimension
        self.generator = generator

        # The branch that propagates x directly forward
        self.x_input_layer = nn.Linear(dimension, 8)
        self.x_prop_layer1 = nn.Linear(8, 8)

        # The branch that goes directly between x and x_star
        self.x_lateral_layer_1 = nn.Linear(dimension, 8)
        self.x_lateral_layer_2 = nn.Linear(8, 8)
        self.x_lateral_layer_out = nn.Linear(8, 1)

        # The branch that propagates x_star forward (We need to enforce convexity here)
        self.conjugate_prop_layer_1 = PositiveLinear(8, 8, bias=False)
        self.conjugate_prop_layer_out= PositiveLinear(8, 1, bias=False)

        self.conjugate_prop_layer_1_mid = nn.Linear(8, 8)
        self.conjugate_prop_layer_out_mid = nn.Linear(8, 8)

        # The branch which always starts at x0_star and ends at arbitrary x_star
        self.conjugate_lateral_layer_in = nn.Linear(dimension, 8, bias=False)
        self.conjugate_lateral_layer_1 = nn.Linear(dimension, 8, bias=False)
        self.conjugate_lateral_layer_out = nn.Linear(dimension, 1, bias=False)

        self.conjugate_lateral_layer_in_mid = nn.Linear(dimension, dimension)
        self.conjugate_lateral_layer_1_mid = nn.Linear(8, dimension)
        self.conjugate_lateral_layer_out_mid = nn.Linear(8, dimension)

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, generator=self.generator)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward_raw(self, state, state_conjugate):
        x0 = state
        x0_star = state_conjugate

        x_star = nn.Softplus()(self.x_lateral_layer_1(x0) 
                               + self.conjugate_lateral_layer_in(mul(x0_star, self.conjugate_lateral_layer_in_mid(x0))))
        x = nn.Softplus()(self.x_input_layer(x0))

        x_star = nn.Softplus()(self.x_lateral_layer_2(x) 
                               + self.conjugate_prop_layer_1(mul(x_star, nn.Softplus()(self.conjugate_prop_layer_1_mid(x))))
                                + self.conjugate_lateral_layer_1(mul(x0_star, self.conjugate_lateral_layer_1_mid(x))))
        x = nn.Softplus()(self.x_prop_layer1(x))

        Xi_out = nn.Softplus()(self.x_lateral_layer_out(x) 
                            + self.conjugate_prop_layer_out(mul(x_star, nn.Softplus()(self.conjugate_prop_layer_out_mid(x))))\
                                + self.conjugate_lateral_layer_out(mul(x0_star, self.conjugate_lateral_layer_out_mid(x))))

        return Xi_out
    
    def forward(self, state, state_conjugate):
        x_star_zeros = zeros_like(state, requires_grad=True)
        Xi_raw = self.forward_raw(state, state_conjugate)
        Xi_at_zero = self.forward_raw(state, x_star_zeros)
        Xi = Xi_raw - Xi_at_zero - (state_conjugate * autograd.grad(Xi_at_zero, x_star_zeros,
                                                                     grad_outputs=ones_like(Xi_at_zero), create_graph=True)[0]).sum(dim=-1).unsqueeze(-1)
        return Xi