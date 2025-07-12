from torch import nn, zeros


class PoissonBivector(nn.Module):
    def __init__(self, dimension, generator):
        super().__init__()
        self.dimensions = dimension
        self.generator = generator

        self.input_layer = nn.Linear(dimension, 8)

        self.prop_layer1 = nn.Linear(8, 8)
        self.prop_layer2 = nn.Linear(8, 8)
        self.prop_layer3 = nn.Linear(8, 8)
        self.prop_layer4 = nn.Linear(8, 8)

        self.output_layer = nn.Linear(8, dimension*(dimension-1)//2)

        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, generator=self.generator)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        batch_size, traj_len, dim = x.shape 

        # Flatten batch and trajectory for linear layers
        x_flat = x.view(batch_size * traj_len, dim)

        y = nn.Softplus()(self.input_layer(x_flat))
        y = nn.Softplus()(self.prop_layer1(y))
        y = nn.Softplus()(self.prop_layer2(y))
        y = nn.Softplus()(self.prop_layer3(y))
        y = nn.Softplus()(self.prop_layer4(y))

        upper_triangle = self.output_layer(y)  # shape [batch_size*traj_len, dim*(dim-1)//2]
        L = zeros(batch_size * traj_len, dim, dim, device=x.device)
        idx = 0
        for i in range(dim):
            for j in range(i + 1, dim):
                L[:, i, j] = upper_triangle[:, idx]
                L[:, j, i] = -upper_triangle[:, idx]
                idx += 1

        L = L.view(batch_size, traj_len, dim, dim)
        return L