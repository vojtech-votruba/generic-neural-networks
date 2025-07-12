# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import torch
from torch import nn
from torch import autograd
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from matplotlib import pyplot as plt
from tqdm import tqdm
import scienceplots


from reversible.HamiltonianNetwork import HamiltonianNewtork
from reversible.PoissionBivector import PoissonBivector
from irreversible.EntropyNetwork import EntropyNetwork
from irreversible.DissipationNetwork import DissipationNetwork


# Adding some arguments
parser = argparse.ArgumentParser(description="A pytorch code for learning and testing state space\
                                 trajectory prediction.")

parser.add_argument("--epochs", default=7000, type=int, help="number of epoches for the model to train")
parser.add_argument("--batch_size", default=128, type=int, help="batch size for training of the model")
parser.add_argument("--dt", default=0.005, type=float, help="size of the time step used in the simulation")
parser.add_argument('--train', default=True, action=argparse.BooleanOptionalAction, help="do you wish to train a new model?")
parser.add_argument("--log", default=True, action=argparse.BooleanOptionalAction, help="using log loss for plotting and such")
parser.add_argument("--lbfgs", default=True, action=argparse.BooleanOptionalAction, help="use lbfgs for optimization")

args, unknown = parser.parse_known_args()

try:
    args = parser.parse_args()
except SystemExit:
    args = parser.parse_args(args=[])

# Extracting the data
if os.path.exists("data/dataset.txt") is False:
    raise Exception("We don't have any training data. It should be stored as dataset.txt in the folder data.")

with open("data/dataset.txt", "r", encoding="utf-8") as f:
    data_raw = f.read().strip().split("\n\n")

class TrajectoryDataset(Dataset):
        def __init__(self):
                loaded_trajectories = []
                for mat_str in data_raw:
                    lines = mat_str.strip().split('\n')
                    mat = [list(map(float, line.split(','))) for line in lines]
                    loaded_trajectories.append(mat)
                loaded_trajectories = np.array(loaded_trajectories, dtype=np.float32)
                data = loaded_trajectories[:,1:-2,:]

                global DIMENSION
                DIMENSION = data.shape[2]-1

                self.z = torch.tensor(np.array(data[:,:,1:]), requires_grad=True).float() # z = (q,p) in our example
                z_dot = (data[:, 2:, 1:] - data[:, :-2, 1:]) / (2 * args.dt)

                # boundary conditions
                z_dot_first = (data[:, 1:2, 1:] - data[:, 0:1, 1:]) / args.dt
                z_dot_last = (data[:, -1:, 1:] - data[:, -2:-1, 1:]) / args.dt
                z_dot = np.concatenate([z_dot_first, z_dot, z_dot_last], axis=1)

                self.z_dot = torch.tensor(np.array(z_dot), requires_grad=True).float()
                self.n_samples = self.z.shape[0]

        def __getitem__(self, index):
            return self.z[index], self.z_dot[index] 

        def __len__(self):
            return self.n_samples

trajectories = TrajectoryDataset()

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

torch.set_default_device(DEVICE)
print(f"Using {DEVICE} for tensor calculations")

generator = torch.Generator(device=DEVICE)
generator.manual_seed(42)
    
# Reversible part
class HamiltonianDynamics(nn.Module):
    def __init__(self):
        super().__init__()
        self.H = HamiltonianNewtork(dimension=DIMENSION, generator=generator)
        self.L = PoissonBivector(dimension=DIMENSION, generator=generator)
    
    def forward(self, x):
        H = self.H(x)
        gradH = autograd.grad(H, x, grad_outputs=torch.ones_like(H), create_graph=True)[0].float()
        x_dot = torch.matmul(self.L(x), gradH.unsqueeze(-1)).squeeze(-1)

        return x_dot
    
# Irreversible part
class GradientDynamics(nn.Module):
    def __init__(self):
        super().__init__()
        self.S = EntropyNetwork(dimension=DIMENSION, generator=generator)
        self.Xi = DissipationNetwork(dimension=DIMENSION, generator=generator)
    
    def forward(self, x):
        S = self.S(x)
        x_star = autograd.grad(S, x, grad_outputs=torch.ones_like(S), create_graph=True)[0].float()
        Xi = self.Xi(x,x_star)
        x_dot = autograd.grad(Xi, x_star, grad_outputs=torch.ones_like(Xi), create_graph=True)[0]

        return x_dot

class GENERIC(nn.Module):
    def __init__(self):
        super().__init__()
        self.grad = GradientDynamics()
        self.hamil = HamiltonianDynamics()
    
    def forward(self,x):
        return self.hamil(x) + self.grad(x)

# Training the Model
L = nn.MSELoss()

if args.train:
    training_trajectories, test_trajectories = random_split(trajectories, [0.8, 0.2], generator=generator)
    model = GENERIC().to(DEVICE)

    adam_dataloader = DataLoader(dataset=training_trajectories, batch_size=args.batch_size // 2, shuffle=True, generator=generator)
    lbfgs_dataloader = DataLoader(dataset=training_trajectories, batch_size=args.batch_size, shuffle=True, generator=generator)

    adam_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, amsgrad=True)
    lbfgs_optimizer = torch.optim.LBFGS(model.parameters(), lr=1e-1, max_iter=10, history_size=20, line_search_fn='strong_wolfe')

    # Training
    losses = []

    for i in range(args.epochs):
        if i < args.epochs * 6 // 7 or not args.lbfgs:
            dataloader = adam_dataloader
            optimizer = adam_optimizer
        else:
            dataloader = lbfgs_dataloader
            optimizer = lbfgs_optimizer

        for j, (z, z_dot) in enumerate(dataloader):
            z = z.to(DEVICE)
            z_dot = z_dot.to(DEVICE)
            if i < args.epochs * 6 // 7 or not args.lbfgs:
                optimizer.zero_grad()
                predicted_z_dot = model(z)
                # TODO: Implement Jacobiator loss
                loss_jacobi = ...

                # TODO: Implement degeneracy loss
                gradH = autograd.grad(model.hamil(z), z, grad_outputs=torch.ones_like(model.hamil(z)), create_graph=True)[0].float()
                loss_degen = ((model.grad(z) + gradH).sum(dim=-1) ** 2).sum()
                loss_degen = 0

                loss = L(predicted_z_dot, z_dot) + loss_degen
                loss.backward()
                optimizer.step()

            else:
                def closure():
                    optimizer.zero_grad()
                    predicted_z_dot = model(z)
                    # TODO: Implement Jacobiator loss
                    loss_jacobi = ...

                    # TODO: Implement degeneracy loss
                    gradH = autograd.grad(model.hamil(z), z, grad_outputs=torch.ones_like(model.hamil(z)), create_graph=True)[0].float()
                    loss_degen = ((model.grad(z) + gradH).sum(dim=-1) ** 2).sum()
                    loss_degen = 0
                    
                    loss = L(predicted_z_dot, z_dot) + loss_degen
                    loss.backward()
                    return loss

                optimizer.step(closure)

            predicted_z_dot = model(z)
            loss = L(predicted_z_dot, z_dot)

        if args.log:
            losses.append(np.log(loss.item()))
        else:
            losses.append(loss.item())

        print(f"Epoch no. {i}/{args.epochs} done! Loss: {loss.item()}.", end="\r")

    if os.path.exists("models"):
        torch.save(model.state_dict(), "models/model.pth")
    else:
        os.mkdir("models")
        torch.save(model.state_dict(), "models/model.pth")
else:
    model = GENERIC().to(DEVICE)
    model.load_state_dict(torch.load("models/model.pth", weights_only=True))
    model.eval()

if args.train:
    test_z = trajectories.z[test_trajectories.indices].to(DEVICE)
    test_z_dot = trajectories.z_dot[test_trajectories.indices].to(DEVICE)
else:
    test_z = trajectories.z.to(DEVICE)
    test_z_dot = trajectories.z_dot.to(DEVICE)

MSE_loss = L(model(test_z), test_z_dot).item()
print(f"Loss on the test set is {MSE_loss}.")

plt.style.use(['science','ieee',])

# Plotting the MSE decline
if args.train:
    fig0,ax0 = plt.subplots()
    ax0.set_xlabel(r"Iterations")
    if args.log:
        ax0.set_ylabel(r"ln(MSE)")
    else:
        ax0.set_ylabel(r"MSE")
    ax0.plot(range(len(losses)), losses, label=r"training loss")
    ax0.legend()

    if DIMENSION == 2:
        fig0.savefig("results/loss1d.pdf")

# Plotting Trajectories 
if DIMENSION == 2:
    # Sampling random trajectory and plotting it along with predicted trajectory
    fig1,ax1 = plt.subplots()
    sample = test_z[np.random.randint(0,len(test_z)-1)].cpu().detach().numpy()
    tensor_sample = torch.from_numpy(sample).unsqueeze(0).clone().detach().to(DEVICE).requires_grad_(True)
    time_set = [args.dt*i for i in range(len(sample))]

    ax1.set_xlabel(r"$t$")
    ax1.set_ylabel(r"$q$")

    ax1.set_title(f"MSE on the test set: {MSE_loss:.3e}")
    ax1.plot(time_set, sample[:,0], label=r"original data")

    prediction = torch.zeros_like(tensor_sample)
    prediction[:,0,:] = tensor_sample[:,0,:]

    print("calculating sample trajectory...")
    states = [tensor_sample[:,0,:]]
    for i in tqdm(range(1, len(sample))):
        prev = states[-1].detach().clone().requires_grad_(True)
        z_dot = model(prev.unsqueeze(0))
        next_state = prev + z_dot[:,0,:] * args.dt
        states.append(next_state)
    prediction = torch.stack(states, dim=1).detach().cpu().numpy().squeeze(0)

    ax1.plot(time_set[:-2], prediction[:-2,0] , label=r"prediction")
    ax1.legend()

    fig1.savefig("results/trajectory1d.pdf")

plt.show()

# Plotting Histogram
fig_hist = plt.figure()
ax_hist = fig_hist.add_subplot()

mse_per_trajectory = torch.mean((test_z_dot - model(test_z)) ** 2, dim=(1, 2))
ax_hist.hist(mse_per_trajectory.cpu().detach().numpy(), bins=15, label="MSE")

fig_hist.legend()

if DIMENSION == 2:
    fig_hist.savefig("results/histogram1d.pdf")

# TODO: Repair the plotting of the potentials
# Plotting the potentials
if DIMENSION == 2:
    # Plotting the Hamiltonian on q
    fig2,ax2 = plt.subplots()

    # Plotting the dissipation potential
    fig3,ax3 = plt.subplots()
    ax3.set_xlabel(r"$x^*$")
    ax3.set_ylabel(r"$\Xi$")
    ax3.set_title(r"Dissipation potential $\Xi = \Xi(x=0, x^*)$")

    x_star_range = torch.linspace(-1,1,500, dtype=torch.float32).reshape(-1,1)
    zeros_column = torch.zeros_like(x_star_range, dtype=torch.float32).reshape(-1,1)

    Xi_analytical = 1/2 * x_star_range.cpu().detach().numpy()**2
    Xi_predicted = model.Xi(zeros_column, x_star_range).cpu().detach().numpy()

    scaling = np.sum(Xi_analytical * Xi_analytical) / np.sum(Xi_predicted * Xi_analytical)

    ax3.plot(x_star_range.cpu(), Xi_analytical, label=r"analytical")
    ax3.plot(x_star_range.cpu(), Xi_predicted * scaling, label=r"learned")
    
    ax3.legend()
    fig3.savefig("results/dissipation1d.pdf")

    # Plotting entropy
    fig4,ax4 = plt.subplots()
    ax4.set_xlabel(r"$x$")
    ax4.set_ylabel(r"$S$")
    x = torch.linspace(-1,1,500, dtype=torch.float32)
    x = x.view(-1, DIMENSION)
    ax4.set_title(r"Entropy $S = S(x)$")

    S_predicted = model.S(x).cpu().detach().numpy() / scaling
    S_analytical = -1/2 * x.cpu().detach().numpy()**2
    distance = np.average(S_analytical - S_predicted)

    ax4.plot(x.cpu(), S_analytical, label=r"analytical")
    ax4.plot(x.cpu(), S_predicted + distance, label=r"learned")

    ax4.legend()
    fig4.savefig("results/entropy1d.pdf")

plt.show()
