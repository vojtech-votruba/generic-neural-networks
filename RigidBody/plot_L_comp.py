import torch
import pandas as pd

import matplotlib.pyplot as plt


points = pd.read_csv("RBTEST/data/dataset.xyz")
multiplier = 1000 # We will be dividing by total J. Lets have errors on the order of maginutde around 1

def compat_error(J_exact, J, m):
    grad_J_0, = torch.autograd.grad(J[:,0].sum(), m, retain_graph=True, create_graph=False) # DJ0/Dm
    grad_J_1, = torch.autograd.grad(J[:,1].sum(), m, retain_graph=True, create_graph=False) # DJ1/Dm
    grad_J_2, = torch.autograd.grad(J[:,2].sum(), m, retain_graph=True, create_graph=False) # DJ2/Dm
    grad_J = torch.stack((grad_J_0, grad_J_1, grad_J_2), axis=2) # Aij DJi/Dmj batch dimension in front

    rot_tuple = (grad_J_2[:,1] - grad_J_1[:,2], grad_J_0[:,2] - grad_J_2[:,0], grad_J_1[:,0] - grad_J_0[:,1])
    rot_J = torch.stack(rot_tuple, axis=1)

    return torch.sum(J_exact*rot_J, axis=1)**2



mx = torch.tensor(points["mx"], dtype=torch.float32)
my = torch.tensor(points["my"], dtype=torch.float32)
mz = torch.tensor(points["mz"], dtype=torch.float32)
m = torch.stack((mx, my, mz), axis=1)
m.requires_grad=True
total_m_batched = torch.split(m, 20)

J_exact = torch.stack((mx, my, mz), axis=1)


### Without error

L_tensor = torch.load("RBTEST/saved_models/without_jacobi_L")


L_batched = [L_tensor(b) for b in total_m_batched]
L = torch.cat(L_batched, axis=0)

J = torch.stack((-L[:, 1,2], L[:, 0, 2], -L[:, 0,1]), axis=1)

#Normalize
#J = J/J.sum().detach()*multiplier

error_array_without = compat_error(J_exact, J, m)

print("Total error without", error_array_without.sum())


### Soft error

L_tensor = torch.load("RBTEST/saved_models/soft_jacobi_L")

L_batched = [L_tensor(b) for b in total_m_batched]
L = torch.cat(L_batched, axis=0)

J = torch.stack((-L[:, 1,2], L[:, 0, 2], -L[:, 0,1]), axis=1)
#J = J/J.sum().detach()*multiplier

error_array_soft = compat_error(J_exact, J, m)

print("Total error soft", error_array_soft.sum())

### Implicit error

J_vector = torch.load("RBTEST/saved_models/implicit_jacobi_J")

J_batched = [J_vector(b)[0] for b in total_m_batched]

J = torch.cat(J_batched, axis=0)
#J = J/J.sum().detach()*multiplier

error_array_implicit = compat_error(J_exact, J, m)

print("Total error implicit", error_array_implicit.sum())

plt.plot(error_array_without, label="without")
plt.plot(error_array_soft, label="soft")
plt.plot(error_array_implicit, label="implicit")
plt.xlabel("Points (from subsequent trajectories)")
plt.ylabel("E(compat)")
plt.legend()
plt.show()


    
    
