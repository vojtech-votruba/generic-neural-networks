import torch
import matplotlib.pyplot as plt
from matplotlib import cm

import argparse


energy = torch.load("RBTEST/saved_models/soft_jacobi_energy")
L_tensor = torch.load("RBTEST/saved_models/soft_jacobi_L")

def generate_E_points(args):
    #generates random initial conditions (uniformly on the ball with radius as in the original args)
    sqm = torch.tensor(args.init_mx**2 + args.init_my**2 + args.init_mz**2) #square magnitude of m
    mmag = torch.sqrt(sqm) #magnitude of m
    
    
    
    mx = torch.linspace(0, mmag, args.density)
    my = torch.linspace(0, mmag, args.density)
    mx_mesh, my_mesh = torch.meshgrid((mx, my)) 
    mxsq_mesh, mysq_mesh = mx_mesh**2, my_mesh**2
    mzsq_mesh = sqm - mxsq_mesh - mysq_mesh
    mz_mesh = torch.sqrt(mzsq_mesh)
    
    total_m = torch.stack((mx_mesh.reshape(-1,), my_mesh.reshape(-1,), mz_mesh.reshape(-1,)), axis=1)
    
    
    E = energy(total_m)
    mx_for_plot = mx_mesh.reshape((args.density, args.density))
    my_for_plot = my_mesh.reshape((args.density, args.density))
    mz_for_plot = mz_mesh.reshape((args.density, args.density))
    E_for_plot = E.detach().reshape((args.density, args.density))
    
    return mx_for_plot, my_for_plot, mz_for_plot, E_for_plot
    
    
def generate_L_points(args):
    #generates random initial conditions (uniformly on the ball with radius as in the original args)
    sqm = torch.tensor(args.init_mx**2 + args.init_my**2 + args.init_mz**2) #square magnitude of m
    mmag = torch.sqrt(sqm) #magnitude of m
    
    
    
    mx = torch.linspace(0, mmag, args.density)
    my = torch.linspace(0, mmag, args.density)
    mx_mesh, my_mesh = torch.meshgrid((mx, my)) 
    mxsq_mesh, mysq_mesh = mx_mesh**2, my_mesh**2
    mzsq_mesh = sqm - mxsq_mesh - mysq_mesh
    mz_mesh = torch.sqrt(mzsq_mesh)
    
    mx_mesh.requires_grad=True
    my_mesh.requires_grad=True
    mz_mesh.requires_grad=True
    
    total_m = torch.stack((mx_mesh.reshape(-1,), my_mesh.reshape(-1,), mz_mesh.reshape(-1,)), axis=1)
    
    total_m_batched = torch.split(total_m, 20)

    L_batched = [L_tensor(b) for b in total_m_batched]
    
    L = torch.cat(L_batched, axis=0)
    return mx_mesh, my_mesh, mz_mesh, L

#arguments used to get plot density and preset |m|**2
parser = argparse.ArgumentParser()
parser.add_argument("--init_mx", default=10.0, type=float, help="A value of momentum, x component")
parser.add_argument("--init_my", default=3.0, type=float, help="A value of momentum, y component") 
parser.add_argument("--init_mz", default=4.0, type=float, help="A value momentum, z component")
parser.add_argument("--density", default=60, type=int, help="Plot density in each dimension (total No. points is density**2)")

args = parser.parse_args([] if "__file__" not in globals() else None)

mx, my, mz, E = generate_E_points(args)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

surf = ax.plot_surface(mx, my, E, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_xlabel("mx")
ax.set_ylabel("my")
ax.set_zlabel("Energy")
plt.show() 


mx, my, mz, L = generate_L_points(args)

dL12dmx, = torch.autograd.grad(L[:,1,2].sum(), mx, retain_graph=True, create_graph=False)
dL12dmy, = torch.autograd.grad(L[:,1,2].sum(), my, retain_graph=True, create_graph=False)
dL12dmz, = torch.autograd.grad(L[:,1,2].sum(), mz, retain_graph=True, create_graph=False)
dL02dmx, = torch.autograd.grad(L[:,0,2].sum(), mx, retain_graph=True, create_graph=False)
dL02dmy, = torch.autograd.grad(L[:,0,2].sum(), my, retain_graph=True, create_graph=False)
dL02dmz, = torch.autograd.grad(L[:,0,2].sum(), mz, retain_graph=True, create_graph=False)
dL01dmx, = torch.autograd.grad(L[:,0,1].sum(), mx, retain_graph=True, create_graph=False)
dL01dmy, = torch.autograd.grad(L[:,0,1].sum(), my, retain_graph=True, create_graph=False)
dL01dmz, = torch.autograd.grad(L[:,0,1].sum(), mz, retain_graph=True, create_graph=False)

plt.scatter(mx.detach().reshape(-1,), dL12dmx.detach().reshape(-1), label="DL23/Dmx")
plt.scatter(my.detach().reshape(-1,), dL02dmy.detach().reshape(-1,), label="DL13/Dmy")
plt.scatter(mz.detach().reshape(-1,), dL01dmz.detach().reshape(-1,), label="DL12/Dmz")
plt.legend()
plt.show()


plt.scatter(mx.detach().reshape(-1,), dL12dmx.detach().reshape(-1), label="DL23/Dmx")
plt.scatter(my.detach().reshape(-1,), dL12dmy.detach().reshape(-1,), label="DL23/Dmy")
plt.scatter(mz.detach().reshape(-1,), dL12dmz.detach().reshape(-1,), label="DL23/Dmz")
plt.legend()
plt.show()

plt.scatter(mx.detach().reshape(-1,), dL02dmx.detach().reshape(-1), label="DL13/Dmx")
plt.scatter(my.detach().reshape(-1,), dL02dmy.detach().reshape(-1,), label="DL13/Dmy")
plt.scatter(mz.detach().reshape(-1,), dL02dmz.detach().reshape(-1,), label="DL13/Dmz")
plt.legend()
plt.show()

plt.scatter(mx.detach().reshape(-1,), dL01dmx.detach().reshape(-1), label="DL12/Dmx")
plt.scatter(my.detach().reshape(-1,), dL01dmy.detach().reshape(-1,), label="DL12/Dmy")
plt.scatter(mz.detach().reshape(-1,), dL01dmz.detach().reshape(-1,), label="DL12/Dmz")
plt.legend()
plt.show()
