import argparse
import numpy as np
import os
from matplotlib import pyplot as plt
import scienceplots

"""
    Code for generating simple trajectories in the spirit of GENERIC
    using the double well potential
"""

parser = argparse.ArgumentParser(prog='simulate_trajectory.py',
                                description='A short script for generating gradient dynamics data used in a machine learning project.',)

parser.add_argument("--num" , default=128, type=int, help="number of trajectories that we want to simulate")
parser.add_argument("--points", default=2048, type=int, help="number of points for each trajectory")
parser.add_argument("--dim", default=1, type=int, help="dimension of the data")
parser.add_argument("--dt", default=0.005, type=float, help="size of the time step used in the simulation")
parser.add_argument("--plot", default=True, type=bool, help="plot the results")
parser.add_argument("--gamma", default=0.3, type=float, help="the dampening constant")
parser.add_argument("--D0", default=1.2, type=float, help="the depth of the well")
parser.add_argument("--a", default=0.4, type=float, help="the distance of the dips")
args = parser.parse_args()

def acceleration(x, v, D0, a, gamma):
    return -gamma * v - 4*D0 / a**4 * x * (x**2 - a**2)

def rk4_step(x, v, dt, D0, a, gamma):
    k1_x = v
    k1_v = acceleration(x, v, D0, a, gamma)

    k2_x = v + 0.5 * dt * k1_v
    k2_v = acceleration(x + 0.5 * dt * k1_x, k2_x, D0, a, gamma)

    k3_x = v + 0.5 * dt * k2_v
    k3_v = acceleration(x + 0.5 * dt * k2_x, k3_x, D0, a, gamma)

    k4_x = v + dt * k3_v
    k4_v = acceleration(x + dt * k3_x, k4_x, D0, a, gamma)

    x_new = x + (dt / 6.0) * (k1_x + 2*k2_x + 2*k3_x + k4_x)
    v_new = v + (dt / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
    return x_new, v_new

data = []
np.random.seed(21)

for n in range(args.num):
    x = np.array([np.random.uniform(-1,1) for i in range(args.dim)])
    v = np.array([np.random.uniform(-1,1) for i in range(args.dim)])
    
    time = 0
    dataset = []
    for i in range(args.points):
        dataset.append([time] + [each for each in x] + [each for each in v])

        x,v = rk4_step(x, v, args.dt, args.D0, args.a, args.gamma)
        time += args.dt

    dataset = np.array(dataset)
    data.append(dataset)

    print(f"{n}/{args.num}", end='\r')

if os.path.exists("data"):
    os.remove("data/dataset.txt")
    for trajectory in data:
        with open("data/dataset.txt", "ab") as f:
            np.savetxt(f, trajectory, delimiter=",")
            f.write("\n".encode())
else:
    os.mkdir("data")    
    for trajectory in data:
        with open("data/dataset.txt", "ab") as f:
            np.savetxt(f, trajectory, delimiter=",")
            f.write("\n".encode())

print("\n Done! Trajectories saved into ./data/dataset.txt")

if args.plot:
    plt.style.use(['science','ieee'])
    
    if args.dim == 1:
        plt.xlabel(r"$t$")
        plt.ylabel(r"$x$")
        random_trajectory = data[np.random.randint(0,args.num-1)]
        plt.plot(random_trajectory[:,0], random_trajectory[:,1])
        plt.savefig("results/example.pdf")

        plt.show()
    
    elif args.dim == 2:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        random_trajectory = data[np.random.randint(0,args.num-1)]

        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
        ax.set_zlabel(r"$t$", labelpad=-4)
        ax.plot(random_trajectory[:,1], random_trajectory[:,2], random_trajectory[:,0], label="original data")

        plt.savefig("results/example2d.pdf")
        plt.show()
        
    else:
        raise Exception("Plotting is not supported for more dimensions than 1 and 2.")
    