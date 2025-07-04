from re import I
import os, sys
import argparse
import simulate
import math
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
from math import sqrt
from learn import *
import multiprocessing as mp


def norm(x, y, z):
    return sqrt(x**2+y**2+z**2)

def update_args_init(args):
    #generates random initial conditions (uniformly on the ball with radius as in the original args)
    sqm = args.init_mx**2 + args.init_my**2 + args.init_mz**2 #square magnitude of m
    m = math.sqrt(sqm) #magnitude of m
    sqr = args.init_rx**2 + args.init_ry**2 + args.init_rz**2 #square magnitude of m
    r = math.sqrt(sqr) #magnitude of m

    n = 2*m #more than m
    while n >= m:
        mx = m*2*(np.random.rand()-0.5)
        my = m*2*(np.random.rand()-0.5)
        mz = m*2*(np.random.rand()-0.5)
        n = norm(mx, my, mz)

    n = 2*r #more than r
    while n >= r:
        rx = r*2*(np.random.rand()-0.5)
        ry = r*2*(np.random.rand()-0.5)
        rz = r*2*(np.random.rand()-0.5)
        n = norm(rx, ry, rz)

    result = argparse.Namespace(**vars(args)) #copy args
    result.init_mx = mx
    result.init_my = my
    result.init_mz = mz
    result.init_rx = rx
    result.init_ry = ry
    result.init_rz = rz

    return result

def simulate_normal(args):
    return simulate.simulate(args, method = "normal") 
def simulate_implicit(args):
    return simulate.simulate(args, method = "implicit") 
def simulate_without(args):
    return simulate.simulate(args, method = "without") 
def simulate_soft(args):
    return simulate.simulate(args, method = "soft") 

def generate_trajectories(args):
    if args.generate:
        print("Generating dataset.")
        #Now we generage initial conditions (deterministic)
        np.random.seed(args.seed)
        argss = []
        for i in range(args.sampling):
            argss.append(update_args_init(args))
        pool = mp.Pool(mp.cpu_count())
        dfs =  pool.map(simulate_normal, argss)
        pool.close()
        #save
        print("Saving dataset")
        total_data_frame = dfs[0].copy()
        for i in range(1, len(dfs)):
            total_data_frame = pd.concat([total_data_frame, dfs[i]])
        #save to file
        simulate.save_simulation(total_data_frame, args.folder_name+"/"+DEFAULT_dataset)
        print("Generated trajectories saved to: ", args.folder_name+"/"+DEFAULT_dataset)
    else: #simulating with the learned models
        total_implicit_data_frame = None
        total_soft_data_frame = None
        total_without_data_frame = None
        total_generalization_data_frame = None
        np.random.seed(args.seed+100*args.sampling) #new seed, safely beyond the last used value

        print("-------------------------------")
        print("Simulating with learned models.")    
        print("-------------------------------")

        #generating initial conditions
        argss = []
        for i in range(args.points):
            argss.append(update_args_init(args))

        #GT
        print("Generating GT.")
        pool = mp.Pool(mp.cpu_count())
        dfs =  pool.map(simulate_normal, argss)
        pool.close()
        total_generalization_data_frame = dfs[0].copy()
        for i in range(1, len(dfs)):
            total_generalization_data_frame = pd.concat([total_generalization_data_frame, dfs[i]])

        if args.implicit:
            print("Simulating with learned implicit.")
            #pool = mp.Pool(mp.cpu_count())
            ctx = mp.get_context('spawn')
            with ctx.Pool(3) as pool:
                dfs =  pool.map(simulate_implicit, argss)
            #dfs =  list(map(simulate_implicit, argss))
            #pool.close()
            total_implicit_data_frame = dfs[0].copy()
            for i in range(1, len(dfs)):
                total_implicit_data_frame = pd.concat([total_implicit_data_frame, dfs[i]])

        if args.soft:
            print("Simulating with learned soft.")
            #pool = mp.Pool(mp.cpu_count())
            ctx = mp.get_context('spawn')
            with ctx.Pool(3) as pool:
                dfs =  pool.map(simulate_soft, argss)
            #dfs =  list(map(simulate_soft, argss))
            #pool.close()
            total_soft_data_frame = dfs[0].copy()
            for i in range(1, len(dfs)):
                total_soft_data_frame = pd.concat([total_soft_data_frame, dfs[i]])

        if args.without:
            print("Simulating with learned without.")
            #pool = mp.Pool(mp.cpu_count())
            ctx = mp.get_context('spawn')
            with ctx.Pool(3) as pool:
                dfs =  pool.map(simulate_without, argss)
            #dfs =  list(map(simulate_without, argss))
            #pool.close()
            total_without_data_frame = dfs[0].copy()
            for i in range(1, len(dfs)):
                total_without_data_frame = pd.concat([total_without_data_frame, dfs[i]])

        #    %if (args.points >= 10) and ((i % int(round(args.points/10))) == 0):
        #        print((i+0.0)/args.points*100, "%")

        #save to file
        if args.implicit:
            simulate.save_simulation(total_implicit_data_frame, args.folder_name+"/data/learned_implicit.xyz") 
        if args.soft:
            simulate.save_simulation(total_soft_data_frame, args.folder_name+"/data/learned_soft.xyz") 
        if args.without:
            simulate.save_simulation(total_without_data_frame, args.folder_name+"/data/learned_without.xyz") 
        simulate.save_simulation(total_generalization_data_frame, args.folder_name+"/data/generalization.xyz") 

def add_plot(ax, x=None,y=None, name=""):
    #ax.scatter(x[::args.plot_every],y[::args.plot_every])
    if x is not None:
        ax.plot(x, y, lw=0.7, label=name)
    else:
        ax.plot(y, lw=0.7, label=name)    

def plot_training_errors(args):
    print("***If Runtime tkinter errors are raised, it is because some matplotlib vs threads problems. Shouldn't be serious.***")

    name = args.folder_name
    if args.soft:
        df_soft_errors = pd.read_csv(name+"/data/errors_soft.csv")
        train_mov_errors = df_soft_errors["train_mov"]
        validation_mov_errors = df_soft_errors["val_mov"]
        add_plot(plt, y=train_mov_errors[1:], name="soft train move")
        add_plot(plt, y=validation_mov_errors[1:], name="soft val move")
        plt.legend()
        plt.show()

        validation_reg_errors = df_soft_errors["val_reg"]
        train_reg_errors = df_soft_errors["train_reg"]
        add_plot(plt, y=train_reg_errors[1:], name="soft train Jacobi")
        add_plot(plt, y=validation_reg_errors[1:], name="soft val Jacobi")
        plt.legend()
        plt.show()

    if args.implicit:
        df_implicit_errors = pd.read_csv(name+"/data/errors_implicit.csv")
        train_mov_errors = df_implicit_errors["train_mov"]
        validation_mov_errors = df_implicit_errors["val_mov"]
        add_plot(plt, y=train_mov_errors[1:], name="impclicit train move")
        add_plot(plt, y=validation_mov_errors[1:], name="implicit val move")
        plt.legend()
        plt.show()

    if args.without:
        df_without_errors = pd.read_csv(name+"/data/errors_without.csv")
        train_mov_errors = df_without_errors["train_mov"]
        validation_mov_errors = df_without_errors["val_mov"]
        add_plot(plt, y=train_mov_errors[1:], name="without train move")
        add_plot(plt, y=validation_mov_errors[1:], name="without val move")
        plt.legend()
        plt.show()

def resolve_automatic_dt(args):
    if args.model == "RB": #rigid body
        omega = sqrt(max([args.init_mx/args.Ix, args.init_mx/args.Ix, args.init_mz/args.Iz]))
    elif args.model == "HT": #heavy top
        omega1 = sqrt(max([args.init_mx/args.Ix, args.init_mx/args.Ix, args.init_mz/args.Iz]))
        omega2 = sqrt(args.Mgl*args.init_rz)*sqrt(max([1.0/args.Ix, 1.0/args.Iy, 1.0/args.Iz]))
        omega = max(omega1, omega2)
    elif args.model == "P3D": #3D Harmonic oscillator
        omega = sqrt(args.alpha/args.M) 
    elif args.model == "P2D": #3D Harmonic oscillator
        omega = sqrt(args.alpha/args.M) 
    elif args.model == "K3D": #3D Kepler problem
        r = math.sqrt(args.init_rx**2+args.init_ry**2+args.init_rz**2)
        p = math.sqrt(args.init_mx**2+args.init_my**2+args.init_mz**2)
        m = r*p
        e = args.M*args.alpha**2/(2*m**2)
        omega = 2/(args.alpha*math.sqrt(args.M/(2*e**3)))
    elif args.model == "Sh": #Shivamoggi
        omega = 2*math.pi
    else:
        raise Exception("Unkonown model.")
    dt = 0.01 * 2*math.pi/omega
    print("Setting dt = ", dt)
    return dt
        

if __name__ == "__main__":
    #Parse arguments
    #Typical usage: python3 comparison.py --generate --steps=100 --implicit --soft --without
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheme", default="IMR", type=str, help="Numerical scheme. FE forward euler, BE backward euler, CN Crank-Nicholson")
    parser.add_argument("--model", default="RB", type=str, help="Model: RB, HT, P3D, K3D, or P2D.")
    parser.add_argument("--steps", default=200, type=int, help="Number of simulation steps")
    parser.add_argument("--init_mx", default=10.0, type=float, help="A value of momentum, x component")
    parser.add_argument("--init_my", default=3.0, type=float, help="A value of momentum, y component") 
    parser.add_argument("--init_mz", default=4.0, type=float, help="A value momentum, z component")
    parser.add_argument("--Mgl", default=9.81*0.1, type=float, help="M*g*l")
    parser.add_argument("--init_rx", default=1.0, type=float, help="Initial r, x component")
    parser.add_argument("--init_ry", default=-3.0, type=float, help="Initial r, y component")
    parser.add_argument("--init_rz", default=10.0, type=float, help="Initial r, z component")
    parser.add_argument("--Ix", default=10.0, type=float, help="Ix")
    parser.add_argument("--Iy", default=20.0, type=float, help="Iy")
    parser.add_argument("--Iz", default=40.0, type=float, help="Iz")
    parser.add_argument("--dt", default=0.0, type=float, help="Timestep, 0.0 for automatic")
    parser.add_argument("--alpha", default=2.0, type=float, help="Potential magnitude or tau prefactor.")
    parser.add_argument("--implicit", default=False, action="store_true", help="Use implicit Jacobi.")
    parser.add_argument("--soft", default=False, action="store_true", help="Use soft Jacobi.")
    parser.add_argument("--without", default=False, action="store_true", help="Use no Jacobi.")
    parser.add_argument("--normalise", default=False, action="store_true", help="Normalise energy matrix at the end")
    parser.add_argument("--generate", default=False, action="store_true", help="Generate new trajectories.")
    parser.add_argument("--verbose", default=False, action="store_true", help="Print a lot of useful output")
    parser.add_argument("--no_show", default=False, action="store_true", help="Don't show the training errors.")
    parser.add_argument("--sampling", default=100, type=int, help="Approximate number of points to be sampled on the sphere.")
    parser.add_argument("--points", default=100, type=int, help="Number of points on the sphere for generalization.")
    parser.add_argument("--prefactor", default=1.0, type=float, help="Loss prefactor")
    parser.add_argument("--jac_prefactor", default=1.0, type=float, help="Loss prefactor for Jacobi identity")
    parser.add_argument("--epochs", default=60, type=int, help="Number of epochs for soft.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--theta_sampling", default=20, type=int, help="Number theta angles.")
    parser.add_argument("--lr", default=0.001, type=float, help="Soft learning rate")
    parser.add_argument("--batch_size", default=20, type=int, help="Batch size.")
    parser.add_argument("--neurons", default=64, type=int, help="Number of neurons.")
    parser.add_argument("--layers", default=2, type=int, help="Number of layers.")
    parser.add_argument("--M", default=0.5, type=float, help="mass")
    parser.add_argument("--folder_name", default=DEFAULT_folder_name, type=str, help="Folder name")
    parser.add_argument("--cuda", default=False, action="store_true", help="Use CUDA (under construction).")

    args = parser.parse_args([] if "__file__" not in globals() else None)

    if args.dt == 0.0: #automatic
        args.dt = resolve_automatic_dt(args)

    check_folder(args.folder_name) #check whether the folders data and saved_models exist, or create them

    #save args to file
    original_stdout = sys.stdout
    with open(args.folder_name+'/args.txt', 'w') as f:
        sys.stdout = f # Change the standard output to the file we created.
        print(args)
        sys.stdout = original_stdout

    if args.generate:
        print("-------------------------------")
        print("Generating trajectories.")    
        print("-------------------------------")
        generate_trajectories(args)
        args.generate = False

    if args.implicit:
        print("-------------------------------")
        print("Learning implicit Jacobi.")    
        print("-------------------------------")
        if args.scheme == "IMR":
            learner = LearnerIMR(model=args.model, neurons = args.neurons, layers = args.layers, batch_size = args.batch_size, dt = args.dt, name = args.folder_name, cuda = args.cuda)
        else:
            learner = Learner(model=args.model, neurons = args.neurons, layers = args.layers, batch_size = args.batch_size, dt = args.dt, name = args.folder_name, cuda = args.cuda)
        learner.learn(method = "implicit", learning_rate = args.lr, epochs = args.epochs, prefactor = args.prefactor)
    if args.soft:
        print("-------------------------------")
        print("Learning soft Jacobi.")    
        print("-------------------------------")
        if args.scheme == "IMR":
            learner = LearnerIMR(model=args.model, neurons = args.neurons, layers = args.layers, batch_size = args.batch_size, dt = args.dt, name = args.folder_name, cuda = args.cuda)
        else:
            learner = Learner(model=args.model, neurons = args.neurons, layers = args.layers, batch_size = args.batch_size, dt = args.dt, name = args.folder_name, cuda = args.cuda)
        learner.learn(method = "soft", learning_rate = args.lr, epochs = args.epochs, prefactor = args.prefactor, jac_prefactor = args.jac_prefactor)
    if args.without:
        print("-------------------------------")
        print("Learning without Jacobi.")    
        print("-------------------------------")
        if args.scheme == "IMR":
            learner = LearnerIMR(model=args.model, neurons = args.neurons, layers = args.layers, batch_size = args.batch_size, dt = args.dt, name = args.folder_name, cuda = args.cuda)
        else:
            learner = Learner(model=args.model, neurons = args.neurons, layers = args.layers, batch_size = args.batch_size, dt = args.dt, name = args.folder_name, cuda = args.cuda)
        learner.learn(method = "without", learning_rate = args.lr, epochs = args.epochs, prefactor = args.prefactor)
    if not args.no_show:
        plot_training_errors(args)

    generate_trajectories(args)  
    
