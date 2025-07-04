import argparse
import os
import datetime
import re
import time

import torch
from torch.utils.data import DataLoader
import torchmetrics
from torch import einsum

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from models.Model import EnergyNet, TensorNet, JacVectorNet
from TrajectoryDataset import TrajectoryDataset

#declare default values of parameters
DEFAULT_dataset = "data/dataset.xyz"
DEFAULT_batch_size = 20
DEFAULT_dt = 0.1 
DEFAULT_learning_rate = 1.0e-05
DEFAULT_epochs = 10 
DEFAULT_prefactor = 1.0
DEFAULT_jacobi_prefactor = 1.0
DEFAULT_neurons = 64
DEFAULT_layers = 2
DEFAULT_folder_name = "."

class Learner(object):
    def __init__(self, model, batch_size = DEFAULT_batch_size, dt = DEFAULT_dt, neurons = DEFAULT_neurons, layers = DEFAULT_layers, name = DEFAULT_folder_name, cuda = False):
        self.model = model
        dim = 0
        if model == "RB":
            dim = 3 #input features
        elif model in ["HT", "P3D", "K3D"]:
            dim = 6
        elif model == "P2D" or model == "Sh":
            dim = 4
        else:
            raise Exception("Unknown model "+model)
        print("Generating Learner for model ", model, " with ", dim, " dimensions.")
        self.name = name


        #self.logdir = os.path.join("logs", "{}-{}-{}".format(
        #    os.path.basename(globals().get("__file__", "notebook")),
        #    datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        #    ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
        #))

        self.df = pd.read_csv(name+"/"+DEFAULT_dataset, dtype=np.float32)

        self.energy = EnergyNet(dim, neurons, layers, batch_size)
        self.L_tensor = TensorNet(dim, neurons, layers, batch_size)
        self.jac_vec = JacVectorNet(dim, neurons, layers, batch_size)
        self.device = None
        if cuda:
            self.device = torch.cuda.current_device()
            self.energy = self.energy.to(self.device)
            self.L_tensor = self.L_tensor.to(self.device)
            self.jac_vec = self.jac_vec.to(self.device)

        self.train, self.test = train_test_split(self.df, test_size=0.4)
        self.train_dataset = TrajectoryDataset(self.train, model = model)
        self.valid_dataset = TrajectoryDataset(self.test, model = model)

        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=batch_size, shuffle=True)

        #loss weighting
        self.dt = dt
        #self.lam = 0.5

        #constant in M

        #for soft and without
        self.train_metric = torchmetrics.MeanSquaredError()
        self.val_metric = torchmetrics.MeanSquaredError()

        #for Jacobi
        self.train_metric_reg = torchmetrics.MeanSquaredError()
        self.val_metric_reg = torchmetrics.MeanSquaredError()

        self.loss_fn = torch.nn.MSELoss()

        self.train_errors = []
        self.validation_errors = []

    def mov_loss_without(self, zn_tensor, zn2_tensor, mid_tensor):
        En = self.energy(zn_tensor)
        En2 = self.energy(zn2_tensor)

        E_z = torch.autograd.grad(En.sum(), zn_tensor, only_inputs=True, create_graph=True)[0]
        E_z2 = torch.autograd.grad(En2.sum(), zn2_tensor, only_inputs=True, create_graph=True)[0]

        Lz = self.L_tensor(zn_tensor)
        Lz2 = self.L_tensor(zn2_tensor)
        return (zn_tensor - zn2_tensor)/self.dt + 1.0/2.0*(torch.matmul(Lz, E_z.unsqueeze(2)).squeeze() \
                                                        + torch.matmul(Lz2, E_z2.unsqueeze(2)).squeeze())

    def mov_loss_without_with_jacobi(self, zn_tensor, zn2_tensor, mid_tensor, reduced_L):
        En = self.energy(zn_tensor)
        En2 = self.energy(zn2_tensor)

        E_z = torch.autograd.grad(En.sum(), zn_tensor, only_inputs=True, create_graph=True)[0]
        E_z2 = torch.autograd.grad(En2.sum(), zn2_tensor, only_inputs=True, create_graph=True)[0]

        Lz = self.L_tensor(zn_tensor)
        Lz2 = self.L_tensor(zn2_tensor)

        jacobi_loss = self.jacobi_loss(zn_tensor, Lz, reduced_L) 

        return (zn_tensor - zn2_tensor)/self.dt + 1.0/2.0*(torch.matmul(Lz, E_z.unsqueeze(2)).squeeze() \
                                                        + torch.matmul(Lz2, E_z2.unsqueeze(2)).squeeze()), jacobi_loss

    def jacobi_loss(self, zn_tensor, Lz, reduced_L):
        Lz_grad = torch.autograd.functional.jacobian(reduced_L, zn_tensor, create_graph=True).permute(2, 0, 1, 3)
        term1 = einsum('mkl,mijk->mijl', Lz, Lz_grad)
        term2 = term1.permute(0,2,3,1)
        term3 = term1.permute(0,3,1,2)
        return term1 + term2 + term3

    def mov_loss_soft(self, zn_tensor, zn2_tensor, mid_tensor, reduced_L):
        En = self.energy(zn_tensor)
        En2 = self.energy(zn2_tensor)

        E_z = torch.autograd.grad(En.sum(), zn_tensor, only_inputs=True, create_graph=True)[0]
        E_z2 = torch.autograd.grad(En2.sum(), zn2_tensor, only_inputs=True, create_graph=True)[0]

        Lz = self.L_tensor(zn_tensor)
        Lz2 = self.L_tensor(zn2_tensor)
        mov_loss = (zn_tensor - zn2_tensor)/self.dt + 1.0/2.0*(torch.matmul(Lz, E_z.unsqueeze(2)).squeeze() \
                                                        + torch.matmul(Lz2, E_z2.unsqueeze(2)).squeeze())
        #Jacobi
        jacobi_loss = self.jacobi_loss(zn_tensor, Lz, reduced_L) 
        return mov_loss, jacobi_loss

        
    def mov_loss_implicit(self, zn_tensor, zn2_tensor, mid_tensor):
        En = self.energy(zn_tensor)
        En2 = self.energy(zn2_tensor)

        E_z = torch.autograd.grad(En.sum(), zn_tensor, only_inputs=True, create_graph=True)[0]
        E_z2 = torch.autograd.grad(En2.sum(), zn2_tensor, only_inputs=True, create_graph=True)[0]
 
        Jz, cass = self.jac_vec(zn_tensor)
        Jz2, cass2 = self.jac_vec(zn2_tensor)
        return (zn_tensor - zn2_tensor)/self.dt + 1.0/2.0*(torch.cross(Jz, E_z, dim=1) + torch.cross(Jz2, E_z2, dim=1))

    def learn(self, method = "without", learning_rate = DEFAULT_learning_rate, epochs = DEFAULT_epochs, prefactor = DEFAULT_prefactor, jac_prefactor = DEFAULT_jacobi_prefactor, scheme="IMR"):
        if method not in ["without", "soft", "implicit"]:
            raise Exception("Unknown method "+method)
        print("Learning from folder "+self.name) 
        print("Method = "+method)
        print("Epochs = "+str(epochs))
        optimizer, scheduler = None, None

        if method in ["without", "soft"]:
            optimizer = torch.optim.Adam(list(self.energy.parameters())
                                    + list(self.L_tensor.parameters()), lr = learning_rate)
        elif method == "implicit":
            optimizer = torch.optim.Adam(list(self.energy.parameters())
                             + list(self.jac_vec.parameters()), lr = learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.98, last_epoch= -1)


        mov_loss, jacobi_loss, Lz, Lz2, Jz, Jz2, cass, cass2, reduced_L = None, None, None, None, None, None, None, None, None
        for epoch in range(epochs):
            print("\nStart of epoch %d" % (epoch,))
            start_time = time.time()
            reduced_L = lambda z: torch.sum(self.L_tensor(z), axis=0)

            # Iterate over the batches of the dataset.
            for step, (zn_tensor, zn2_tensor, mid_tensor) in enumerate(self.train_loader):
                #print("zn = ", zn_tensor)
                #print("zn2 = ", zn2_tensor)
                # zero the parameter gradients
                if self.device != None:
                    zn_tensor = zn_tensor.to(self.device)
                    zn2_tensor = zn2_tensor.to(self.device)
                    mid_tensor = mid_tensor.to(self.device)
                optimizer.zero_grad()
                zn_tensor.requires_grad = True
                zn2_tensor.requires_grad = True
                mid_tensor.requires_grad = True

                if method == "without":
                    mov_loss = self.mov_loss_without(zn_tensor, zn2_tensor, mid_tensor)
                elif method == "implicit":
                    mov_loss = self.mov_loss_implicit(zn_tensor, zn2_tensor, mid_tensor)
                elif method == "soft":
                    mov_loss, jacobi_loss = self.mov_loss_soft(zn_tensor, zn2_tensor, mid_tensor, reduced_L)

                mov_value = self.loss_fn(torch.zeros_like(mov_loss), prefactor*mov_loss)
                loss = mov_value 
                self.train_metric(torch.zeros_like(mov_value), mov_value)
                if method == "soft":
                    reg_value = self.loss_fn(torch.zeros_like(jacobi_loss), jac_prefactor*jacobi_loss)
                    loss += reg_value
                    self.train_metric_reg(torch.zeros_like(reg_value), reg_value)

                # Use the backward method tape to automatically retrieve
                # the gradients of the trainable variables with respect to the loss.
                loss.backward()
                optimizer.step()

                # Run one step of gradient descent by updating
                # the value of the variables to minimize the loss.

                # Log every 200 batches.
                if step % 200 == 0:
                    if method == "soft":
                        print(
                        "Training loss (for one batch) at step %d: movement %.4f and reg %.4f"
                        % (step, float(mov_value), float(reg_value))
                        )
                    else:
                        print(
                            "Training loss (for one batch) at step %d: movement %.4f "
                            % (step, float(mov_value))
                        )
                    print("Seen so far: %s samples" % ((step + 1) * 64))


            # Display metrics at the end of each epoch.
            train_acc = self.train_metric.compute()
            self.train_metric.reset()
            if method == "soft":
                train_acc_reg = self.train_metric_reg.compute()
                self.train_metric_reg.reset()
                print("Training err over epoch: %.4f reg %.4f" % (float(train_acc), float(train_acc_reg)))
                self.train_errors.append([float(train_acc), float(train_acc_reg)])
            else:
                print("Training err over epoch: %.4f " % (float(train_acc)))
                self.train_errors.append([float(train_acc), 0.0])

            # Run a validation loop at the end of each epoch.
            jacobi_loss = None
            for step, (zn_tensor, zn2_tensor, mid_tensor) in enumerate(self.valid_loader):

                zn_tensor.requires_grad = True
                zn2_tensor.requires_grad = True
                mid_tensor.requires_grad = True

                if method == "without":
                    mov_loss, jacobi_loss = self.mov_loss_without_with_jacobi(zn_tensor, zn2_tensor, mid_tensor, reduced_L)
                elif method == "implicit":
                    mov_loss = self.mov_loss_implicit(zn_tensor, zn2_tensor, mid_tensor) #Jacobi identity automatically satisfied
                elif method == "soft":
                    mov_loss, jacobi_loss = self.mov_loss_soft(zn_tensor, zn2_tensor, mid_tensor, reduced_L)

                mov_value = self.loss_fn(torch.zeros_like(mov_loss), prefactor*mov_loss)
                self.val_metric(torch.zeros_like(mov_value), mov_value)
                if jacobi_loss != None:
                    reg_value = self.loss_fn(torch.zeros_like(jacobi_loss), jac_prefactor*jacobi_loss)
                    self.val_metric_reg(torch.zeros_like(reg_value), reg_value)

            val_acc_val = self.val_metric.compute()
            self.val_metric.reset()
            if jacobi_loss != None:
                val_acc_reg = self.val_metric_reg.compute()
                self.val_metric_reg.reset()
                self.validation_errors.append([float(val_acc_val), float(val_acc_reg)])
                print("Validation error: value %.4f" % float(val_acc_val), float(val_acc_reg))
            else: #implicit
                self.validation_errors.append([float(val_acc_val), 0.0])
                print("Validation error: value %.4f" % (float(val_acc_val)))

            print("Time taken: %.2fs" % (time.time() - start_time))
            scheduler.step()

        errors = np.hstack((self.train_errors, self.validation_errors))
        errors_df = pd.DataFrame(errors, columns = ["train_mov", "train_reg", "val_mov", "val_reg"]) 
        if method == "without":
            torch.save(self.energy, self.name+'/saved_models/without_jacobi_energy')
            torch.save(self.L_tensor, self.name+'/saved_models/without_jacobi_L')
            errors_df.to_csv(self.name+"/data/errors_without.csv")
        elif method == "implicit":
            torch.save(self.energy, self.name+'/saved_models/implicit_jacobi_energy')
            torch.save(self.jac_vec, self.name+'/saved_models/implicit_jacobi_J')
            errors_df.to_csv(self.name+"/data/errors_implicit.csv")
        elif method == "soft":
            torch.save(self.energy, self.name+'/saved_models/soft_jacobi_energy')
            torch.save(self.L_tensor, self.name+'/saved_models/soft_jacobi_L')
            errors_df.to_csv(self.name+"/data/errors_soft.csv")

class LearnerIMR(Learner):
    def mov_loss_without(self, zn_tensor, zn2_tensor, mid_tensor):
        En = self.energy(mid_tensor)
        E_z = torch.autograd.grad(En.sum(), mid_tensor, only_inputs=True, create_graph=True)[0]
        Lz = self.L_tensor(mid_tensor)
        return (zn_tensor - zn2_tensor)/self.dt + torch.matmul(Lz, E_z.unsqueeze(2)).squeeze()
                                                        

    def mov_loss_without_with_jacobi(self, zn_tensor, zn2_tensor, mid_tensor, reduced_L):
        En = self.energy(mid_tensor)
        E_z = torch.autograd.grad(En.sum(), mid_tensor, only_inputs=True, create_graph=True)[0]
        Lz = self.L_tensor(mid_tensor)
        jacobi_loss = self.jacobi_loss(mid_tensor, Lz, reduced_L) 
        return (zn_tensor - zn2_tensor)/self.dt + torch.matmul(Lz, E_z.unsqueeze(2)).squeeze(), jacobi_loss

    def mov_loss_soft(self, zn_tensor, zn2_tensor, mid_tensor, reduced_L):
        En = self.energy(mid_tensor)
        E_z = torch.autograd.grad(En.sum(), mid_tensor, only_inputs=True, create_graph=True)[0]
        Lz = self.L_tensor(mid_tensor)
        mov_loss = (zn_tensor - zn2_tensor)/self.dt + torch.matmul(Lz, E_z.unsqueeze(2)).squeeze() 
        #Jacobi
        jacobi_loss = self.jacobi_loss(mid_tensor, Lz, reduced_L) 
        return mov_loss, jacobi_loss

        
    def mov_loss_implicit(self, zn_tensor, zn2_tensor, mid_tensor):
        En = self.energy(mid_tensor)
        E_z = torch.autograd.grad(En.sum(), mid_tensor, only_inputs=True, create_graph=True)[0]
 
        Jz, cass = self.jac_vec(mid_tensor)
        Jz2, cass2 = self.jac_vec(mid_tensor)
        return (zn_tensor - zn2_tensor)/self.dt + torch.cross(Jz, E_z, dim=1) 


def check_folder(name):
    print("Checking folder ", name)
    name = os.getcwd()+"/"+name
    data_name = name+"/data"
    models_name = name+"/saved_models"
    if not os.path.exists(data_name):
        print("Making folder: "+data_name)
        os.makedirs(data_name)
    if not os.path.exists(models_name):
        print("Making folder: "+models_name)
        os.makedirs(models_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=DEFAULT_epochs, type=int, help="Number of epochs")
    parser.add_argument("--neurons", default=DEFAULT_neurons, type=int, help="Number of hidden neurons")
    parser.add_argument("--layers", default=DEFAULT_layers, type=int, help="Number of layers")
    parser.add_argument("--batch_size", default=DEFAULT_batch_size, type=int, help="Batch size")
    parser.add_argument("--dt", default=DEFAULT_dt, type=float, help="Step size")
    parser.add_argument("--prefactor", default=DEFAULT_prefactor, type=float, help="Prefator in the loss")
    parser.add_argument("--learning_rate", default=DEFAULT_learning_rate, type=float, help="Learning rate")
    parser.add_argument("--model", type=str, help="Model = RB, HT, or P3D.", required = True)
    parser.add_argument("--name", default = DEFAULT_folder_name, type=str, help="Folder name")
    parser.add_argument("--method", default = "without", type=str, help="Method: without, implicit, or soft")
    #parser.parse_args(['-h'])

    args = parser.parse_args([] if "__file__" not in globals() else None)

    check_folder(args.folder_name) #check whether the folders data and saved_models exist, or create them

    learner = Learner(args.model, neurons = args.neurons, layers = args.layers, batch_size = args.batch_size, dt = args.dt, name = args.folder_name)
    learner.learn(method = args.method, learning_rate = args.learning_rate, epochs = args.epochs, prefactor = args.prefactor)


