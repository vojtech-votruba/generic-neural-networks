from copy import deepcopy
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import numpy as np
import comparison
from math import sqrt
from learn import DEFAULT_folder_name
import os
import torch
from matplotlib import cm

parser = argparse.ArgumentParser()
parser.add_argument("--plot_steps", default=None, type=int, help="Steps to plot")
parser.add_argument("--plot_every", default=1, type=int, help="Every nth point to plot")
parser.add_argument("--plot_m", default=False, action="store_true", help="Plot m.")
parser.add_argument("--plot_E", default=False, action="store_true",help="Plot energy.")
parser.add_argument("--plot_L", default=False, action="store_true", help="Plot Poisson bivector.")
parser.add_argument("--plot_msq", default=False, action="store_true", help="Plot m squared.")
parser.add_argument("--plot_L_errors", default=False, action="store_true", help="Plot erros in L.")
parser.add_argument("--plot_RB_errors", default=False, action="store_true", help="Plot erros in m.")
parser.add_argument("--plot_Sh_errors", default=False, action="store_true", help="Plot erros in Shivamoggi coordinates.")
parser.add_argument("--plot_P2D_errors", default=False, action="store_true", help="Plot erros in m and r, 2D.")
parser.add_argument("--plot_P3D_errors", default=False, action="store_true", help="Plot erros in m and r, 3D.")
parser.add_argument("--plot_HT_errors", default=False, action="store_true", help="Plot erros in m and r, 3D.")
parser.add_argument("--plot_msq_errors", default=False, action="store_true", help="Plot erros in m.")
parser.add_argument("--plot_rsq_errors", default=False, action="store_true", help="Plot erros in r.")
parser.add_argument("--plot_mrs_errors", default=False, action="store_true", help="Plot erros in m.r.")
parser.add_argument("--plot_am_errors", default=False, action="store_true", help="Plot erros in angular momentum (r x p).")
parser.add_argument("--plot_training_errors", default=False, action="store_true", help="Plot training erros.")
parser.add_argument("--plot_Casimir", default=False, action="store_true", help="Plot Casimirs.")
parser.add_argument("--plot_r", default=False, action="store_true", help="Plot r (in HT).")
parser.add_argument("--plot_dataset", default=False, action="store_true", help="Plot the dataset")
parser.add_argument("--model", default="RB", type=str, help="Model: RB or HT.")
parser.add_argument("--plot_det", default = False, action="store_true", help="Plot det of L.")
parser.add_argument("--plot_det_log", default = False, action="store_true", help="Plot log det of L.")
parser.add_argument("--implicit", default = False, action="store_true", help="Implicit")
parser.add_argument("--without", default = False, action="store_true", help="Without")
parser.add_argument("--soft", default = False, action="store_true", help="Soft")
parser.add_argument("--dataset", default=False, action="store_true", help="dataset")
parser.add_argument("--GT", default=False, action="store_true", help="grand truth")
parser.add_argument("--plot_field", default="None", type=str, help="Plot field")
parser.add_argument("--plot_first", default="None", type=str, help="Plot field")
parser.add_argument("--folder_name", default=DEFAULT_folder_name, type=str, help="Folder name")
parser.add_argument("--export", default=False, action="store_true", help="Save figures and logs.")
parser.add_argument("--plot_first_mx", default=False, action="store_true", help="Show mx in the first path.")
parser.add_argument("--plot_first_rx", default=False, action="store_true", help="Show rx in the first path.")
parser.add_argument("--plot_Jacobi", default=False, action="store_true", help="Plot Jacobiator during validation.")
parser.add_argument("--no_clean", default=False, action="store_true", help="Don't clean the log.")
parser.add_argument("--init_mx", default=10.0, type=float, help="A value of momentum, x component")
parser.add_argument("--init_my", default=3.0, type=float, help="A value of momentum, y component") 
parser.add_argument("--init_mz", default=4.0, type=float, help="A value momentum, z component")
parser.add_argument("--density", default=60, type=int, help="Plot density in each dimension (total No. points is density**2)")
parser.add_argument("--plot_Es", default=False, action="store_true", help="Plot E from uniform ball.")
parser.add_argument("--plot_Ls", default=False, action="store_true", help="Plot L from uniform ball.")
parser.add_argument("--plot_spectrum_errors", default=False, action="store_true", help="Plot spectrum-of-L errors.")
parser.add_argument("--plot_compatibility", default=False, action="store_true", help="Plot L compatibility errors")
#parser.add_argument("--plot_compatibility_hist", default=False, action="store_true", help="Plot L compatibility errors histogram")


args = parser.parse_args([] if "__file__" not in globals() else None)

flds, dfli, dflw, dfls, dfgt = None, None, None, None, None

file_name_dataset = args.folder_name+"/data/dataset.xyz"
dfds = pd.read_csv(file_name_dataset, nrows=args.plot_steps)

if args.GT:
    file_name_GT = args.folder_name+"/data/generalization.xyz"
    dfgt = pd.read_csv(file_name_GT, nrows=args.plot_steps)

if args.implicit:
    file_name_learned_implicit = args.folder_name+"/data/learned_implicit.xyz"
    dfli = pd.read_csv(file_name_learned_implicit, nrows=args.plot_steps)

if args.soft:
    file_name_learned_soft = args.folder_name+"/data/learned_soft.xyz"
    dfls = pd.read_csv(file_name_learned_soft, nrows=args.plot_steps)

if args.without:
    file_name_learned_without = args.folder_name+"/data/learned_without.xyz"
    dflw = pd.read_csv(file_name_learned_without, nrows=args.plot_steps)

methods = {
    "without":{"df": dflw, "title": "Learned Without"},
    "soft":{"df": dfls, "title": "Learned Soft"},
    "GT":{"df": dfgt, "title": "GT"},
    "dataset":{"df": dfds, "title": "dataset"}}
if args.implicit:
    methods["implicit"] = {"df": dfli, "title": "Learned implicit"}

def get_frames_and_titles():
    fields = []
    titles = []
    if args.without:
        fields.append(methods["without"]["df"])        
        titles.append(methods["without"]["title"])              
    if args.soft:
        fields.append(methods["soft"]["df"])        
        titles.append(methods["soft"]["title"])        
    if args.implicit:
        fields.append(methods["implicit"]["df"])        
        titles.append(methods["implicit"]["title"])        
    if args.GT:
        fields.append(methods["GT"]["df"])        
        titles.append(methods["GT"]["title"])        
    if args.dataset:
        fields.append(methods["dataset"]["df"])        
        titles.append(methods["dataset"]["title"])        
    return fields, titles

frames, titles = get_frames_and_titles()
print("Dataframes: ", titles)

def add_log(key, value):
    log_file = open(args.folder_name+"/log.txt", "a")
    log_file.write(str(key)+", "+str(value)+"\n")
    log_file.close()

if args.export and not args.no_clean:
    if os.path.exists(args.folder_name+"/log.txt"):
        print("Cleaning the log.")
        os.remove(args.folder_name+"/log.txt")

def remove_returns(xs, ys): #remove points that have earlier timestep than their predecessor
    print(len(xs))
    newxs = [xs[0]]
    newys = [ys[0]]
    for i in range(1,len(xs)):
        if xs[i] > xs[i-1]:
            newxs.append(xs[i])
            newys.append(ys[i])
    return newxs, newys

def filter(xs, ys):
    newxs = [xs[0]]
    newys = [ys[0]]
    for i in range(1,len(xs)):
        if i % args.plot_every == 0:
            newxs.append(xs[i])
            newys.append(ys[i])
    return np.array(newxs), np.array(newys)

def sort_data(xs, ys):
    p = np.argsort(xs)
    newxs = xs[p]
    newys = ys[p]
    return newxs, newys

def add_plot(ax, x,y, name="", apply_filter=False, split = True):
    #ax.scatter(x[::args.plot_every],y[::args.plot_every])
    #newxs, newys = remove_returns(x[::args.plot_every],y[::args.plot_every])
    newxs, newys = x, y
    if apply_filter:
        newxs, newys = filter(x[::args.plot_every],y[::args.plot_every])
    #if sort:
    #    newxs, newys = sort_data(newxs, newys)
    if split:
        paths = split_data_to_forward_paths(x, y)
        ts = x[:len(paths[0])]
        ax.plot(ts, paths[0], lw=0.7, label = name)
        for i in range(1, len(paths)):
            ax.plot(ts, paths[i], lw=0.7)
    else:
        ax.plot(newxs, newys, lw=0.7, label=name)
    #newxs, newys = sort_data(x[::args.plot_every],y[::args.plot_every])
    #ax.plot(remove_returns(x[::args.plot_every],y[::args.plot_every]), lw=0.7, label=name)

def fields_to_string(fields):
    result = str(fields[0])
    for i in range(1, len(fields)):
        result += "_"+str(fields[i])
    return result

def plot_field(fields=["mx", "my", "mz"]):
    file_name = ""
    for data_frame, name in zip(frames, titles):
        for col in fields:
            plt.xlabel("time")
            add_plot(plt, data_frame["time"], data_frame[col], name=name+": "+col)
            file_name += "_"+name+"-"+col
    #elif True:
    #    for field in fields:
    #        plt.plot(df["time"], df[field])
    #        plt.plot(dfl["time"], dfl[field])
    #        plt.plot(dfls["time"], dfls[field])
    #else:
    #    for field in fields:
    #        ax[0].plot(df["time"], df[field])
    #        ax[0].plot(dfl["time"], dfl[field])
    #        ax[0].plot(dfls["time"], dfls[field])
    plt.legend()
    if args.export:
        file_name = args.folder_name+"/"+file_name+".png"
        print("Exporting figure to: "+file_name)
        plt.savefig(file_name) 
    plt.show()

def split_data_to_forward_paths(xs, ys):
    if len(xs) != len(ys):
        raise Exception("Length of xs and ys not equal.")
    paths = [[ys[0]]]
    for i in range(1,len(xs)):
        if xs[i] < xs[i-1]: #return to time zero, new path
            paths.append([ys[i]])
        else:
            paths[len(paths)-1].append(ys[i]) #add value to the current path
    return paths

def split_to_forward_paths(df, field): #return array of forward paths for data frame df and a given field
    ts = df["time"] #times
    values = df[field] #values to be split to pths
    return split_data_to_forward_paths(ts, values)

if args.plot_m:
    plot_field(fields = ["mx", "my", "mz"])

if args.plot_field != "None":
    plot_field(fields = [args.plot_field])

if args.plot_r:
    plot_field(fields = ["rx", "ry", "rz"])

if args.plot_E:
    plot_field(fields = ["E"])

if args.plot_L:
    if args.model == "RB":
        plot_field(fields = ["L_01", "L_02", "L_12"])
    else:
        if args.model in ["HT", "P3D", "K3D"]:
            plot_field(fields = ["L_01", "L_02", "L_03", "L_04", "L_05", "L_12", "L_13", "L_14", "L_15", "L_23", "L_24", "L_25", "L_34", "L_35", "L_45"])

if args.plot_msq:
    plot_field(fields = ["sqm"])

def normalize(Ls):
    dim = len(Ls[0])
    print("Normalization to canonical L for dimension ", dim)
    normalization = np.linalg.norm(Ls)/sqrt(len(Ls)/dim)
    return Ls*dim/normalization # overall normalization to the canonical form
    #return [Ls[i]/np.linalg.norm(Ls[i])*sqrt(dim) for i in range(len(Ls))] #normalization of each time step

def compat_error3D(J_exact, J, m):
    grad_J_0, = torch.autograd.grad(J[:,0].sum(), m, retain_graph=True, create_graph=False) # DJ0/Dm
    grad_J_1, = torch.autograd.grad(J[:,1].sum(), m, retain_graph=True, create_graph=False) # DJ1/Dm
    grad_J_2, = torch.autograd.grad(J[:,2].sum(), m, retain_graph=True, create_graph=False) # DJ2/Dm
    #grad_J = torch.stack((grad_J_0, grad_J_1, grad_J_2), axis=2) # Aij DJi/Dmj batch dimension in front

    rot_tuple = (grad_J_2[:,1] - grad_J_1[:,2], grad_J_0[:,2] - grad_J_2[:,0], grad_J_1[:,0] - grad_J_0[:,1])
    rot_J = torch.stack(rot_tuple, axis=1)

    return torch.square(torch.sum(J_exact*rot_J, axis=1))

def dotUV(U, V):
    shape = U.shape
    return [np.dot(U[i], V[i]) for i in range(shape[0])]

def compat_error_superintegrable(U_exact, V_exact, U, V):
    #shape = U_exact.shape
    Lambda = dotUV(U_exact, V) + dotUV(U, V_exact)
    return np.square(Lambda)

def get_learned_models():
    models = {}
    if args.without:
        models["without"] = {"energy":torch.load(args.folder_name+"/saved_models/without_jacobi_energy"), "L":torch.load(args.folder_name+"/saved_models/without_jacobi_L")}
    if args.soft:
        models["soft"] = {"energy":torch.load(args.folder_name+"/saved_models/soft_jacobi_energy"), "L":torch.load(args.folder_name+"/saved_models/soft_jacobi_L")}
    if args.implicit:
        models["implicit"] = {"energy":torch.load(args.folder_name+"/saved_models/implicit_jacobi_energy"), "J":torch.load(args.folder_name+"/saved_models/implicit_jacobi_J")}
    return models

def load_normalized_Ls(df, dim): 
    if dim == 3:
        L_01s = data_frame["L_01"]
        L_02s = data_frame["L_02"]
        L_12s = data_frame["L_12"]

        Ls = np.array([
            [[0.0, L_01s[i], L_02s[i]],
            [0.0, 0.0, L_12s[i]],
            [0.0, 0.0, 0.0]] for i in range(0,len(L_01s))
            ])
    elif dim == 4:
        L_01s = data_frame["L_01"]
        L_02s = data_frame["L_02"]
        L_03s = data_frame["L_03"]
        L_12s = data_frame["L_12"]
        L_13s = data_frame["L_13"]
        L_23s = data_frame["L_23"]

        Ls = np.array([
            [[0.0, L_01s[i], L_02s[i], L_03s[i]],
            [0.0, 0.0, L_12s[i], L_13s[i]],
            [0.0, 0.0, 0.0, L_23s[i]],
            [0.0, 0.0, 0.0, 0.0]] for i in range(0,len(L_01s))
            ])
    elif dim == 6:
        L_01s = data_frame["L_01"]
        L_02s = data_frame["L_02"]
        L_03s = data_frame["L_03"]
        L_04s = data_frame["L_04"]
        L_05s = data_frame["L_05"]
        L_12s = data_frame["L_12"]
        L_13s = data_frame["L_13"]
        L_14s = data_frame["L_14"]
        L_15s = data_frame["L_15"]
        L_23s = data_frame["L_23"]
        L_24s = data_frame["L_24"]
        L_25s = data_frame["L_25"]
        L_34s = data_frame["L_34"]
        L_35s = data_frame["L_35"]
        L_45s = data_frame["L_45"]

        Ls = np.array([
            [[0.0, L_01s[i], L_02s[i], L_03s[i], L_04s[i], L_05s[i]],
            [0.0, 0.0, L_12s[i], L_13s[i], L_14s[i], L_15s[i]],
            [0.0, 0.0, 0.0, L_23s[i], L_24s[i], L_25s[i]],
            [0.0, 0.0, 0.0, 0.0, L_34s[i], L_35s[i]],
            [0.0, 0.0, 0.0, 0.0, 0.0, L_45s[i]],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] for i in range(0,len(L_01s))])
    else:
        raise Exception("Dimension not implemented")

    Ls -= np.transpose(Ls, (0,2,1))
    Ls = normalize(Ls)
    return Ls

def reject_outliers(data, m=3):
    result = data[abs(data - np.mean(data)) < m * np.std(data)]
    rejected = len(data) - len(results)
    if rejected > 0:
        print("Warning: Rejecting ", rejected, " out of ", len(data))
    return result

if args.plot_compatibility:
    file_name = "compatibility_errors"
    models = get_learned_models() #learinig models: without, implicit, or soft
    if args.model == "RB":
        for model in models:
            data_frame = methods[model]["df"]
            title = methods[model]["title"]
            mx = torch.tensor(data_frame["mx"], dtype=torch.float32)
            my = torch.tensor(data_frame["my"], dtype=torch.float32)
            mz = torch.tensor(data_frame["mz"], dtype=torch.float32)
            m = torch.stack((mx, my, mz), axis=1)
            m.requires_grad=True
            total_m_batched = torch.split(m, 20)
            J_exact = torch.stack((mx, my, mz), axis=1)
            
            if model != "implicit":
                L_tensor = models[model]["L"]
                L_batched = [L_tensor(b) for b in total_m_batched]
                L = torch.cat(L_batched, axis=0)
                J = torch.stack((-L[:, 1,2], L[:, 0, 2], -L[:, 0,1]), axis=1)
            else:
                J_vector = models["implicit"]["J"]
                J_batched = [J_vector(b)[0] for b in total_m_batched]
                J = torch.cat(J_batched, axis=0)

            #Normalize
            multiplier = 1000 # We will be dividing by total J. Lets have errors on the order of maginutde around 1
            J = J/J.sum().detach()*multiplier

            errors = np.log10(compat_error3D(J_exact, J, m))
            #average_error = (np.trapz(errors)+0.0)/len(errors)
            average_error = np.median(errors)
            print("Median square L error for ", model, " is: ", average_error)

            #total_error = error.sum()
            #print("Total error "+model, total_error)
            #plt.plot(errors, label=model)

            try:
                plt.hist(errors, bins=100, label=model, alpha=0.6)
            except:
                pass
            if args.export:
                add_log(file_name+" "+model, average_error)

    elif args.model == "P2D":
        for data_frame, name in zip(frames, titles):
            Ls = load_normalized_Ls(data_frame, 4)

            rxs = data_frame["rx"]
            rys = data_frame["ry"]
            mxs = data_frame["mx"]
            mys = data_frame["my"]

            Ls_exact = np.array(
                [[0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0]])
            Ls_exact -= np.transpose(Ls_exact)

            errors = np.log10([np.linalg.norm(np.matmul(Ls[i], Ls_exact) - np.matmul(Ls_exact, Ls[i])) for i in range(len(Ls))])
            #iterations = np.linspace(1,len(errors),len(errors))

            average_error = np.median(errors)
            print("Median norm log_10 of L error for ", name, " is: ", average_error)

            #add_plot(plt, iterations, errors, name=name+": L_error")
            file_name += "-"+name
            try:
                plt.hist(errors, bins=100, label=name, alpha=0.6)
            except:
                pass
            if args.export:
                add_log(file_name+" "+name, average_error)
            
    elif args.model == "Sh": 
        for data_frame, name in zip(frames, titles):
            if name == "Learned implicit":
                raise Exception("Implicit not implemented.")

            Ls = load_normalized_Ls(data_frame, 4)
            U = np.transpose(np.vstack((-Ls[:, 0,1], -Ls[:, 0, 2], -Ls[:, 0,3])))
            V = np.transpose(np.vstack((-Ls[:, 2,3], Ls[:, 1, 3], -Ls[:, 1,2])))
            print("U.V = ",np.linalg.norm(dotUV(U, V))/np.sqrt(U.shape[0]))

            u = data_frame["u"]
            x = data_frame["x"]
            y = data_frame["y"]
            z = data_frame["z"]
            zeros = np.zeros(len(u))
            U_exact = np.transpose(np.vstack((zeros, -u, zeros)))
            V_exact = np.transpose(np.vstack((-x, zeros, z)))

            #Normalize
            #multiplier = 1000 # We will be dividing by total J. Lets have errors on the order of maginutde around 1
            #U = U/U.sum().detach()*multiplier
            #V = V/V.sum().detach()*multiplier

            errors = np.log10(compat_error_superintegrable(U_exact, V_exact, U, V))

            average_error = np.median(errors)
            print("Median square L error for ", name, " is: ", average_error)

            try:
                plt.hist(errors, bins=100, label=name, alpha=0.6)
            except:
                pass
            #plt.plot(errors, label=name)
            if args.export:
                add_log(file_name+" "+name, average_error)
       
    elif args.model == "HT":
        raise Exception("Not implemented")

    elif args.model in ["P3D", "K3D"]:
        for data_frame, name in zip(frames, titles):
            if args.model == "P3D":
                file_name += "_P3D"
            elif args.model == "K3D":
                file_name += "_K3D"
            if name == "Learned implicit": #not implemented
                print("Implicit not yet implemented, skipping.")
                continue

            Ls = load_normalized_Ls(data_frame, 6)
            J = np.array([
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0, 0.0, 0.0]])

            errors = np.log10([np.linalg.norm(np.matmul(Ls[i], J) - np.matmul(J, Ls[i])) for i in range(len(Ls))])
            #errors = [np.linalg.norm(Ls[i]-J) for i in range(len(Ls))]
            #iterations = np.linspace(1,len(errors),len(errors))

            average_error = np.median(errors)
            print("Median log10 norm L error for ", name, " is: ", average_error)

            try:
                plt.hist(errors, bins=100, label=name, alpha=0.6)
            except:
                pass
            #add_plot(plt, iterations, errors, name=name+": L_error")
            file_name += "-"+name
            if args.export:
                add_log(file_name+" "+name, average_error)
    else:
        raise Exception("Unknown model")

    plt.xlabel("compatibility error (log_10)")
    #plt.ylabel("Compatibility error")
    plt.legend()
    if args.export:
        name = file_name
        file_name = args.folder_name+"/"+file_name+".png"
        print("Exporting figure to: "+file_name)
        plt.savefig(file_name) 
    plt.show()

if args.plot_L_errors:
    file_name = "L_errors"
    for data_frame, name in zip(frames, titles):
        if args.model == "RB":
            file_name += "_RB"

            Ls = load_normalized_Ls(data_frame, 3)
            mxs = data_frame["mx"]
            mys = data_frame["my"]
            mzs = data_frame["mz"]

            Ls_exact = np.array([
                [[0.0, -mzs[i], mys[i]],
                [0.0, 0.0, -mxs[i]],
                [0.0, 0.0, 0.0]]
                for i in range(len(mxs))])
            #Ls_exact = np.array([(Ls_exact[i]-np.transpose(Ls[i])) for i in range(len(Ls))])
            Ls_exact -= np.transpose(Ls_exact, (0,2,1))
            #print("Lex pred: ", Ls_exact[0])
            Ls_exact = normalize(Ls_exact)
            #print("Lex po: ", Ls_exact[0])

            total_error = [np.linalg.norm(Ls[i]-Ls_exact[i]) for i in range(len(Ls))]
            iterations = np.linspace(1,len(total_error),len(total_error))

            average_error = (np.trapz(total_error)+0.0)/len(total_error)
            print("Average L error for ", name, " is: ", average_error)
            if args.export:
                add_log(file_name+" "+name, average_error)

            #plt.ylim(top=10)
            #plt.ylim(bottom=-10)
            add_plot(plt, iterations, total_error, name=name+": L_error")
            #add_plot(plt, data_frame["time"], error12, name=name+"_error12")
            #add_plot(plt, data_frame["time"], error23, name=name+"_error23")
            file_name += "-"+name
        elif args.model == "HT":
            file_name += "_HT"
            if name == "Learned implicit": #not implemented
                print("Implicit not yet implemented, skipping.")
                continue
            mxs = data_frame["mx"]
            mys = data_frame["my"]
            mzs = data_frame["mz"]
            rxs = data_frame["rx"]
            rys = data_frame["ry"]
            rzs = data_frame["rz"]

            Ls = load_normalized_Ls(data_frame, 6)

            Ls_exact = np.array([
                [[0.0, -mzs[i], mys[i], 0.0, -rzs[i], rys[i]],
                [mzs[i], 0.0, -mxs[i], rzs[i], 0.0, -rxs[i]],
                [-mys[i], mxs[i], 0.0, -rys[i], rxs[i], 0.0],
                [0.0, -rzs[i], rys[i], 0.0, 0.0, 0.0],
                [rzs[i], 0.0, -rxs[i], 0.0, 0.0, 0.0],
                [-rys[i], rxs[i], 0.0, 0.0, 0.0, 0.0]]
                for i in range(len(mxs))])
            Ls_exact = normalize(Ls_exact)

            total_error = [np.linalg.norm(Ls[i]-Ls_exact[i]) for i in range(len(Ls))]
            iterations = np.linspace(1,len(total_error),len(total_error))

            average_error = (np.trapz(total_error)+0.0)/len(total_error)
            print("Average L error for ", name, " is: ", average_error)
            if args.export:
                add_log(file_name+" "+name, average_error)

            #plt.ylim(top=10)
            #plt.ylim(bottom=-10)
            add_plot(plt, iterations, total_error, name=name+": L_error")
            file_name += "-"+name
            #add_plot(plt, data_frame["time"], error12, name=name+"_error12")
            #add_plot(plt, data_frame["time"], error23, name=name+"_error23")
        elif args.model in ["P3D", "K3D"]:
            if args.model == "P3D":
                file_name += "_P3D"
            elif args.model == "K3D":
                file_name += "_K3D"
            if name == "Learned implicit": #not implemented
                print("Implicit not yet implemented, skipping.")
                continue

            Ls = load_normalized_Ls(data_frame, 6)
            J = np.array([
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0, 0.0, 0.0]])

            errors = [np.linalg.norm(np.matmul(Ls[i], J) - np.matmul(J, Ls[i])) for i in range(len(Ls))]
            #errors = [np.linalg.norm(Ls[i]-J) for i in range(len(Ls))]
            iterations = np.linspace(1,len(errors),len(errors))

            average_error = (np.trapz(errors)+0.0)/len(errors)
            print("Average L error for ", name, " is: ", average_error)

            #plt.ylim(top=10)
            #plt.ylim(bottom=-10)
            add_plot(plt, iterations, errors, name=name+": L_error")
            file_name += "-"+name
            if args.export:
                add_log(file_name+" "+name, average_error)
            #add_plot(plt, data_frame["time"], error12, name=name+"_error12")
            #add_plot(plt, data_frame["time"], error23, name=name+"_error23")


    plt.legend()
    if args.export:
        name = file_name
        file_name = args.folder_name+"/"+file_name+".png"
        print("Exporting figure to: "+file_name)
        plt.savefig(file_name) 
    plt.show()

if args.plot_spectrum_errors:
    file_name = "σ_errors"
    for data_frame, name in zip(frames, titles):
        if args.model == "RB":
            file_name += "_RB"

            Ls = load_normalized_Ls(data_frame, 3)
            σs = [np.sort_complex(1.j*np.linalg.eig(Ls[i])[0]) for i in range(len(Ls))]

            mxs = data_frame["mx"]
            mys = data_frame["my"]
            mzs = data_frame["mz"]

            Ls_exact = np.array([
                [[0.0, -mzs[i], mys[i]],
                [0.0, 0.0, -mxs[i]],
                [0.0, 0.0, 0.0]]
                for i in range(len(mxs))])
            #Ls_exact = np.array([(Ls_exact[i]-np.transpose(Ls[i])) for i in range(len(Ls))])
            Ls_exact -= np.transpose(Ls_exact, (0,2,1))
            Ls_exact = normalize(Ls_exact)
            σs_exact = [np.sort_complex(1.j*np.linalg.eig(Ls_exact[i])[0]) for i in range(len(Ls))]

            total_error = [np.linalg.norm(σs[i]-σs_exact[i]) for i in range(len(Ls))]
            iterations = np.linspace(1,len(total_error),len(total_error))

            average_error = (np.trapz(total_error)+0.0)/len(total_error)
            print("Average σ error for ", name, " is: ", average_error)
            if args.export:
                add_log(file_name+" "+name, average_error)

            #plt.ylim(top=10)
            #plt.ylim(bottom=-10)
            add_plot(plt, iterations, total_error, name=name+": σ_error")
            #add_plot(plt, data_frame["time"], error12, name=name+"_error12")
            #add_plot(plt, data_frame["time"], error23, name=name+"_error23")
            file_name += "-"+name
        elif args.model == "HT":
            file_name += "_HT"
            if name == "Learned implicit": #not implemented
                print("Implicit not yet implemented, skipping.")
                continue
            
            mxs = data_frame["mx"]
            mys = data_frame["my"]
            mzs = data_frame["mz"]
            rxs = data_frame["rx"]
            rys = data_frame["ry"]
            rzs = data_frame["rz"]

            Ls = load_normalized_Ls(data_frame, 6)
            σs = [np.sort_complex(1.j*np.linalg.eig(Ls[i])[0]) for i in range(len(Ls))]

            Ls_exact = np.array([
                [[0.0, -mzs[i], mys[i], 0.0, -rzs[i], rys[i]],
                [mzs[i], 0.0, -mxs[i], rzs[i], 0.0, -rxs[i]],
                [-mys[i], mxs[i], 0.0, -rys[i], rxs[i], 0.0],
                [0.0, -rzs[i], rys[i], 0.0, 0.0, 0.0],
                [rzs[i], 0.0, -rxs[i], 0.0, 0.0, 0.0],
                [-rys[i], rxs[i], 0.0, 0.0, 0.0, 0.0]]
                for i in range(len(mxs))])
            #Ls_exact = np.array([(Ls[i]-np.transpose(Ls[i])) for i in range(len(Ls))])
            #Ls_exact -= np.transpose(Ls_exact, (0,2,1))
            Ls_exact = normalize(Ls_exact)
            σs_exact = [np.sort_complex(1.j*np.linalg.eig(Ls_exact[i])[0]) for i in range(len(Ls))]

            total_error = [np.linalg.norm(σs[i]-σs_exact[i]) for i in range(len(Ls))]
            iterations = np.linspace(1,len(total_error),len(total_error))

            average_error = (np.trapz(total_error)+0.0)/len(total_error)
            print("Average σ error for ", name, " is: ", average_error)
            if args.export:
                add_log(file_name+" "+name, average_error)

            #plt.ylim(top=10)
            #plt.ylim(bottom=-10)
            add_plot(plt, iterations, total_error, name=name+": σ_error")
            file_name += "-"+name
            #add_plot(plt, data_frame["time"], error12, name=name+"_error12")
            #add_plot(plt, data_frame["time"], error23, name=name+"_error23")
        elif args.model in ["P3D", "K3D"]:
            if args.model == "P3D":
                file_name += "_P3D"
            elif args.model == "K3D":
                file_name += "_K3D"
            if name == "Learned implicit": #not implemented
                print("Implicit not yet implemented, skipping.")
                continue

            Ls = load_normalized_Ls(data_frame, 6)
            σs = [np.sort_complex(1.j*np.linalg.eig(Ls[i])[0]) for i in range(len(Ls))]

            Ls_exact = np.array([
                [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0, 0.0, 0.0]]
                for i in range(len(Ls))])
            Ls_exact = normalize(Ls_exact)
            σs_exact = [np.sort_complex(1.j*np.linalg.eig(Ls_exact[i])[0]) for i in range(len(Ls))]

            errors = [np.linalg.norm(σs[i]-σs_exact[i]) for i in range(len(Ls))]
            iterations = np.linspace(1,len(errors),len(errors))

            average_error = (np.trapz(errors)+0.0)/len(errors)
            print("Average σ error for ", name, " is: ", average_error)

            #plt.ylim(top=10)
            #plt.ylim(bottom=-10)
            add_plot(plt, iterations, errors, name=name+": σ_error")
            file_name += "-"+name
            if args.export:
                add_log(file_name+" "+name, average_error)
            #add_plot(plt, data_frame["time"], error12, name=name+"_error12")
            #add_plot(plt, data_frame["time"], error23, name=name+"_error23")
    plt.legend()
    if args.export:
        name = file_name
        file_name = args.folder_name+"/"+file_name+".png"
        print("Exporting figure to: "+file_name)
        plt.savefig(file_name) 
    plt.show()

if args.plot_det_log:
    file_name = "det_log"+args.model
    for data_frame, name in zip(frames, titles):
        print(name)
        if args.model == "RB":
            print("det of skew-symmetric odd-dimensional matrix is always zero.")
            continue
        elif args.model in ["P2D", "Sh"]:
            if name == "Learned implicit": #not implemented
                print("Skipping det of L for the implicit (not implemented).")
                continue
            Ls = load_normalized_Ls(data_frame, 4)
        elif args.model in ["HT", "P3D", "K3D"]:
            if name == "Learned implicit": #not implemented
                print("Skipping det of L for the implicit (not implemented).")
                continue
            Ls = load_normalized_Ls(data_frame, 6)
        
        dets = np.log10(np.abs(np.linalg.det(Ls)))

        average_det = np.median(dets)
        print("Median Log_10(|det L|) for ", name, " is: ", average_det)
        if args.export:
            add_log(file_name+" "+name, average_det)
        if name != "GT":
            try:
                plt.hist(dets, bins=100, label=name, alpha=0.6)
            except:
                pass

    plt.legend()
    plt.xlabel("log_10(det(L))")
    if args.export:
        name = file_name
        file_name = args.folder_name+"/"+file_name+".png"
        print("Exporting figure to: "+file_name)
        plt.savefig(file_name) 
    plt.show()

if args.plot_det:
    file_name = "det_"+args.model
    for data_frame, name in zip(frames, titles):
        print(name)
        if args.model == "RB":
            print("det of skew-symmetric odd-dimensional matrix is always zero.")
            continue
        elif args.model in ["P2D", "Sh"]:
            if name == "Learned implicit": #not implemented
                print("Skipping det of L for the implicit (not implemented).")
                continue
            Ls = load_normalized_Ls(data_frame, 4)
        elif args.model in ["HT", "P3D", "K3D"]:
            if name == "Learned implicit": #not implemented
                print("Skipping det of L for the implicit (not implemented).")
                continue
            Ls = load_normalized_Ls(data_frame, 6)
        
        #print(Ls[0])
        #print("det = ", np.linalg.det(Ls[0]))
        #if name != "GT":
        #    dets = reject_outliers(np.linalg.det(Ls), m=2)
        #    if len(dets < len(Ls)):
        #        print("Rejected outliers number = ", len(Ls)-len(dets))
        #else:
        #    dets = np.linalg.det(Ls)
        dets = np.linalg.det(Ls)

        total_dets= dets
        #iterations = np.linspace(1,len(total_dets),len(total_dets))
        average_error = np.median(total_dets)
        #add_plot(plt, iterations, total_dets, name=name+": det", split=False)
        if name != "GT":
            try:
                plt.hist(dets, bins=100, label=name, alpha=0.6)
            except:
                pass

        average_det = np.mean(dets)
        average_median = np.median(dets)
        std = np.std(dets)
        print("Average det L for ", name, " is: ", average_det)
        print("Median det L for ", name, " is: ", average_median)
        print("Standard deviation of det L for ", name, " is: ", std)
        if args.export:
            add_log(file_name+" "+name, average_error)
            add_log(file_name+" std "+name, std)
            add_log(file_name+" median "+name, average_median)

        #plt.ylim(top=10)
        #plt.ylim(bottom=-10)
        #add_plot(plt, iterations, dets, name=name+": det(L)")
        file_name += "-"+name
    plt.legend()
    plt.xlabel("det(L)")
    if args.export:
        name = file_name
        file_name = args.folder_name+"/"+file_name+".png"
        print("Exporting figure to: "+file_name)
        plt.savefig(file_name) 
    plt.show()


def plot_fields_errors(fields, field_name = ""):
    print("Plotting errors of fields: ", fields)
    file_name = args.model+"_"+field_name+"-errors"
    for data_frame, name in zip(frames, titles):
        values = {}
        gt = {}
        for field in fields:
            values[field] = data_frame[field]
            gt[field] = dfgt[field]

        total_error = np.sum(np.square([values[field]-gt[field] for field in fields]), axis=0)
        #iterations = np.linspace(1,len(total_error),len(total_error))

        average_error = np.median(total_error)
        print("Median "+field_name+" error for ", name, " is: ", average_error)
        file_name += "-"+name
        if args.export:
            add_log(file_name+" "+field_name, average_error)
        #add_plot(plt, iterations, total_error, name=name+": "+name+" error")
        try:
            plt.hist(np.log10(total_error), bins=100, label=name, alpha=0.6)
        except:
            pass
        #add_plot(plt, data_frame["time"], error12, name=name+"_error12")
        #add_plot(plt, data_frame["time"], error23, name=name+"_error23")

    plt.legend()
    plt.xlabel("Trajectory errors (log_10): "+field_name)
    if args.export:
        name = file_name
        file_name = args.folder_name+"/"+file_name+".png"
        print("Exporting figure to: "+file_name)
        plt.savefig(file_name) 
    plt.show()

if args.plot_RB_errors:
    plot_fields_errors(["mx", "my", "mz"], field_name = "m")

if args.plot_Sh_errors:
    plot_fields_errors(["u", "x", "y", "z"], field_name = "uxyz")

if args.plot_RB_errors:
    plot_fields_errors(["mx", "my", "mz"], field_name = "m")

if args.plot_P2D_errors:
    plot_fields_errors(["mx", "my"], field_name = "m")
    plot_fields_errors(["rx", "ry"], field_name = "r")

if args.plot_P3D_errors:
    plot_fields_errors(["mx", "my", "mz"], field_name = "m")
    plot_fields_errors(["rx", "ry", "rz"], field_name = "r")

if args.plot_HT_errors:
    plot_fields_errors(["mx", "my", "mz"], field_name = "m")
    plot_fields_errors(["rx", "ry", "rz"], field_name = "r")

if args.plot_msq_errors:
    file_name = args.model+"_msq-errors"
    for data_frame, name in zip(frames,titles):
        mxs = split_to_forward_paths(data_frame, "mx")
        mys = split_to_forward_paths(data_frame, "my")
        if args.model in ["HT", "P3D"]:
            mzs = split_to_forward_paths(data_frame, "mz")

        msq_variations = []
        for i in range(len(mxs)):
            if args.model in ["HT", "P3D"]:
                total_msq = np.square(mxs[i]) + np.square(mys[i]) + np.square(mzs[i])
                #mean_msq = np.mean(total_msq) * np.ones(len(total_msq))
                #msq_variations.append(np.mean(np.abs(total_msq-mean_msq)))
                msq_variations.append(np.std(total_msq)/len(mxs[i]))
            
        iterations = np.linspace(1,len(mxs),len(mxs))

        avg = np.median(msq_variations)
        print("Std msq for ", name, " is: ", avg)
        if args.export:
            add_log(file_name+" "+name, avg)

        plt.xlabel("paths")

        try:
            plt.hist(np.log10(msq_variations), bins=100, label=name, alpha=0.6)
        except:
            pass
        #add_plot(plt, iterations, msq_variations, name=name+": msq", apply_filter=False, split=False)
        file_name += "-"+name
        #add_plot(plt, data_frame["time"], error12, name=name+"_error12")
        #add_plot(plt, data_frame["time"], error23, name=name+"_error23")

    plt.legend()
    if args.export:
        name = file_name
        file_name = args.folder_name+"/"+file_name+".png"
        print("Exporting figure to: "+file_name)
        plt.savefig(file_name) 
    plt.show()

if args.plot_rsq_errors:
    #for data_frame, name in zip([dfl, dfls, dflw], ["Learned implicit", "Learned Soft", "Learned Without"]):
    file_name = args.model+"_rsq-errors"
    for data_frame, name in zip(frames,titles):
        rxs = split_to_forward_paths(data_frame, "rx")
        rys = split_to_forward_paths(data_frame, "ry")
        rzs = split_to_forward_paths(data_frame, "rz")
        if len(rxs) != len(rys) or len(rys)!= len(rzs) or len(rzs)!=len(rxs):
            raise Exception("Paths of different lengths found.")

        rsq_variations = []
        for i in range(len(rxs)):
            total_rsq = np.square(rxs[i]) + np.square(rys[i]) + np.square(rzs[i])
            mean_rsq = np.mean(total_rsq) * np.ones(len(total_rsq))
            rsq_variations.append(sqrt(np.mean(np.square(total_rsq-mean_rsq)))/len(rxs[i]))
            
        iterations = np.linspace(1,len(rxs),len(rxs))

        avg = np.median(rsq_variations)
        print("Median rsq variation for ", name, " is: ", avg)
        if args.export:
            add_log(file_name+" "+name, avg)

        plt.xlabel("log_10(r^2 variations)")
        try:
            plt.hist(np.log10(rsq_variations), bins=100, label=name, alpha=0.6)
        except:
            pass
        #add_plot(plt, iterations, rsq_variations, name=name+": rsq", apply_filter=False, split=False)
        file_name += "-"+name

    plt.legend()
    if args.export:
        name = file_name
        file_name = args.folder_name+"/"+file_name+".png"
        print("Exporting figure to: "+file_name)
        plt.savefig(file_name) 
    plt.show()

if args.plot_mrs_errors:
    file_name = args.model+"_mrs-errors"
    for data_frame, name in zip(frames, titles):
        mxs = split_to_forward_paths(data_frame, "mx")
        mys = split_to_forward_paths(data_frame, "my")
        mzs = split_to_forward_paths(data_frame, "mz")
        rxs = split_to_forward_paths(data_frame, "rx")
        rys = split_to_forward_paths(data_frame, "ry")
        rzs = split_to_forward_paths(data_frame, "rz")
        if len(rxs) != len(rys) or len(rys)!= len(rzs) or len(rzs)!=len(rxs):
            raise Exception("Paths of different lengths found.")

        mr_variations = []
        for i in range(len(rxs)):
            total_mr = np.multiply(mxs[i],rxs[i]) + np.multiply(mys[i],rys[i]) + np.multiply(mzs[i],rzs[i])
            mean_mr = np.mean(total_mr) * np.ones(len(total_mr))
            mr_variations.append(sqrt(np.mean(np.square(total_mr-mean_mr)))/len(rxs[i]))
            
        iterations = np.linspace(1,len(rxs),len(rxs))

        avg = np.median(mr_variations)
        print("Median m.r variation for ", name, " is: ", avg)
        if args.export:
            add_log(file_name+" "+name, avg)

        plt.xlabel("log_10(m.r variations)")
        try:
            plt.hist(np.log10(mr_variations), bins=100, label=name, alpha=0.6)
        except:
            pass
        #add_plot(plt, iterations, mr_variations, name=name+": mr", apply_filter=False, split=False)
        file_name += "-"+name

    plt.legend()
    if args.export:
        name = file_name
        file_name = args.folder_name+"/"+file_name+".png"
        print("Exporting figure to: "+file_name)
        plt.savefig(file_name) 
    plt.show()

if args.plot_am_errors: #angular momentum
    file_name = args.model+"_am-errors"
    for data_frame, name in zip(frames, titles):
        if args.model == "P2D":
            mxs = split_to_forward_paths(data_frame, "mx")
            mys = split_to_forward_paths(data_frame, "my")
            ms = [[[mxs[i][j], mys[i][j]] for j in range(len(mxs[i]))] for i in range(len(mxs))]
            rxs = split_to_forward_paths(data_frame, "rx")
            rys = split_to_forward_paths(data_frame, "ry")
            rs = [[[rxs[i][j], rys[i][j]] for j in range(len(rxs[i]))] for i in range(len(rxs))]
            if len(rxs) != len(rys):
                raise Exception("Paths of different lengths found.")
        elif args.model == "P3D":
            mxs = split_to_forward_paths(data_frame, "mx")
            mys = split_to_forward_paths(data_frame, "my")
            mzs = split_to_forward_paths(data_frame, "mz")
            ms = [[[mxs[i][j], mys[i][j], mzs[i][j]] for j in range(len(mxs[i]))] for i in range(len(mxs))]
            rxs = split_to_forward_paths(data_frame, "rx")
            rys = split_to_forward_paths(data_frame, "ry")
            rzs = split_to_forward_paths(data_frame, "rz")
            rs = [[[rxs[i][j], rys[i][j], rzs[i][j]] for j in range(len(rxs[i]))] for i in range(len(rxs))]
            if len(rxs) != len(rys) or len(rys)!= len(rzs) or len(rzs)!=len(rxs):
                raise Exception("Paths of different lengths found.")

        am_variations = []
        for i in range(len(rxs)): #over trajectories
            total_am = np.cross(rs[i], ms[i])
            mean_am = np.mean(total_am, axis=0)
            diffs = [np.linalg.norm(total_am[j]-mean_am) for j in range(len(total_am))]
            am_variations.append(np.linalg.norm(diffs)/len(rxs[i]))
            
        iterations = np.linspace(1,len(rxs),len(rxs))

        avg = np.median(am_variations)
        print("Median r x p variation for ", name, " is: ", avg)
        if args.export:
            add_log(file_name+" "+name, avg)

        plt.xlabel("paths")
        try:
            plt.hist(np.log10(am_variations), bins=100, label=name, alpha=0.6)
        except:
            pass
        #add_plot(plt, iterations, am_variations, name=name+": am", apply_filter=False, split=False)
        file_name += "-"+name

    plt.legend()
    if args.export:
        name = file_name
        file_name = args.folder_name+"/"+file_name+".png"
        print("Exporting figure to: "+file_name)
        plt.savefig(file_name) 
    plt.show()

if args.plot_Casimir:
    for data_frame, name in zip(frames,titles):
        if name != "Learned implicit":
            continue #not yet implemented
        casss = data_frame["cass"]

        iterations = np.linspace(1,len(casss),len(casss))

        add_plot(plt, iterations, casss, name=name+"_Cas")
        #add_plot(plt, data_frame["time"], error12, name=name+"_error12")
        #add_plot(plt, data_frame["time"], error23, name=name+"_error23")

    plt.legend()
    plt.show()

def plot_first(field):
    x_gt = split_to_forward_paths(dfgt, field)[1]
    times = dfgt["time"][:len(x_gt)]
    if args.soft:
        add_plot(plt, times, dfls[field][:len(times)], name=args.model +" soft: "+field )
    if args.implicit:
        add_plot(plt, times, dfli[field][:len(times)], name=args.model +" implicit: "+field )
    if args.without:
        add_plot(plt, times, dflw[field][:len(times)], name=args.model+" without: "+field )
    if args.GT:
        add_plot(plt, times, dfgt[field][:len(times)], name=args.model+" GT: "+field )
    plt.legend()
    if args.export:
        file_name = args.folder_name+"/"+args.model+"_first_"+field+".png"
        print("Exporting figure to: "+file_name)
        plt.savefig(file_name) 
    plt.show()

if args.plot_first != "None":
    plot_first(args.plot_first)

if args.plot_first_mx:
    plot_first("mx")

if args.plot_first_rx:
    plot_first("rx")

if args.plot_Jacobi:
    if args.soft:
        df_errors = pd.read_csv(args.folder_name+"/data/errors_soft.csv")
        validation_reg_errors = df_errors["val_reg"]
        times = np.linspace(1, len(validation_reg_errors[1:]), len(validation_reg_errors[1:]))
        add_plot(plt, times, validation_reg_errors[1:], name="soft: Jacobi", split=False)
    #if args.implicit:
    #    df_errors = pd.read_csv(args.folder_name+"/data/errors_implicit.csv")
    #    validation_reg_errors = df_errors["val_reg"]
    #    times = np.linspace(1, len(validation_reg_errors[1:]), len(validation_reg_errors[1:]))
    #    print(times)
    #    print(validation_reg_errors[1:])
    #    add_plot(plt, times, validation_reg_errors[1:], name="implicit: Jacobi", split=False)
    if args.without:
        df_errors = pd.read_csv(args.folder_name+"/data/errors_without.csv")
        validation_reg_errors = df_errors["val_reg"]
        times = np.linspace(1, len(validation_reg_errors[1:]), len(validation_reg_errors[1:]))
        add_plot(plt, times, validation_reg_errors[1:], name="without: Jacobi", split=False)
    plt.legend()
    if args.export:
        file_name = args.folder_name+"/jacobi.png"
        print("Exporting figure to: "+file_name)
        plt.savefig(file_name) 
    plt.show()

if args.plot_training_errors:
    comparison.plot_training_errors(args)

def generate_E_points(args, energy):
    #generates random initial conditions (uniformly on the ball with radius as in the original args)
    sqm = torch.tensor(args.init_mx**2 + args.init_my**2 + args.init_mz**2) #square magnitude of m
    mmag = torch.sqrt(sqm) #magnitude of m
    mx = torch.linspace(0, mmag, args.density)
    my = torch.linspace(0, mmag, args.density)
    mx_mesh, my_mesh = torch.meshgrid((mx, my)) 
    mxsq_mesh, mysq_mesh = mx_mesh**2, my_mesh**2
    mzsq_mesh = sqm - mxsq_mesh - mysq_mesh
    mz_mesh = torch.sqrt(mzsq_mesh)

    if args.model in ["HT", "P3D", "K3D"]:
        sqr = torch.tensor(args.init_rx**2 + args.init_ry**2 + args.init_rz**2) #square magnitude of m
        rmag = torch.sqrt(sqr) #magnitude of m
        rx = torch.linspace(0, rmag, args.density)
        ry = torch.linspace(0, rmag, args.density)
        rx_mesh, ry_mesh = torch.meshgrid((rx, ry)) 
        rxsq_mesh, rysq_mesh = rx_mesh**2, ry_mesh**2
        rzsq_mesh = sqr - rxsq_mesh - rysq_mesh
        rz_mesh = torch.sqrt(rzsq_mesh)
        total_m = torch.stack((
            mx_mesh.reshape(-1,), 
            my_mesh.reshape(-1,), 
            mz_mesh.reshape(-1,), 
            rx_mesh.reshape(-1,), 
            ry_mesh.reshape(-1,), 
            rz_mesh.reshape(-1,)), axis=1)
    else: #RB
        total_m = torch.stack((
            mx_mesh.reshape(-1,), 
            my_mesh.reshape(-1,), 
            mz_mesh.reshape(-1,)), axis=1)

    E = energy(total_m)
    E_for_plot = E.detach().reshape((args.density, args.density))
    mx_for_plot = mx_mesh.reshape((args.density, args.density))
    my_for_plot = my_mesh.reshape((args.density, args.density))
    mz_for_plot = mz_mesh.reshape((args.density, args.density))
    if args.model in ["HT", "P3D", "K3D"]:
        rx_for_plot = rx_mesh.reshape((args.density, args.density))
        ry_for_plot = ry_mesh.reshape((args.density, args.density))
        rz_for_plot = rz_mesh.reshape((args.density, args.density))
        return mx_for_plot, my_for_plot, mz_for_plot, rx_for_plot, ry_for_plot, rz_for_plot, E_for_plot
    else: #RB
        return mx_for_plot, my_for_plot, mz_for_plot, E_for_plot
    
    
def generate_L_points(args, L_tensor):
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
 

    if args.model in ["HT", "P3D", "K3D"]:
        sqr = torch.tensor(args.init_rx**2 + args.init_ry**2 + args.init_rz**2) #square magnitude of m
        rmag = torch.sqrt(sqr) #magnitude of m
        rx = torch.linspace(0, rmag, args.density)
        ry = torch.linspace(0, rmag, args.density)
        rx_mesh, ry_mesh = torch.meshgrid((rx, ry)) 
        rxsq_mesh, rysq_mesh = rx_mesh**2, ry_mesh**2
        rzsq_mesh = sqr - rxsq_mesh - rysq_mesh
        rz_mesh = torch.sqrt(rzsq_mesh)
        total_m = torch.stack((
            mx_mesh.reshape(-1,), 
            my_mesh.reshape(-1,), 
            mz_mesh.reshape(-1,), 
            rx_mesh.reshape(-1,), 
            ry_mesh.reshape(-1,), 
            rz_mesh.reshape(-1,)), axis=1)
    else: #RB
        total_m = torch.stack((
            mx_mesh.reshape(-1,), 
            my_mesh.reshape(-1,), 
            mz_mesh.reshape(-1,)), axis=1)
  
    total_m_batched = torch.split(total_m, 20)

    L_batched = [L_tensor(b) for b in total_m_batched]
    
    L = torch.cat(L_batched, axis=0)
    if args.model == "RB":
        return mx_mesh, my_mesh, mz_mesh, L
    else:
        return mx_mesh, my_mesh, mz_mesh, rx_mesh, ry_mesh, rz_mesh, L

if args.plot_Es:
    models = get_learned_models()
    for model in models:
        energy = models[model]["energy"]
        if args.model == "RB":
            mx, my, mz, E = generate_E_points(args, energy)
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            surf = ax.plot_surface(mx, my, E, cmap=cm.coolwarm,
                                linewidth=0, antialiased=False)
            ax.set_xlabel("mx")
            ax.set_ylabel("my")
            ax.set_zlabel("Energy")
            plt.show()
        elif args.model == "HT":
            mx, my, mz, rx, ry, rz, E = generate_E_points(args, energy)
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            surf = ax.plot_surface(mx, my, E, cmap=cm.coolwarm,
                                linewidth=0, antialiased=False)
            ax.set_xlabel("mx")
            ax.set_ylabel("my")
            ax.set_zlabel("Energy")
            plt.show()
        else:
            raise Exception("Model not implemented.")

if args.plot_Ls:
    models = get_learned_models()
    for model in models:
        L_tensor = models[model]["L"]
        if args.model == "RB":
            mx, my, mz, L = generate_L_points(args, L_tensor)
            plt.title(model)
            plt.scatter(mx.detach().reshape(-1,), L.detach()[:,1,2], label="L23 vs mx")
            plt.scatter(my.detach().reshape(-1,), L.detach()[:,0,2], label="L13 vs my")
            plt.scatter(mz.detach().reshape(-1,), L.detach()[:,0,1], label="L12 vs mz")
            plt.legend()
            plt.show()

            #derivatives of L wrt m
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
            plt.title(model)
            plt.legend()
            plt.show()

            plt.scatter(mx.detach().reshape(-1,), dL12dmx.detach().reshape(-1), label="DL23/Dmx")
            plt.scatter(my.detach().reshape(-1,), dL12dmy.detach().reshape(-1,), label="DL23/Dmy")
            plt.scatter(mz.detach().reshape(-1,), dL12dmz.detach().reshape(-1,), label="DL23/Dmz")
            plt.title(model)
            plt.legend()
            plt.show()

            plt.scatter(mx.detach().reshape(-1,), dL02dmx.detach().reshape(-1), label="DL13/Dmx")
            plt.scatter(my.detach().reshape(-1,), dL02dmy.detach().reshape(-1,), label="DL13/Dmy")
            plt.scatter(mz.detach().reshape(-1,), dL02dmz.detach().reshape(-1,), label="DL13/Dmz")
            plt.title(model)
            plt.legend()
            plt.show()

            plt.scatter(mx.detach().reshape(-1,), dL01dmx.detach().reshape(-1), label="DL12/Dmx")
            plt.scatter(my.detach().reshape(-1,), dL01dmy.detach().reshape(-1,), label="DL12/Dmy")
            plt.scatter(mz.detach().reshape(-1,), dL01dmz.detach().reshape(-1,), label="DL12/Dmz")
            plt.title(model)
            plt.legend()
            plt.show()
        elif args.model == "HT":
            mx, my, mz, rx, ry, rz, L = generate_L_points(args, L_tensor)
            plt.scatter(mx.detach().reshape(-1,), L.detach()[:,1,2], label="L23 vs mx")
            plt.scatter(my.detach().reshape(-1,), L.detach()[:,0,2], label="L13 vs my")
            plt.scatter(mz.detach().reshape(-1,), L.detach()[:,0,1], label="L12 vs mz")
            plt.scatter(rx.detach().reshape(-1,), L.detach()[:,1,5], label="L26 vs rx")
            plt.scatter(ry.detach().reshape(-1,), L.detach()[:,0,5], label="L16 vs ry")
            plt.scatter(rz.detach().reshape(-1,), L.detach()[:,0,4], label="L15 vs rz")
            plt.title(model)
            plt.legend()
            plt.show()
        else:
            raise Exception("Model not implemented.")


if args.plot_dataset:
    args.dataset = True
    args.without = False
    args.soft = False
    args.implicit = False
    args.GT = False

    frames, titles = get_frames_and_titles()
    if args.model == "HT":
        #plot_field(fields=["mx", "my", "mz", "rx", "ry", "rz", "E", "m.r", "r*r"], data_frames=[dfds], titles=["Dataset"])
        plot_field(fields=["mz", "rz", "E", "m.r", "r*r"])
    elif args.model == "P3D":
        plot_field(fields=["rx", "mx", "E", "sqm"])

