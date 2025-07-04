import torch
from torch import einsum

import numpy as np

import timeit


#Batched version of L that fulfills the Jacobi identity
def real_L(x):
    t = torch.zeros(x.shape[0], x.shape[1], x.shape[1])
    t[:,0,0] = 0
    t[:,0,1] = -x[:,2]
    t[:,0,2] = x[:,1]
    t[:,1,0] = x[:,2]
    t[:,1,1] = 0
    t[:,1,2] = -x[:,0]
    t[:,2,0] = -x[:,1]
    t[:,2,1] = x[:,0]
    t[:,2,2] = 0
    return t

#Unbatched version of L that fulfills the Jacobi identity
def real_L_unbatched(x):
    t = torch.zeros(x.shape[0],x.shape[0])
    t[0,0] = 0
    t[0,1] = -x[2]
    t[0,2] = x[1]
    t[1,0] = x[2]
    t[1,1] = 0
    t[1,2] = -x[0]
    t[2,0] = -x[1]
    t[2,1] = x[0]
    t[2,2] = 0
    return t

def Martin_Jacobi_vector(L_tensor, Lz, zn_tensor):
    #Lz_grad = torch.autograd.grad(torch.sum(Lz, axis=0), zn_tensor, create_graph=True)
    reduced_L = lambda z: torch.sum(L_tensor(z), axis=0)
    Lz_grad = torch.autograd.functional.jacobian(reduced_L, zn_tensor, create_graph=True).permute(2, 0, 1, 3)
    term1 = einsum('mkl,mijk->mijl', Lz, Lz_grad)
    term2 = einsum('mkj,mlik->mijl', Lz, Lz_grad)
    term3 = einsum('mki,mjlk->mijl', Lz, Lz_grad)
    jacobi_loss = (term1 + term2 + term3)
    return jacobi_loss

def Martin_Jacobi(L_tensor, Lz, zn_tensor):
    Lz_grad = torch.diagonal(torch.autograd.functional.jacobian(L_tensor, zn_tensor, create_graph=True), dim1=0, dim2=3).permute(3, 0, 1, 2)
    term1 = einsum('mkl,mijk->mijl', Lz, Lz_grad)
    term2 = einsum('mkj,mlik->mijl', Lz, Lz_grad)
    term3 = einsum('mki,mjlk->mijl', Lz, Lz_grad)
    jacobi_loss = (term1 + term2 + term3)
    return jacobi_loss

def Martin_Jacobi_fored(L_tensor, Lz, zn_tensor):
    Lz_grad = torch.zeros(zn_tensor.shape[0], zn_tensor.shape[1], zn_tensor.shape[1], zn_tensor.shape[1])
    for i in range(zn_tensor.shape[0]):
        Lz_grad[i] = torch.autograd.functional.jacobian(L_tensor, zn_tensor[i], create_graph=True)
    term1 = einsum('mkl,mijk->mijl', Lz, Lz_grad)
    term2 = einsum('mkj,mlik->mijl', Lz, Lz_grad)
    term3 = einsum('mki,mjlk->mijl', Lz, Lz_grad)
    jacobi_loss = (term1 + term2 + term3)
    return jacobi_loss

def Martin_Jacobi_tupled(L_tensor, Lz, zn_tensor):
    zn_splitted = torch.tensor_split(zn_tensor,zn_tensor.shape[0])
    print(zn_splitted)
    Lz_grad_splitted = torch.autograd.functional.jacobian(L_tensor, zn_splitted, create_graph=True)
    Lz_grad = torch.stack(Lz_grad_splitted)
    print(Lz_grad.shape)
    term1 = einsum('mkl,mijk->mijl', Lz, Lz_grad)
    term2 = einsum('mkj,mlik->mijl', Lz, Lz_grad)
    term3 = einsum('mki,mjlk->mijl', Lz, Lz_grad)
    jacobi_loss = (term1 + term2 + term3)
    return jacobi_loss

def Bea_Jacobi(L_net,z):
    L=L_net.reshape(-1)
    jac = torch.zeros(L.shape[0], z.shape[0]) 
    for i in range(L.shape[0]):
        grad_outputs = torch.zeros_like(L)
        grad_outputs[i] = 1
        j = torch.autograd.grad(L, [z], grad_outputs = grad_outputs, allow_unused=True, create_graph=True)[0]
        jac[i] = j if j is not None else 0
    
    #dimensiones de z=d x 1
    #la matriz jac tiene dimensiones n^2 x d
    K = torch.matmul(L_net.T,jac.T) #en la matriz K almacenamos las sumas en k

    n_l=L_net.shape[1]
    n_i=L_net.shape[0]
    n_j=L_net.shape[1]
    suma = []

    for l in range(n_l):

        for i in range(n_i):

            for j in range(n_j):

                sum = K[l,(i-1)*4+j]+K[j,(l-1)*4+i]+K[i,(j-1)*4+l]
                suma=np.append(suma,sum.detach()) #añado todos el nuevo elemento. Para eso tengo que "detach" el resultado
    
    loss = np.dot(suma, suma.T)  #al cuadrado para tener un escalar
    return loss

def time_Martin():
    m_sample = torch.rand(30, 3)
    m_sample.requires_grad = True
    L = real_L(m_sample)
    Martin_loss = Martin_Jacobi(real_L, L, m_sample)

    print("Result: ", (Martin_loss*Martin_loss).sum().detach().numpy())

def time_Martin_vector():
    m_sample = torch.rand(30, 3)
    m_sample.requires_grad = True
    L = real_L(m_sample)
    Martin_loss = Martin_Jacobi_vector(real_L, L, m_sample)

    print("Result: ", (Martin_loss*Martin_loss).sum().detach().numpy())

def time_Martin_fored():
    m_sample = torch.rand(30, 3)
    m_sample.requires_grad = True
    L = real_L(m_sample)
    Martin_loss = Martin_Jacobi_fored(real_L_unbatched, L, m_sample)
    print("Result: ", (Martin_loss*Martin_loss).sum().detach().numpy())

def time_Martin_tupled():
    m_sample = torch.rand(30, 3)
    m_sample.requires_grad = True
    L = real_L(m_sample)
    Martin_loss = Martin_Jacobi_tupled(real_L, L, m_sample)

    print("Result: ", (Martin_loss*Martin_loss).sum().detach().numpy())

def time_Bea():
    m_sample = torch.rand(30,3)
    m_sample.requires_grad = True
    for m in m_sample:
        L = real_L_unbatched(m)
        loss = Bea_Jacobi(L, m)
    print("Result: ", loss)

#############################################################
#Comment or uncomment run you would like to examine
#############################################################

#My standard code
execution_time = timeit.timeit(time_Martin, number=5)
print("Time to execute normal: ", execution_time)

#My optimal code
execution_time = timeit.timeit(time_Martin_vector, number=5)
print("Time to execute normal: ", execution_time)

#Your code
#execution_time = timeit.timeit(time_Bea, number=5)
#print("Time to execute Beas: ", execution_time)

