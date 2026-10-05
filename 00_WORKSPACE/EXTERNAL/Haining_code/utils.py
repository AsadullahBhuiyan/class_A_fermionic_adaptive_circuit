from functools import lru_cache
from itertools import permutations
import scipy.linalg as la
import numpy.linalg as nla
import numpy as np
import scipy.sparse as sp
from scipy.stats import special_ortho_group

def kraus(n):
    return np.array([[0,n[0],n[1],n[2]],
                    [-n[0],0,-n[2],n[1]],
                    [-n[1],n[2],0,-n[0]],
                    [-n[2],-n[1],n[0],0]])

def get_Born_tri_op(p,Gamma,rng=None,sigma=1,alpha=1):
    """sigma attenuate the variance of unitary, alpha suppress the strength of measurement"""
    num=Gamma.shape[0]
    rng=np.random.default_rng(rng)
    sign=rng.random(size=num)
    n1= (sign<p*(1+Gamma)/2)*(-1)+(sign>p*(1+Gamma)/2+1-p)
    n1 = n1* alpha
    n2,n3=get_inplane(n1, num,rng=rng,sigma=sigma)
    return np.c_[n1,n2,n3]

def get_inplane(n1,num,rng=None,sigma=1):
    r=np.sqrt(1-n1**2)
    rng=np.random.default_rng(rng)
    phi=rng.random(num)*2*np.pi*sigma
    n2,n3=r*np.cos(phi),r*np.sin(phi)
    return n2,n3
def get_O(rng,n):
    rng=np.random.default_rng(rng)
    A=rng.normal(size=(n,n))
    AA=(A-A.T)/2
    return la.expm(AA)
# def get_O(rng,n):
#     return special_ortho_group.rvs(dim=n,random_state=rng)
    
def P_contraction_2(Gamma,Upsilon,ix,ix_bar,Gamma_like=None,reset_Gamma_like=True):
    """ same analytical expression for contraction as _contraction(), differences:
    1. assume intput and output tensor have the same shape, thus, it should be Gamma(L,R) -- Upsilon (L,R), where Gamma_R = Upsilon_L =Upsilon_R, such that in-place operator is applied here.
    2. manually compute the inverse of mat2 before
    Here, Gamma is m, and proj is Upsilon
    Assume Upsilon = [A,B;C,D], the logic is to first compute C= (1+ Gamma_RR @ Upsilon_LL)^{-1}, (where then B=-C.T) 
    then A= Upsilon_LL @C, D= Gamma_RR@ C.T
    ---
    reset_Gamma_like: in usual case, because each application of the gate will be like the brick layer, therefore, ix_bar will overwrite the previous. However, for the staircase pattern, one should reset it. 
    """
    Gamma_RR=Gamma[np.ix_(ix,ix)]
    Gamma_LR=Gamma[np.ix_(ix_bar,ix)]
    Upsilon_LL=Upsilon[:len(ix),:len(ix)]
    Upsilon_RR=Upsilon[len(ix):,len(ix):]
    Upsilon_RL=Upsilon[len(ix):,:len(ix)]
    eye=np.eye(len(ix))
    try: 
        C=nla.inv(Gamma_RR@Upsilon_LL+eye)
    except:
        raise ValueError("the contraction will lead to a vanishing state")
    A=Upsilon_LL@C
    D=Gamma_RR@C.T
    tmp=Gamma_LR@A@Gamma_LR.T
    if Gamma_like is None:
        Gamma_like=np.zeros_like(Gamma)
    if reset_Gamma_like:
        Gamma_like.fill(0)
    Gamma_like[np.ix_(ix_bar,ix_bar)]=tmp
    Gamma+=Gamma_like
    Gamma[np.ix_(ix,ix_bar)]=Upsilon_RL@C@Gamma_LR.T
    Gamma[np.ix_(ix,ix)]=Upsilon_RR+Upsilon_RL@D@Upsilon_RL.T
    Gamma[np.ix_(ix_bar,ix)]=-Gamma[np.ix_(ix,ix_bar)].T
    # why is it neccessary?
    # Gamma-=Gamma.T
    # Gamma/=2

    if np.abs(contract(Gamma,[0,1],Gamma,[1,0],[0])+1).max()>1e-10:
        Gamma[:,:]=purify(Gamma)
        Gamma-=Gamma.T
        Gamma/=2

def purify(A):
    # purify A, see App. B2 in PhysRevB.106.134206
    val,vec=np.linalg.eigh(A/1j)
    val[val<0]=-1
    val[val>0]=1
    return -(vec@np.diag(val)@vec.conj().T).imag
def purify_O(A,A_D):
    O,_,_=block_diagonalize(A)
    return O.T@A_D@O


def block_diagonalize(A,thres=1e-10):
    '''A is an anti symmetry matrix for covariance matrix
    block diagonalize is to find a real othorgonal matrix such that OAO^T=A_D, where A_D=\oplus a_k \omega, where \omega = [0,1;-1,0]

    See : arxiv:0902.1502 App B for more details
    '''
    assert np.abs(A.imag).max()<1e-10, f'A is not a real matrix {np.abs(A.imag).max()}'
    A=A.real
    assert np.abs(A+A.T).max()<1e-10, f'A is not antisymmetric'

    val,vec=np.linalg.eigh(A/1j)
    val_arg=val.argsort()
    val=val[val_arg[:A.shape[0]//2]]
    vec=vec[:,val_arg[:A.shape[0]//2]]
    perm_list=permutations(range(vec.shape[1]))
    for perm in perm_list:
        vec=vec[:,perm]
        diag_element=np.array([vec[2*x,x] for x in range(vec.shape[1])])
        if np.all(diag_element !=0):
            break

    # diag_element=np.array([vec[2*x,x] for x in range(vec.shape[1])])
    phase_factor=diag_element.conj()/np.abs(diag_element)
    # phase_factor[np.isinf(phase_factor)|np.isnan(phase_factor)]=1
    vec=phase_factor.reshape((1,-1))*vec
    vec_conj=vec.conj()
    U=np.zeros(A.shape,dtype=complex)
    U[:,1::2]=vec_conj
    U[:,::2]=vec
    G=lambda l:np.kron(np.eye(l),np.array([[1,1],[-1j,1j]])/np.sqrt(2))
    O=G(A.shape[0]//2)@U.T.conj()
    assert np.abs(O.imag).max()<1e-10, f'O is not a real matrix {np.abs(O.imag).max()}'
    O=O.real
    A_D=O@A@O.T
    return O,U,A_D

def op_weak_onsite(A):
    assert -1<=A<=1, "A should be within [0,1]"
    Gamma=np.zeros((4,4),dtype=float)
    Gamma[0,1]=Gamma[3,2]=A
    Gamma[0,2]=Gamma[1,3]=np.sqrt(1-A**2)
    return (Gamma-Gamma.T)

def op_weak_nn_x(A):
    """exp(beta* (c_i^dag c_j + c_j^dag c_i)), x stands for the Pauli x like nearest neighbor coupling"""
    assert -1<=A<=1, "A should be within [0,1]"
    Gamma=np.zeros((8,8),dtype=float)
    Gamma[0,3]=Gamma[5,6]=A
    Gamma[1,2]=Gamma[4,7]=-A
    Gamma[0,4]=Gamma[1,5]=Gamma[2,6]=Gamma[3,7]=np.sqrt(1-A**2)
    return (Gamma-Gamma.T)

def op_weak_nn_y(A):
    """exp(beta* (-1j*c_i^dag c_j + 1j*c_j^dag c_i)), y stands for the Pauli y like nearest neighbor coupling"""
    assert -1<=A<=1, "A should be within [0,1]"
    Gamma=np.zeros((8,8),dtype=float)
    Gamma[0,2]=Gamma[1,3]=A
    Gamma[4,6]=Gamma[5,7]=-A
    Gamma[0,4]=Gamma[1,5]=Gamma[2,6]=Gamma[3,7]=np.sqrt(1-A**2)
    return (Gamma-Gamma.T)
@lru_cache(maxsize=None)
def op_single_mode(kind,sparse=False):
    mode, n = kind
    op= Gamma_othor(u=mode,epsilon11=np.array([[0,2*n-1],[1-2*n,0]]),epsilon12=np.zeros((2,2)))
    if sparse:
        return sp.csr_matrix(op)
    else:
        return op

from opt_einsum import contract
def chern_number(Gamma,A_idx,B_idx,C_idx):
    P=(np.eye(Gamma.shape[0])-1j*Gamma)/2
    h=12*np.pi*1j*(contract("jk,kl,lj->jkl",P,P,P)-contract("jl,lk,kj->jkl",P,P,P))
    return h[np.ix_(A_idx,B_idx,C_idx)].sum()

def chern_number_quick(Gamma,A_idx,B_idx,C_idx,U1=True):
    P=(np.eye(Gamma.shape[0])-1j*Gamma)/2
    P_AB=P[np.ix_(A_idx,B_idx)]
    P_BC=P[np.ix_(B_idx,C_idx)]
    P_CA=P[np.ix_(C_idx,A_idx)]
    P_AC=P[np.ix_(A_idx,C_idx)]
    P_CB=P[np.ix_(C_idx,B_idx)]
    P_BA=P[np.ix_(B_idx,A_idx)]
    h=12*np.pi*1j*(contract("jk,kl,lj->jkl",P_AB,P_BC,P_CA)-contract("jl,lk,kj->jkl",P_AC,P_CB,P_BA))
    assert np.abs(h.imag).max()<1e-10, "Imaginary part of h is too large"
    nu=h.real.sum()
    # return h
    if U1:
        return nu/2
    else:
        return nu

def circle(i,j,center,radius, angle, Lx=None,Ly=None):
    i_c=i-center[0]
    j_c=j-center[1]
    return i_c**2/radius[0]**2+j_c**2/radius[1]**2<=1 and (angle[0]<=(np.angle(i_c+j_c*1j)%(2*np.pi))<angle[1])


@lru_cache(maxsize=None)
def Gamma_n1(u,n):
    epsilon11=np.array([[0,2*n-1],[1-2*n,0]])
    epsilon12=np.zeros((2,2))
    return Gamma_othor(u,epsilon11,epsilon12)

def Gamma_othor(u,epsilon11,epsilon12):
    """start with a real cov matrix of [[epsilon11,epsilon12],[-epsilon12.T,-epsilon11]] of the shape (4,4) in the eigenbasis, find the new cov matrix with basis transformation, as c^dag = sum u_i c_i^dag
    This can be extended in many scenarios, for example, 
    1. fSWAP is with u = [1,-1], and epsilon11 = zeros(2,2), and epsilon12 = -eye(2)
    2. fSWAP any two arbitary basis is with u =[u1,u2, .., -v1, v2, ...], and epsilon11 = zeros(2,2), and epsilon12 = -eye(2)
    3. real space mode, with u = [u1,u2,..], and construct a projector as u^dag u (u u^dag) , is with epsilon11 = [[0,1],[-1,0]] ([[0,-1],[1,0]]) and epsilon12 = zeros(2).
     """
    L= len(u)
    u=np.array(u)/np.linalg.norm(u)
    X = c2g(u)
    Gamma11 = X.T@epsilon11@X
    VdagV = np.eye(L) - np.outer(u.conj(),u)
    Y = c2g(VdagV)
    Gamma12 = X.T@epsilon12@X + Y
    Gamma21 = -Gamma12.T
    Gamma22 = -Gamma11
    return np.block([[Gamma11,Gamma12],[Gamma21,Gamma22]])

def c2g(u):
    """ convert from BdG nambu spinor to Majorana operators: (I_L \otimes S^dag) (X \otimes [[1,0],[0,0]] + X.conj() \otimes [[0,0],[0,1]]) (I_L \otimes S)"""
    if len(u.shape)==1:
        u=u.reshape(1,-1)
    X = np.zeros((2*u.shape[0],2*u.shape[1]),dtype=float)
    X[::2,::2]=u.real
    X[1::2,1::2]=u.real
    X[::2,1::2]=-u.imag
    X[1::2,::2]=u.imag
    return X

def get_Born_single_mode(Gamma,mode,rng=None):
    """get the outcome of Born measurement for a single mode, 0 or 1, where mode is sum mode[i] c_i^dag"""
    rng=np.random.default_rng(rng)
    prob = get_Born(Gamma,mode)
    # print(prob)
    if rng.random()< prob:
        return 1
    else:
        return 0
@lru_cache(maxsize=None)
def op_fSWAP(state1,state2):
    """state1 mode = \sum_i u_i c_i^dag, encoded in "u", same for the state2"""
    state1=np.array(state1)/np.linalg.norm(state1)
    state2=np.array(state2)/np.linalg.norm(state2)
    u = np.hstack([state1,-state2])/np.sqrt(2)
    epsilon11 = np.zeros((2,2))
    epsilon12 = -np.eye(2)
    return Gamma_othor(u,epsilon11,epsilon12)


def Gamma_othor(u,epsilon11,epsilon12):
    """start with a real cov matrix of [[epsilon11,epsilon12],[-epsilon12.T,-epsilon11]] of the shape (4,4) in the eigenbasis, find the new cov matrix with basis transformation, as c^dag = sum u_i c_i^dag
    This can be extended in many scenarios, for example, 
    1. fSWAP is with u = [1,-1], and epsilon11 = zeros(2,2), and epsilon12 = -eye(2)
    2. fSWAP any two arbitary basis is with u =[u1,u2, .., -v1, -v2, ...], and epsilon11 = zeros(2,2), and epsilon12 = -eye(2)
    3. real space mode, with u = [u1,u2,..], and construct a projector as u^dag u (u u^dag) , is with epsilon11 = [[0,1],[-1,0]] ([[0,-1],[1,0]]) and epsilon12 = zeros(2).
     """
    L= len(u)
    u=np.array(u)/np.linalg.norm(u)
    X = c2g(u)
    Gamma11 = X.T@epsilon11@X
    VdagV = np.eye(L) - np.outer(u.conj(),u)
    Y = c2g(VdagV)
    Gamma12 = X.T@epsilon12@X + Y
    Gamma21 = -Gamma12.T
    Gamma22 = -Gamma11
    return np.block([[Gamma11,Gamma12],[Gamma21,Gamma22]])

def c2g(u):
    """ convert from BdG nambu spinor to Majorana operators: (I_L \otimes S^dag) (X \otimes [[1,0],[0,0]] + X.conj() \otimes [[0,0],[0,1]]) (I_L \otimes S)"""
    if len(u.shape)==1:
        u=u.reshape(1,-1)
    X = np.zeros((2*u.shape[0],2*u.shape[1]),dtype=float)
    X[::2,::2]=u.real
    X[1::2,1::2]=u.real
    X[::2,1::2]=-u.imag
    X[1::2,::2]=u.imag
    return X

def get_C_f(Gamma,normal=True):
    """ get the correlation matrix defined as <c_i^dag c_j>"""
    L=Gamma.shape[0]//2
    S = np.kron(np.eye(L),np.array([[1,1j],[1,-1j]])/2)
    C_f = S@ (np.eye(2*L)-1j * Gamma) @S.conj().T
    if normal:
        return C_f[::2,::2]
    else:
        return C_f

def get_Born(Gamma,u):
    """ get the number density of <V^dag V> where V^dag = sum u_i c_i^dag, C_f is the correlation matrix defined as <c_i^dag c_j>"""
    C_f = get_C_f(Gamma)
    u = np.array(u)/np.linalg.norm(u)
    n = u@C_f@u.conj()
    assert np.abs(n.imag)<1e-10, f'number density is not real {n.imag.max()}'
    return n.real

def get_P(Gamma):
    return (np.eye(Gamma.shape[0])-1j*Gamma)/2

def local_Chern_marker(Gamma,Lx,Ly,shift=[0,0],n_orbit=2,n_maj=2,n_replica=1,n_layer=2,U1=True):
    C_f = get_C_f(Gamma,normal=False)

    replica,layer,x,y,orbit,maj = np.unravel_index(np.arange(C_f.shape[0]),(n_replica,n_layer,Lx,Ly,n_orbit,n_maj))
    x = (x+shift[0])%Lx
    y = (y+shift[1])%Ly
    xy_comm = contract("ij,j,jk,k,ki->i",C_f,x,C_f,y,C_f) - contract("ij,j,jk,k,ki->i",C_f,y,C_f,x,C_f)
    C_r = (xy_comm * 2 * np.pi* 1j)
    C_r=C_r.reshape((n_replica,n_layer,Lx,Ly,n_orbit,n_maj))
    assert np.abs(C_r.imag).max()<1e-10, f'imaginary part is {C_r.imag.max()}'
    if U1:
        return C_r.sum(axis=(-1,-2)).real/2
    else:
        return C_r.sum(axis=(-1,-2)).real