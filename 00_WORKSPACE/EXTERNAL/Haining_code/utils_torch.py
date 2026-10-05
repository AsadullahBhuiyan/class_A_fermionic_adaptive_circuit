import torch
import time
def P_contraction_torch(Gamma,Upsilon,ix,ix_bar,device,err,Gamma_like=None,reset_Gamma_like=True,):
    """ same analytical expression for contraction as _contraction(), differences:
    1. assume intput and output tensor have the same shape, thus, it should be Gamma(L,R) -- Upsilon (L,R), where Gamma_R = Upsilon_L =Upsilon_R, such that in-place operator is applied here.
    2. manually compute the inverse of mat2 before
    Here, Gamma is m, and proj is Upsilon
    Assume Upsilon = [A,B;C,D], the logic is to first compute C= (1+ Gamma_RR @ Upsilon_LL)^{-1}, (where then B=-C.T) 
    then A= Upsilon_LL @C, D= Gamma_RR@ C.T
    ---
    reset_Gamma_like: in usual case, because each application of the gate will be like the brick layer, therefore, ix_bar will overwrite the previous. However, for the staircase pattern, one should reset it. 
    """
    Gamma_RR=Gamma[ix[:,None],ix[None,:]]
    Gamma_LR=Gamma[ix_bar[:,None],ix[None,:]]
    Upsilon_LL=Upsilon[:len(ix),:len(ix)]
    Upsilon_RR=Upsilon[len(ix):,len(ix):]
    Upsilon_RL=Upsilon[len(ix):,:len(ix)]
    eye=torch.eye(len(ix),device=device)
    try: 
        C=torch.linalg.inv(Gamma_RR@Upsilon_LL+eye)
        # lu,pivots=torch.linalg.lu_factor(Gamma_RR@Upsilon_LL+eye)
    except:
        raise ValueError("the contraction will lead to a vanishing state")
    A=Upsilon_LL@C
    D=Gamma_RR@C.T
    # A= torch.linalg.lu_solve(lu,pivots,Upsilon_LL,left=False)
    # D= torch.linalg.lu_solve(lu,pivots,Gamma_RR.T).T
    tmp=Gamma_LR@A@Gamma_LR.T
    if Gamma_like is None:
        Gamma_like=torch.zeros_like(Gamma)
    if reset_Gamma_like:
        Gamma_like.fill_(0)
    Gamma_like[ix_bar[:,None],ix_bar[None,:]]=tmp
    Gamma+=Gamma_like
    Gamma[ix[:,None],ix_bar[None,:]]=Upsilon_RL@C@Gamma_LR.T
    # Gamma[ix[:,None],ix_bar[None,:]]=torch.linalg.lu_solve(lu,pivots,Upsilon_RL,left=False)@Gamma_LR.T
    Gamma[ix[:,None],ix[None,:]]=Upsilon_RR+Upsilon_RL@D@Upsilon_RL.T
    Gamma[ix_bar[:,None],ix[None,:]]=-Gamma[ix[:,None],ix_bar[None,:]].T
    # why is it neccessary?
    # Gamma-=Gamma.T
    # Gamma/=2
    # print(torch.abs(torch.einsum(Gamma,[0,1],Gamma,[1,0],[0])+1).max())
    max_err=max_error(Gamma)
    if  max_err > err:
        st=time.time()
        print(f'Purification: {max_err}')
        Gamma=purify(Gamma)
        print('Purification done in {:.4f}s with new error{}'.format(time.time()-st,max_error(Gamma)))
        

def max_error(Gamma):
    return torch.abs(torch.einsum(Gamma,[0,1],Gamma,[1,0],[0])+1).max()
# def purify(A):
#     A = (A-A.T)/2
#     U, _, Vh=torch.linalg.svd(A)
#     return U@Vh

def purify(A):
    # purify A, see App. B2 in PhysRevB.106.134206
    val,vec=torch.linalg.eigh(A/1j)
    mask_neg=val<0
    mask_pos=val>0
    val[mask_neg]=-1
    val[mask_pos]=1
    val=val+0j
    A= -(vec@torch.diag(val)@vec.conj().T).imag
    A=A-A.T
    A/=2
    return A

def get_O(rng,n,device,dtype):
    # rng=np.random.default_rng(rng)
    A=torch.normal(0,1,size=(n,n),generator=rng,device=device,dtype=dtype)/2
    A=A-A.T
    return torch.linalg.matrix_exp(A)

def chern_number_quick(Gamma,A_idx,B_idx,C_idx,device,dtype,U1=True,):
    st=time.time()
    P=(torch.eye(Gamma.shape[0],device=device,dtype=dtype)-1j*Gamma)/2
    P_AB=P[A_idx[:,None],B_idx[None,:]]
    P_BC=P[B_idx[:,None],C_idx[None,:]]
    P_CA=P[C_idx[:,None],A_idx[None,:]]
    P_AC=P[A_idx[:,None],C_idx[None,:]]
    P_CB=P[C_idx[:,None],B_idx[None,:]]
    P_BA=P[B_idx[:,None],A_idx[None,:]]
    h=12*torch.pi*1j*(torch.einsum("jk,kl,lj->jkl",P_AB,P_BC,P_CA)-torch.einsum("jl,lk,kj->jkl",P_AC,P_CB,P_BA))
    # assert np.abs(h.imag).max()<1e-10, "Imaginary part of h is too large"
    nu=h.real.sum()
    print('Chern number done in {:.4f}'.format(time.time()-st))
    if U1:
        return nu/2
    else:
        return nu

def correlation_length(C_m,replica,layer,Lx,Ly,):
    """C_m is expected to be the element-wise square of the covariance matrix"""
    D=(replica,layer,Lx,Ly,2,2)
    C_ij=C_m.reshape(D+D)[0,0,:,:,:,:,0,0,:,:,:,:].mean(dim=(2,3,6,7))
    Cr_i=torch.stack([C_ij[i,j,(i+torch.arange(Lx//2)+1)%Lx,j] for i in range(Lx) for j in range(Ly)]).mean(dim=0)
    Cr_j=torch.stack([C_ij[i,j,i,(j+torch.arange(Ly//2)+1)%Ly] for i in range(Lx) for j in range(Ly)]).mean(dim=0)
    c_ij=C_m.reshape(D+D)[0,1,:,:,:,:,0,1,:,:,:,:].mean(dim=(2,3,6,7))
    cr_i=torch.stack([c_ij[i,j,(i+torch.arange(Lx//2)+1)%Lx,j] for i in range(Lx) for j in range(Ly)]).mean(dim=0)
    cr_j=torch.stack([c_ij[i,j,i,(j+torch.arange(Ly//2)+1)%Ly] for i in range(Lx) for j in range(Ly)]).mean(dim=0)
    return Cr_i,Cr_j,cr_i,cr_j

def correlation_i_length(C_m,replica,layer,Lx,Ly,i0):
    """C_m is expected to be the element-wise square of the covariance matrix"""
    D=(replica,layer,Lx,Ly,2,2)
    C_ij=C_m.reshape(D+D)[0,0,:,:,:,:,0,0,:,:,:,:].mean(dim=(2,3,6,7))
    Cr_j=torch.stack([C_ij[i,j,i,(j+torch.arange(Ly//2)+1)%Ly] for i in [i0] for j in range(Ly)]).mean(dim=0)
    return Cr_j

def correlation_i2_length(C_m,replica,layer,Lx,Ly,i0,i1):
    D=(replica,layer,Lx,Ly,2,2)
    C_ij=C_m.reshape(D+D)[0,0,:,:,:,:,0,0,:,:,:,:].mean(dim=(2,3,6,7))  # trace out the sublattice and Majorana degree of freedom
    Cr_j=torch.stack([C_ij[i0,j,i1,j] for j in range(Ly)]).mean()
    return Cr_j.item()


def fidelity(A,B):
    # https://doi.org/10.1063/1.5093326
    assert A.shape[0]==B.shape[0], f'A {A.shape[0]} has different dim than B{B.shape[0]}'
    L=A.shape[0]//2
    # identity=np.eye(A.shape[0])
    # id_AB=(identity-A@B)
    AB=-A@B
    AB.diagonal().add_(1.) # (I-A@B)
    AB.div_(2.)  # ../2
    prod1=torch.linalg.det(AB)
    if prod1 == 0:
        return 0.
    else:
        AB.mul_(2)
        AB_tilde = (A+B)@torch.linalg.inv(AB) # G_tilde=(A+B)@(1-A@B)^{-1}
        AB_tilde = AB_tilde@AB_tilde    # G_tilde^2
        AB_tilde.diagonal().add_(1.) # 1+..^2
        AB_tilde = sqrt(AB_tilde) # (..)^{1/2}
        AB_tilde.diagonal().add_(1.) # det(1+..)
        prod2=torch.linalg.det(AB_tilde)
        return torch.sqrt(prod1*prod2).item().real


def sqrt(A):
    val,vec=torch.linalg.eigh(A)
    val_sqrt = torch.sqrt(val+0j)
    vec=vec+0j
    return vec@torch.diag(val_sqrt)@vec.conj().T

def get_C_f(Gamma, normal=True, device=None):
    """
    Convert covariance matrix in Majorana basis to correlation matrix in fermionic basis.

    Parameters
    ----------
    Gamma : torch.Tensor
        Covariance matrix in Majorana basis, shape (..., 2L, 2L)
        Supports arbitrary batch dimensions. Should be real-valued.
    normal : bool, optional
        If True, return only the normal part C_f[..., ::2, ::2], shape (..., L, L)
        If False, return the full correlation matrix, shape (..., 2L, 2L)
        Default is True
    device : torch.device, optional
        Device for computation. If None, uses Gamma.device

    Returns
    -------
    C_f : torch.Tensor
        Correlation matrix <c_i^dag c_j> in fermionic basis
        Shape (..., L, L) if normal=True, (..., 2L, 2L) if normal=False

    Notes
    -----
    The transformation uses S = kron(I_L, S0) where S0 = [[1,1j],[1,-1j]]/2
    Formula: C_f = S @ (I - i*Gamma) @ S^dag
    Works on batched inputs through PyTorch broadcasting.
    The complex dtype is automatically inferred from Gamma's dtype (float32→complex64, float64→complex128).

    Examples
    --------
    >>> Gamma = torch.randn(10, 20, 20)  # batch of 10 matrices, L=10
    >>> C_f = get_C_f(Gamma)  # Returns shape (10, 10, 10)
    """
    if device is None:
        device = Gamma.device

    # Infer complex dtype from input dtype
    if Gamma.dtype == torch.float32:
        dtype_complex = torch.complex64
    elif Gamma.dtype == torch.float64:
        dtype_complex = torch.complex128
    else:
        dtype_complex = torch.complex128  # default fallback

    L = Gamma.shape[-1] // 2
    S0 = torch.tensor([[1, 1j], [1, -1j]], device=device, dtype=dtype_complex) / 2
    S = torch.kron(torch.eye(L, device=device, dtype=dtype_complex), S0)
    C_f = S @ (torch.eye(2*L, device=device, dtype=dtype_complex) - 1j * Gamma) @ S.conj().T

    if normal:
        return C_f[..., ::2, ::2]
    else:
        return C_f


def get_C_m(C_f, normal=True, device=None, dtype_complex=None):
    """
    Convert correlation matrix in fermionic basis to covariance matrix in Majorana basis.

    This is the inverse transformation of get_C_f().

    Parameters
    ----------
    C_f : torch.Tensor
        Correlation matrix <c_i^dag c_j> in fermionic basis
        If normal=True, expected shape is (..., L, L)
        If normal=False, not yet implemented
        Supports arbitrary batch dimensions. Should be complex-valued.
    normal : bool, optional
        If True, input C_f is the normal part only (shape (..., L, L))
        If False, not yet implemented
        Default is True
    device : torch.device, optional
        Device for computation. If None, uses C_f.device
    dtype_complex : torch.dtype, optional
        Complex dtype to use. If None, uses C_f.dtype

    Returns
    -------
    C_m : torch.Tensor
        Covariance matrix in Majorana basis (imaginary part only), shape (..., 2L, 2L)
        Real-valued tensor with same precision as input.

    Notes
    -----
    The transformation constructs a projector P from C_f and uses:
    C_m = I - 2*P where P is derived from the fermionic correlation matrix
    Works on batched inputs through PyTorch broadcasting.
    The dtype is automatically inferred from C_f's dtype if not specified.

    Raises
    ------
    NotImplementedError
        If normal=False
    AssertionError
        If the resulting C_m has significant real part (>1e-10)

    Examples
    --------
    >>> C_f = torch.randn(10, 5, 5, dtype=torch.complex64)  # batch of 10 matrices, L=5
    >>> C_m = get_C_m(C_f)  # Returns shape (10, 10, 10), dtype float32
    """
    if device is None:
        device = C_f.device

    if dtype_complex is None:
        dtype_complex = C_f.dtype

    if normal:
        # Get batch shape and L
        batch_shape = C_f.shape[:-2]
        L = C_f.shape[-1]

        S0 = torch.tensor([[1, 1j], [1, -1j]], device=device, dtype=dtype_complex) / 2
        S = torch.kron(torch.eye(L, device=device, dtype=dtype_complex), S0)

        # Construct full correlation matrix with proper batch dimensions
        C_f_ = torch.eye(L, dtype=dtype_complex, device=device) - C_f.transpose(-2, -1)
        C_ = torch.zeros(batch_shape + (L, 2, L, 2), dtype=dtype_complex, device=device)
        C_[..., :, 0, :, 0] = C_f
        C_[..., :, 1, :, 1] = C_f_
        C_ = C_.reshape(batch_shape + (2*L, 2*L))

        # Compute covariance matrix
        P_ = S.conj().T @ C_ @ S * 2

        # Determine appropriate real dtype for identity matrix
        if dtype_complex == torch.complex64:
            dtype_real = torch.float32
        else:
            dtype_real = torch.float64

        C_m = torch.eye(2*L, device=device, dtype=dtype_real) - P_ * 2

        # Check that result is purely imaginary
        max_real = C_m.real.abs().max()
        assert max_real < 1e-10, f'largest real part is {max_real}'
        return C_m.imag
    else:
        raise NotImplementedError("The 'normal=False' case is not yet implemented.")


def compute_current(Gamma, replica, layer, Lx, Ly, n_orbitals=2, direction='y', which_replica=0, which_layer=0,phase='real'):
    """
    Compute the current between neighboring sites from the Majorana covariance matrix.

    J_{x,y,y+1} = (1/2) * sum_{μ,μ'} Im[G_{(x,y,μ), (x,y+1,μ')}]
    where G = <c^†_i c_j> is the correlation matrix in fermionic basis.

    Parameters
    ----------
    Gamma : torch.Tensor
        Covariance matrix in Majorana basis, shape (2L, 2L) where L = replica*layer*Lx*Ly*n_orbitals
    replica, layer, Lx, Ly : int
        Dimensions of the system
    n_orbitals : int, optional
        Number of orbitals per site (default: 2)
    direction : str, optional
        'y' for vertical or 'x' for horizontal current (default: 'y')
    which_replica, which_layer : int, optional
        Indices to select specific replica/layer (default: 0, 0)

    Returns
    -------
    J : torch.Tensor, shape (Lx, Ly)
        Current map where J[x,y] is the current from (x,y) to its neighbor
    """
    C_f = get_C_f(Gamma, normal=True)

    D = (replica, layer, Lx, Ly, n_orbitals)
    C_f_reshaped = C_f.reshape(D + D)

    G = C_f_reshaped[which_replica, which_layer, :, :, :, which_replica, which_layer, :, :, :]

    if direction == 'y':
        x_idx = torch.arange(Lx)
        y_idx = torch.arange(Ly)
        y_next_idx = (y_idx + 1) % Ly
        G_neighbors = G[x_idx[:, None], y_idx[None, :], :, x_idx[:, None], y_next_idx[None, :], :]
        J = getattr(G_neighbors,phase).sum(dim=(2, 3)) / 2.0

    elif direction == 'x':
        x_idx = torch.arange(Lx)
        y_idx = torch.arange(Ly)
        x_next_idx = (x_idx + 1) % Lx
        G_neighbors = G[x_idx[:, None], y_idx[None, :], :, x_next_idx[:, None], y_idx[None, :], :]
        J = getattr(G_neighbors,phase).sum(dim=(2, 3)) / 2.0

    else:
        raise ValueError(f"direction must be 'x' or 'y', got '{direction}'")

    return J


def compute_current_orbital_weighted(Gamma, replica, layer, Lx, Ly, direction='y', which_replica=0, which_layer=0):
    """
    Compute orbital-weighted current using specific orbital transition weights.

    J_x(r+ŷ/2) = Im[-1/2 a†_r a_{r+x̂} + 1/2 b†_r b_{r+x̂} + 1/2 a†_r b_{r+x̂}]
    J_y(r+x̂/2) = Im[-1/2 a†_r a_{r+ŷ} + 1/2 b†_r b_{r+ŷ} - 1/2 a†_r b_{r+ŷ}]

    Parameters
    ----------
    Gamma : torch.Tensor
        Covariance matrix in Majorana basis
    replica, layer, Lx, Ly : int
        Dimensions of the system
    direction : str, optional
        'y' for vertical or 'x' for horizontal current (default: 'y')
    which_replica, which_layer : int, optional
        Indices to select specific replica/layer (default: 0, 0)

    Returns
    -------
    J : torch.Tensor, shape (Lx, Ly)
        Orbital-weighted current map
    """
    C_f = get_C_f(Gamma, normal=True)

    D = (replica, layer, Lx, Ly, 2)  # n_orbitals = 2 hardcoded
    C_f_reshaped = C_f.reshape(D + D)

    G = C_f_reshaped[which_replica, which_layer, :, :, :, which_replica, which_layer, :, :, :]

    x_idx = torch.arange(Lx)
    y_idx = torch.arange(Ly)

    if direction == 'y':
        y_next_idx = (y_idx + 1) % Ly
        # Extract orbital-specific transitions: (x,y,orbital) -> (x,y+1,orbital')
        G_aa = G[x_idx[:, None], y_idx[None, :], 0, x_idx[:, None], y_next_idx[None, :], 0]
        G_bb = G[x_idx[:, None], y_idx[None, :], 1, x_idx[:, None], y_next_idx[None, :], 1]
        G_ab = G[x_idx[:, None], y_idx[None, :], 0, x_idx[:, None], y_next_idx[None, :], 1]
        J = (-0.5 * G_aa + 0.5 * G_bb - 0.5 * G_ab).imag

    elif direction == 'x':
        x_next_idx = (x_idx + 1) % Lx
        # Extract orbital-specific transitions: (x,y,orbital) -> (x+1,y,orbital')
        G_aa = G[x_idx[:, None], y_idx[None, :], 0, x_next_idx[:, None], y_idx[None, :], 0]
        G_bb = G[x_idx[:, None], y_idx[None, :], 1, x_next_idx[:, None], y_idx[None, :], 1]
        G_ab = G[x_idx[:, None], y_idx[None, :], 0, x_next_idx[:, None], y_idx[None, :], 1]
        J = (-0.5 * G_aa + 0.5 * G_bb + 0.5 *1j * G_ab).imag

    else:
        raise ValueError(f"direction must be 'x' or 'y', got '{direction}'")

    return J


def check_current_conservation(Gamma, replica, layer, Lx, Ly, n_orbitals=2, which_replica=0, which_layer=0):
    """
    Check current conservation (continuity equation) at each site.

    For a stationary, number-conserving ground state, the local divergence
    δ[x,y] = ∇·J = (J_x[x,y] - J_x[x-1,y]) + (J_y[x,y] - J_y[x,y-1])
    should be ≈ 0 at every site.

    Parameters
    ----------
    Gamma : torch.Tensor
        Covariance matrix in Majorana basis
    replica, layer, Lx, Ly : int
        Dimensions of the system
    n_orbitals : int, optional
        Number of orbitals per site (default: 2)
    which_replica, which_layer : int, optional
        Indices to select specific replica/layer (default: 0, 0)

    Returns
    -------
    divergence : torch.Tensor, shape (Lx, Ly)
        Local divergence at each site. Should be ≈ 0 for bulk sites.
    max_div : float
        Maximum absolute divergence
    mean_div : float
        Mean absolute divergence
    """
    # Compute currents in both directions
    J_x = compute_current(Gamma, replica, layer, Lx, Ly, n_orbitals, direction='x',
                          which_replica=which_replica, which_layer=which_layer)
    J_y = compute_current(Gamma, replica, layer, Lx, Ly, n_orbitals, direction='y',
                          which_replica=which_replica, which_layer=which_layer)

    # Compute divergence: ∇·J = ∂J_x/∂x + ∂J_y/∂y
    # J_x[x,y] is current OUT of (x,y) in +x direction
    # J_x[x-1,y] is current INTO (x,y) from -x direction
    # Net outflow in x: J_x[x,y] - J_x[x-1,y]

    div_x = J_x - torch.roll(J_x, shifts=1, dims=0)  # J_x[x,y] - J_x[x-1,y]
    div_y = J_y - torch.roll(J_y, shifts=1, dims=1)  # J_y[x,y] - J_y[x,y-1]

    divergence = div_x + div_y

    max_div = divergence.abs().max().item()
    mean_div = divergence.abs().mean().item()

    return divergence, max_div, mean_div
