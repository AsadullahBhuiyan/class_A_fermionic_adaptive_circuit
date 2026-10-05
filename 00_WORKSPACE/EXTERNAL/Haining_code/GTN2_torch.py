import numpy as np
import numpy.linalg as nla
import torch
import time
from utils_torch import P_contraction_torch, get_O
from utils import op_single_mode, op_fSWAP, circle

class GTN2_torch:
    def __init__(self,Lx,Ly,history=True,seed=None,random_init=False,random_U1=False,bcx=1,bcy=1,orbit=1,layer=1,replica=1,nshell=1,gpu=True,complex128=True,err=1e-8):
        self.Lx= Lx # complex fermion sites
        self.Ly=Ly # complex fermion sites
        self.L = Lx* Ly*orbit # (Lx,Ly) in complex fermion sites
        self.orbit=orbit # number of orbitals
        self.layer=layer # number of layers
        self.replica=replica # number of replicas, for the reference sites
        self.gpu=gpu
        self.device=self._initialize_device()
        self.history = history
        self.random_init = random_init
        self.random_U1 = random_U1
        self.dtype_float=torch.float64 if complex128 else torch.float32
        self.dtype_complex=torch.complex128 if complex128 else torch.complex64
        self.err=torch.tensor(err,device=self.device)
        self.rng=torch.Generator(device=self.device).manual_seed(seed)
        self.C_m=self.correlation_matrix()
        self.Gamma_like=torch.zeros_like(self.C_m)
        self.C_m_history=[self.C_m.cpu().clone()]
        self.n_history=[]
        self.i_history=[]
        self.p_history=[]
        self.seed = seed
        self.bcx = bcx # boundary condition in x direction, 0 for open, 1 for periodic, -1 for antiperiodic
        self.bcy = bcy # boundary condition in y direction, 0 for open, 1 for periodic, -1 for antiperiodic
        self.full_ix=set(range(self.C_m.shape[0]))
        self.ix_bool=torch.zeros(self.C_m.shape[0],dtype=torch.bool,device=self.device)
        self.nshell = nshell
        self.S0=torch.tensor([[1,1j],[1,-1j]],device=self.device,dtype=self.dtype_complex)/2
    
    def _initialize_device(self):
        """Initialize the device, if `gpu` is True, use GPU, otherwise, use CPU.

        Returns
        -------
        cuda instance
            the name of GPU device

        Raises
        ------
        ValueError
            If GPU is not available, raise error.
        """
        if self.gpu:
            if torch.cuda.is_available():
                device = torch.device("cuda")
                gpu_name = torch.cuda.get_device_name(0)
                print('Using',device)
                print(f"GPU Model: {gpu_name}")
                return device
            else:
                raise ValueError('CUDA is not available')
        else:
            print('Using cpu')
    
    def set(self,ij_list,n,Gamma=None):
        """ij_list: [[i,j],...]
        n: [1,-1,...]
        simply set all Gamma[i,j]=n, Gamma[j,i]=-n
        """
        if Gamma is None:
            Gamma=self.C_m
        for ij,n in zip(ij_list,n):
            i,j=ij
            Gamma[i,j]=n
            Gamma[j,i]=-n

    def correlation_matrix(self):
        L_complex_f=self.replica*self.layer*self.L
        Omega=torch.tensor([[0,1.],[-1.,0]],device=self.device,dtype=self.dtype_float)
        eyeL=torch.eye(L_complex_f,device=self.device,dtype=self.dtype_float)
        if self.replica==2:
            # the reference sites are entangled with their own site indices
            Omega_diag=torch.kron(Omega,eyeL)
        else:
            Omega_diag=torch.kron(eyeL,Omega)
        if self.random_init:
            if self.random_U1:
                # random with U1
                i_list = torch.rand(size=(L_complex_f,), generator=self.rng, device=self.device, dtype=self.dtype_float)
                i_list = 2*torch.arange(L_complex_f,device=self.device)[i_list<0.5]
                j_list = i_list+1
                ij_list=torch.vstack([i_list,j_list]).T
                self.set(ij_list=ij_list,n=[-1]*i_list.shape[0],Gamma=Omega_diag)
                Gamma=Omega_diag
            else:
                # random without U1, in this scenario, the sparse matrix is not optimal
                O=get_O(self.rng,2*L_complex_f,device=self.device,dtype=self.dtype_float)
                Gamma=O@Omega_diag@O.T
        else:
            Gamma=Omega_diag
            # flip half of the bit to make it half-filling
            i_list = torch.arange( 0,Gamma.shape[0],4 )
            j_list = i_list+1
            ij_list=torch.vstack([i_list,j_list]).T
            self.set(ij_list=ij_list,n=[-1]*i_list.shape[0],Gamma=Omega_diag)
        return (Gamma-Gamma.T)/2
    
    def measure_feedback(self,ij,mu=None,tau=None,feedback=True,region=None):
        """ix is the 2 fermionic site index
        this can be used to incorporate feedback"""
        i,j=ij
        if mu is None:
            mu = list(self.a_i.keys())[0]
        legs_t_lower,wf_lower=self.generate_ij_wf(i,j,self.a_i[mu,tau],self.b_i[mu,tau],self.bcx,self.bcy,region=region)
        legs_t_upper,wf_upper=self.generate_ij_wf(i,j,self.A_i[mu,tau],self.B_i[mu,tau],self.bcx,self.bcy,region=region)

        legs_bA = [self.linearize_idx(i=i,j=j,orbit_idx=0,majorana=majorana)+2*self.L for majorana in range(2)]
        legs_bB = [self.linearize_idx(i=i,j=j,orbit_idx=1,majorana=majorana)+2*self.L for majorana in range(2)]

        # fill lower band
        mode_m,n_m=self.measure_single_mode_Born(legs_t_lower,mode=wf_lower)
        if n_m ==1:
            # this is good
            pass
        elif n_m ==0:
            self.fSWAP(legs_t_lower+legs_bA,state1 = wf_lower, state2=(1,))
            # self.fSWAP(legs_t_lower+legs_bB,state1 = wf_lower, state2=(1,))

        # deplete upper band
        mode_p,n_p=self.measure_single_mode_Born(legs_t_upper,mode=wf_upper)
        if n_p ==0:
            # this is good
            pass
        elif n_p == 1:
            # self.fSWAP(legs_t_upper+legs_bA,state1 = wf_upper, state2=(1,))
            self.fSWAP(legs_t_upper+legs_bB,state1 = wf_upper, state2=(1,))

    def measure_feedback_force(self,ij,mu=None,tau=None,feedback=True,region=None):
        """ix is the 2 fermionic site index
        this can be used to incorporate feedback"""
        i,j=ij
        if mu is None:
            mu = list(self.a_i.keys())[0]
        legs_t_lower,wf_lower=self.generate_ij_wf(i,j,self.a_i[mu,tau],self.b_i[mu,tau],self.bcx,self.bcy,region=region)
        legs_t_upper,wf_upper=self.generate_ij_wf(i,j,self.A_i[mu,tau],self.B_i[mu,tau],self.bcx,self.bcy,region=region)

        legs_bA = [self.linearize_idx(i=i,j=j,orbit_idx=0,majorana=majorana)+2*self.L for majorana in range(2)]
        legs_bB = [self.linearize_idx(i=i,j=j,orbit_idx=1,majorana=majorana)+2*self.L for majorana in range(2)]

        # fill lower band
        mode_m,n_m=self.measure_single_mode_set(legs_t_lower,mode=wf_lower,n=1)

        # deplete upper band
        mode_p,n_p=self.measure_single_mode_set(legs_t_upper,mode=wf_upper,n=0)
    
    def order_parameter(self,mu,tau_list=[(1,0),(0,1)],region=None):
        """ the order parameter is defined as sum_{ij} (1-n_- + n_+)/L; such that it is effective the defect density per unit cell"""
        # if mu is None:
        #     mu = list(self.a_i.keys())[0]
        n_lower, n_upper = 0,0
        for i in range(self.Lx):
        # for i in range(1):
            for j in range(self.Ly):
            # for j in range(1):
                for tau in tau_list:
                    legs_t_lower,wf_lower=self.generate_ij_wf(i,j,self.a_i[mu,tau],self.b_i[mu,tau],self.bcx,self.bcy,region=region)
                    legs_t_upper,wf_upper=self.generate_ij_wf(i,j,self.A_i[mu,tau],self.B_i[mu,tau],self.bcx,self.bcy,region=region)
                    legs_t_lower=torch.tensor(legs_t_lower,device=self.device)
                    legs_t_upper=torch.tensor(legs_t_upper,device=self.device)
                    Gamma = self.C_m[legs_t_lower[:,None],legs_t_lower[None,:]]
                    n_lower += self.get_Born(Gamma,u=wf_lower)
                    Gamma = self.C_m[legs_t_upper[:,None],legs_t_upper[None,:]]
                    n_upper += self.get_Born(Gamma,u=wf_upper)
        return (2-(n_lower-n_upper)/(self.Lx*self.Ly))/2


                

    def measure_single_mode_Born(self,legs,mode):
        """measure the single mode with mode = (wf, n), wavefunction and occupation number 
        """
        legs=torch.tensor(legs,device=self.device)
        Gamma = self.C_m[legs[:,None],legs[None,:]]
        n = self.get_Born_single_mode(Gamma=Gamma,mode=mode,rng=self.rng)
        self.measure_single_mode_force(kind=(mode,n),ix=legs)
        return (mode,n)

    def measure_single_mode_set(self,legs,mode,n):
        """measure the single mode with mode = (wf, n), wavefunction and occupation number 
        """
        legs=torch.tensor(legs,device=self.device)
        Gamma = self.C_m[legs[:,None],legs[None,:]]
        self.measure_single_mode_force(kind=(mode,n),ix=legs)
        return (mode,n)

    def measure_single_mode_force(self,kind,ix,):
        ''' Majorana site index for ix'''
        assert len(ix)==len(kind[0])*2, 'len of ix should be 2*len(kind[0])'
        Psi=self.C_m
        # ix_bar=torch.tensor(list(self.full_ix-set(ix.tolist())),device=self.device)
        ix_bar=self.complement(ix)
        kind = (tuple(kind[0]),kind[1])
        proj=torch.tensor(op_single_mode(kind),device=self.device,dtype=self.dtype_float)
        P_contraction_torch(Psi,proj,ix,ix_bar,device=self.device,err=self.err,Gamma_like=self.Gamma_like,reset_Gamma_like=True)
        if self.history:
            self.C_m_history.append(Psi.cpu().clone())
            self.n_history.append(kind)
            self.i_history.append(ix)
            # self.MI_history.append(self.mutual_information_cross_ratio())
        else:
            # self.C_m_history=[Psi]
            self.n_history=[kind]
            self.i_history=[ix]
            # self.MI_history=[self.mutual_information_cross_ratio()]

    def fSWAP(self,ix,state1=None,state2=None):
        """ Majorana site index for ix ,exp(i*pi*c_-^dag c_-)"""
        # ix_bar=torch.tensor(list(self.full_ix-set(ix)),device=self.device)
        ix_bar=self.complement(ix)
        ix=torch.tensor(ix,device=self.device)
        Psi=self.C_m
        op=torch.tensor(op_fSWAP(state1,state2),device=self.device,dtype=self.dtype_float)
        P_contraction_torch(Psi,op,ix,ix_bar,device=self.device,err=self.err,Gamma_like=self.Gamma_like,reset_Gamma_like=True)
        if self.history:
            self.C_m_history.append(Psi.cpu().clone())
            self.n_history.append([state1,state2])
            self.i_history.append(ix)
        else:
            self.n_history=[state1,state2]
            self.i_history=[ix]

    def randomize(self,legs, scale=1):
        """ legs is the majorana site index, 
        simply randomize the parity"""
        phi=torch.rand((1,),generator=self.rng,device=self.device,dtype=self.dtype_float)*2*np.pi * scale
        n_list=torch.tensor([0,torch.cos(phi),torch.sin(phi)],device=self.device,dtype=self.dtype_float)
        self.measure(n_list,ix=legs)

    def measure(self,n,ix):
        ''' Majorana site index for ix, 
        n should be a scalar'''
        # ix_bar=torch.tensor(list(self.full_ix-set(ix)),device=self.device)
        ix_bar=self.complement(ix)
        ix=torch.tensor(ix,device=self.device)
        Psi=self.C_m
        proj=self.kraus(n)
        P_contraction_torch(Psi,proj,ix,ix_bar,device=self.device,err=self.err,Gamma_like=self.Gamma_like,reset_Gamma_like=True)

        if self.history:
            self.C_m_history.append(Psi.cpu().clone())
            self.n_history.append(n)
            self.i_history.append(ix)
            # self.MI_history.append(self.mutual_information_cross_ratio())
        else:
            # self.C_m_history=[Psi]
            self.n_history=[n]
            self.i_history=[ix]
            # self.MI_history=[self.mutual_information_cross_ratio()]

    def generate_ij_wf(self,i,j,a_i,b_i,bcx,bcy,region=None):
        """generate ij_list from a local mode a_i;
        assume a_i, and b_i have the same keys"""
        ij_list = []
        wf = []
        for di,dj in a_i.keys():
            i1,j1=(i+di),(j+dj)
            if bcx==1:
                i1=i1%self.Lx
            elif bcx==0:
                if i1<0 or i1>=self.Lx:
                    continue
            if bcy==1:
                j1=j1%self.Ly
            elif bcy==0:
                if j1<0 or j1>=self.Ly:
                    continue
            if region is not None:
                if (i1,j1) not in region:
                    continue
            ij_list.append((i1,j1))
            wf.append(a_i[di,dj])
            wf.append(b_i[di,dj])
        legs=[self.linearize_idx(*ij,orbit_idx=orbit_idx,majorana=idx) for ij in ij_list for orbit_idx in range(2) for idx in range(2)]
        return legs, tuple(wf)
    
    def linearize_idx(self,i,j,majorana=0,orbit_idx=0,layer=0,replica=0):
        return np.ravel_multi_index((replica,layer,i,j,orbit_idx,majorana),(self.replica,self.layer,self.Lx,self.Ly,self.orbit,2))

    # def linearize_idx_0(self,i,j,majorana=0,orbit_idx=0,layer=0,replica=0):
    #     if isinstance(i,int):
    #         i=[i]
    #     linear_idx_rev= torch.tensor([majorana,orbit_idx,j,i,layer,replica],device=self.device)
    #     weights= torch.tensor([self.replica,self.layer,self.Lx,self.Ly,self.orbit,2],device=self.device).comprod_(dim=0)
        # return np.ravel_multi_index(,())


    # def linearize_idx(self,index, shape):
    #     out = []
    #     for dim in reversed(shape):
    #         out.append(index % dim)
    #         index = index // dim
    #     return tuple(reversed(out))

    def linearize_idx_span(self,ilist,jlist,layer = 0,replica=0,shape_func=lambda i,j: True, shift=(0,0)):
        multi_index=np.array([(replica,layer,(i+shift[0])%self.Lx,(j+shift[1])%self.Ly,orbit,maj) for i in ilist for j in jlist for orbit in range(self.orbit) for maj in range(2) if shape_func(i%self.Lx,j%self.Ly)])
        return np.ravel_multi_index(multi_index.T,(self.replica,self.layer,self.Lx,self.Ly,self.orbit,2))

    def delinearize_idx(self,idx):
        return np.unravel_index(idx,(self.replica,self.layer,self.Lx,self.Ly,self.orbit,2))
        
    def get_Born_single_mode(self,Gamma,mode,rng):
        """get the outcome of Born measurement for a single mode, 0 or 1, where mode is sum mode[i] c_i^dag"""
        prob = self.get_Born(Gamma,mode)
        if torch.rand((1,),generator=self.rng,device=self.device,dtype=self.dtype_float)< prob:
            return 1
        else:
            return 0

    def get_Born(self,Gamma,u):
        """ get the number density of <V^dag V> where V^dag = sum u_i c_i^dag, C_f is the correlation matrix defined as <c_i^dag c_j>"""
        C_f = self.get_C_f(Gamma)
        u = torch.tensor(u,device=self.device,dtype=self.dtype_complex)
        u/=torch.linalg.norm(u)
        n = u@C_f@u.conj()
        # assert np.abs(n.imag)<1e-10, f'number density is not real {n.imag.max()}'
        return n.real


    def get_C_f(self,Gamma,normal=True):
        """ get the correlation matrix defined as <c_i^dag c_j>"""
        L=Gamma.shape[0]//2
        S = torch.kron(torch.eye(L,device=self.device),self.S0)
        C_f = S@ (torch.eye(2*L,device=self.device)-1j * Gamma) @S.conj().T
        if normal:
            return C_f[::2,::2]
        else:
            return C_f
    
    def get_C_m(self,C_f,normal=True):
        if normal:
            S = torch.kron(torch.eye(C_f.shape[0],device=self.device),self.S0)
            C_f_=torch.eye(C_f.shape[0],dtype=self.dtype_complex,device=self.device)-C_f.T
            C_ = torch.zeros((C_f.shape[0],2,C_f.shape[0],2),dtype=self.dtype_complex,device=self.device)
            C_[:,0,:,0]=C_f
            C_[:,1,:,1]=C_f_
            C_=C_.reshape((2*C_f.shape[0],2*C_f.shape[0]))
            P_ = S.conj().T @ C_ @ S *2
            C_m = torch.eye(2*C_f.shape[0],device=self.device) - P_ *2 
            # return C_m
            assert C_m.real.max()<1e-10, f'largest real part is {C_m.real.max()}'
            return C_m.imag
        else:
            raise NotImplementedError("The 'normal=False' case is not yet implemented.")
    
    def kraus(self,n):
        return torch.tensor([[0,n[0],n[1],n[2]],
                        [-n[0],0,-n[2],n[1]],
                        [-n[1],n[2],0,-n[0]],
                        [-n[2],-n[1],n[0],0]],device=self.device,dtype=self.dtype_float)
    
    def complement(self,ix):
        self.ix_bool.fill_(False)
        self.ix_bool[ix]=True
        ix_bar = torch.nonzero(~self.ix_bool,as_tuple=True)[0]
        return ix_bar
    
    def generate_tripartite_circle(self,radius_factor,center=None,shift=(0,0)):
        if center is None:
            center=[self.Lx/2,self.Ly/2]
        radius = [self.Lx/radius_factor[0],self.Ly/radius_factor[1]]
        
        A_idx_0=self.linearize_idx_span(np.arange(self.Lx),np.arange(self.Ly),shape_func=lambda i,j: circle(i,j,Lx=self.Lx,Ly=self.Ly,center=center,radius=radius,angle=[0,np.pi/3*2]),shift=shift)
        B_idx_0=self.linearize_idx_span(np.arange(self.Lx),np.arange(self.Ly),shape_func=lambda i,j: circle(i,j,Lx=self.Lx,Ly=self.Ly,center=center,radius=radius,angle=[np.pi/3*2,np.pi/3*4]),shift=shift)
        C_idx_0=self.linearize_idx_span(np.arange(self.Lx),np.arange(self.Ly),shape_func=lambda i,j: circle(i,j,Lx=self.Lx,Ly=self.Ly,center=center,radius=radius,angle=[np.pi/3*4,np.pi/3*6]),shift=shift)
        return torch.tensor(A_idx_0,device=self.device),torch.tensor(B_idx_0,device=self.device),torch.tensor(C_idx_0,device=self.device)
    
    
    def chern_number_quick(self,radius_factor=(2.6,2.6),U1=True,shift=(0,0),selfaverage=False,):
        # st=time.time()
        if selfaverage:
            return torch.stack([self.chern_number_quick(shift=(i,j)) for i in range(self.Lx) for j in range(self.Ly)]).mean()
        else:
            A_idx,B_idx,C_idx = self.generate_tripartite_circle(shift=shift,radius_factor=radius_factor)
            P=(torch.eye(self.C_m.shape[0],device=self.device,dtype=self.dtype_complex)-1j*self.C_m)/2
            P_AB=P[A_idx[:,None],B_idx[None,:]]
            P_BC=P[B_idx[:,None],C_idx[None,:]]
            P_CA=P[C_idx[:,None],A_idx[None,:]]
            P_AC=P[A_idx[:,None],C_idx[None,:]]
            P_CB=P[C_idx[:,None],B_idx[None,:]]
            P_BA=P[B_idx[:,None],A_idx[None,:]]
            h=-12*torch.pi*(torch.einsum("jk,kl,lj->jkl",P_AB,P_BC,P_CA)-torch.einsum("jl,lk,kj->jkl",P_AC,P_CB,P_BA)).imag
            # assert np.abs(h.imag).max()<1e-10, "Imaginary part of h is too large"
            nu=h.sum()
        # print('Chern number done in {:.4f}'.format(time.time()-st))
            if U1:
                return nu/2
            else:
                return nu

    def local_Chern_marker(self,Gamma,shift=[0,0],n_maj=2,U1=True):
        replica,layer,x,y,orbit,maj = np.unravel_index(np.arange(Gamma.shape[0]),(self.replica,self.layer,self.Lx,self.Ly,self.orbit,n_maj))
        x = torch.tensor((x+shift[0])%self.Lx,device=self.device)
        y = torch.tensor((y+shift[1])%self.Ly,device=self.device)
        C_f = self.get_C_f(Gamma,normal=False)
        xy_comm = torch.einsum("ij,j,jk,k,ki->i",C_f,x,C_f,y,C_f) - torch.einsum("ij,j,jk,k,ki->i",C_f,y,C_f,x,C_f)
        C_r = (xy_comm * 2 * torch.pi* 1j)
        C_r=C_r.reshape((self.replica,self.layer,self.Lx,self.Ly,self.orbit,n_maj))
        # assert np.abs(C_r.imag).max()<1e-10, f'imaginary part is {C_r.imag.max()}'
        if U1:
            return C_r.sum(axis=(-1,-2)).real/2
        else:
            return C_r.sum(axis=(-1,-2)).real

    def entanglement_contour(self,subregion,fermion=False, Gamma=None, fermion_idx=True,n=1):
        # c_A=self.c_subregion_m(subregion)
        c_A=self.c_subregion_m(subregion,Gamma,fermion_idx=fermion_idx)
        C_f=(torch.eye(c_A.shape[0],device=self.device)+1j*c_A)/2
        if n==1:
            f=self.xlogx(C_f,)
        if fermion:
            return torch.diag(f).real.reshape((-1,2)).sum(axis=1)
        else:
            return torch.diag(f).real
        
    def c_subregion_m(self,subregion,Gamma=None,fermion_idx=True):
        if Gamma is None:
            Gamma=self.C_m
        if fermion_idx:
            raise NotImplementedError("fermion_idx=True not supported in GTN2_torch; use linearize_idx_span and pass fermion_idx=False")
            # subregion=self.linearize_index(subregion,2)
        # return Gamma[np.ix_(subregion,subregion)]
        return Gamma[subregion[:,None],subregion[None,:]]
        
    def von_Neumann_entropy_m(self,subregion,Gamma=None,fermion_idx=True,verbose=False):
        """Von Neumann entropy S_A from the Majorana covariance matrix Gamma.
        
        Computes S_A = sum_{k=1}^{N} h((1+nu_k)/2) where nu_k are the N positive
        eigenvalues of i*Gamma_A and h(x) = -x ln x - (1-x) ln(1-x).
        
        Reference: Peschel, J. Phys. A 36, L205 (2003); Peschel & Eisler,
        J. Phys. A 42, 504003 (2009).
        """
        st=time.time()
        c_A=self.c_subregion_m(subregion,Gamma,fermion_idx=fermion_idx)
        val=torch.linalg.eigvalsh(1j*c_A)
        val=(1-val)/2  
        val = val[(val>0) & (val<1)]
        if verbose:
            print('entanglement entropy done in {:.4f}'.format(time.time()-st))
        return (-torch.sum(val*torch.log(val))-torch.sum((1-val)*torch.log(1-val)))/2

    def half_cut_entanglement_entropy(self,shift=(0,0),selfaverage=False):
        """ this function is inefficient because it involves many redundant calculations"""
        if selfaverage:
            return torch.stack([self.half_cut_entanglement_entropy(shift=(i,j)) for i in range(self.Lx) for j in range(self.Ly)]).mean()
        else:
            Lx_first_half = (np.arange(self.Lx//2) + shift[0])%self.Lx
            Ly_first_half = (np.arange(self.Ly//2) +shift[1])%self.Ly
            Lx_second_half = (np.arange(self.Lx//2,self.Lx) + shift[0])%self.Lx
            Ly_second_half = (np.arange(self.Ly//2,self.Ly) + shift[1])%self.Ly
            subA=self.c2g(ilist=Lx_first_half,jlist=Ly_first_half)
            subB=self.c2g(ilist=Lx_first_half,jlist=Ly_second_half)
            subC=self.c2g(ilist=Lx_second_half,jlist=Ly_first_half)
            # subD=self.c2g(ilist=Lx_second_half,jlist=Ly_second_half)
            SAB=self.von_Neumann_entropy_m(torch.cat([subA,subB]),fermion_idx=False)
            SAC=self.von_Neumann_entropy_m(torch.cat([subA,subC]),fermion_idx=False)
            return (SAB+SAC)/2

    def half_cut_entanglement_y_entropy(self,shift=(0,0),selfaverage=False):
        """half cut entanglement entropy with Lx x Ly/2"""
        if selfaverage:
            return torch.stack([self.half_cut_entanglement_y_entropy(shift=(0,j)) for j in range(self.Ly)]).mean()
        else:
            Lx_ = (np.arange(self.Lx))
            Ly_first_half = (np.arange(self.Ly//2) +shift[1])%self.Ly
            subA=self.c2g(ilist=Lx_,jlist=Ly_first_half)
            SA=self.von_Neumann_entropy_m(subA,fermion_idx=False)
            return SA
    
    def entanglement_y_entropy(self,ly,shift=(0,0),selfaverage=False):
        """half cut entanglement entropy with Lx x Ly/2"""
        if selfaverage:
            return torch.stack([self.entanglement_y_entropy(ly=ly,shift=(0,j)) for j in range(self.Ly)]).mean()
        else:
            Lx_ = (np.arange(self.Lx))
            Ly_first_half = (np.arange(ly) +shift[1])%self.Ly
            subA=self.c2g(ilist=Lx_,jlist=Ly_first_half)
            SA=self.von_Neumann_entropy_m(subA,fermion_idx=False)
            return SA
            
    def half_cut_entanglement_x_entropy(self,shift=(0,0),selfaverage=False):
        if selfaverage:
            return torch.stack([self.half_cut_entanglement_x_entropy(shift=(i,0)) for i in range(self.Lx)]).mean()
        else:
            Ly_ = (np.arange(self.Ly))
            Lx_first_half = (np.arange(self.Lx//2) + shift[0])%self.Lx
            subA=self.c2g(ilist=Lx_first_half,jlist=Ly_)
            SA=self.von_Neumann_entropy_m(subA,fermion_idx=False)
            return SA


    def tripartite_mutual_information(self,shift=(0,0),selfaverage=False):
        """
        TMI uses four quadrants, covers both layer, [Assuming only one 1 replica]
        compute the tripartite mutual information as S(A)+S(B)+S(C)-S(AB)-S(BC)-S(AC)+S(ABC)
        """
        assert self.replica==1, "Tripartite mutual information only works for one replica"
        if selfaverage:
            return torch.stack([self.tripartite_mutual_information(shift=(i,j)) for i in range(self.Lx) for j in range(self.Ly)]).mean()
        else:
            Lx_first_half = (np.arange(self.Lx//2) + shift[0])%self.Lx
            Ly_first_half = (np.arange(self.Ly//2) +shift[1])%self.Ly
            Lx_second_half = (np.arange(self.Lx//2,self.Lx) + shift[0])%self.Lx
            Ly_second_half = (np.arange(self.Ly//2,self.Ly) + shift[1])%self.Ly
            subA=self.c2g(ilist=Lx_first_half,jlist=Ly_first_half)
            subB=self.c2g(ilist=Lx_first_half,jlist=Ly_second_half)
            subC=self.c2g(ilist=Lx_second_half,jlist=Ly_first_half)
            subD=self.c2g(ilist=Lx_second_half,jlist=Ly_second_half)

            SA=self.von_Neumann_entropy_m(subA,fermion_idx=False)
            SB=self.von_Neumann_entropy_m(subB,fermion_idx=False)
            SC=self.von_Neumann_entropy_m(subC,fermion_idx=False)
            SAB=self.von_Neumann_entropy_m(torch.cat([subA,subB]),fermion_idx=False)
            SBC=self.von_Neumann_entropy_m(torch.cat([subB,subC]),fermion_idx=False)
            SAC=self.von_Neumann_entropy_m(torch.cat([subA,subC]),fermion_idx=False)
            # SABC=self.von_Neumann_entropy_m(torch.cat([subA,subB,subC]),fermion_idx=False)
            SABC=self.von_Neumann_entropy_m(subD,fermion_idx=False)
            return SA+SB+SC-SAB-SBC-SAC+SABC

    def tripartite_mutual_information_quasi_1d(self,shift=(0,0),selfaverage=False):
        """
        TMI uses four quadrants, covers both layer, [Assuming only one 1 replica]
        compute the tripartite mutual information as S(A)+S(B)+S(C)-S(AB)-S(BC)-S(AC)+S(ABC)
        """
        assert self.replica==1, "Tripartite mutual information only works for one replica"
        if selfaverage:
            return torch.stack([self.tripartite_mutual_information_quasi_1d(shift=(i,0)) for i in range(self.Lx)]).mean()
        else:
            Ly_ = (np.arange(self.Ly))
            Lx_first_ = (np.arange(self.Lx//4) + shift[0])%self.Lx
            subA=self.c2g(ilist=Lx_first_,jlist=Ly_)
            Lx_second_ = (np.arange(self.Lx//4)+self.Lx//4 + shift[0])%self.Lx
            subB=self.c2g(ilist=Lx_second_,jlist=Ly_)
            Lx_third_ = (np.arange(self.Lx//4)+self.Lx//2 + shift[0])%self.Lx
            subC=self.c2g(ilist=Lx_third_,jlist=Ly_)
            Lx_fourth_ = (np.arange(self.Lx//4)+self.Lx//4*3 + shift[0])%self.Lx
            subD=self.c2g(ilist=Lx_fourth_,jlist=Ly_)

            SA=self.von_Neumann_entropy_m(subA,fermion_idx=False)
            SB=self.von_Neumann_entropy_m(subB,fermion_idx=False)
            SC=self.von_Neumann_entropy_m(subC,fermion_idx=False)
            SAB=self.von_Neumann_entropy_m(torch.cat([subA,subB]),fermion_idx=False)
            SBC=self.von_Neumann_entropy_m(torch.cat([subB,subC]),fermion_idx=False)
            SAC=self.von_Neumann_entropy_m(torch.cat([subA,subC]),fermion_idx=False)
            SABC=self.von_Neumann_entropy_m(subD,fermion_idx=False)
            return SA+SB+SC-SAB-SBC-SAC+SABC

    def bipartite_mutual_information(self,shift=(0,0),selfaverage=False):
        assert self.replica==1, "Bipartite mutual information only works for one replica"
        if selfaverage:
            return torch.stack([self.bipartite_mutual_information(shift=(i,j)) for i in range(self.Lx) for j in range(self.Ly)]).mean()
        else:
            # # A, C are diagonal
            Lx_first_half = (np.arange(self.Lx//2) + shift[0])%self.Lx
            Ly_first_half = (np.arange(self.Ly//2) +shift[1])%self.Ly
            Lx_second_half = (np.arange(self.Lx//2,self.Lx) + shift[0])%self.Lx
            Ly_second_half = (np.arange(self.Ly//2,self.Ly) + shift[1])%self.Ly
            subA=self.c2g(ilist=Lx_first_half,jlist=Ly_first_half)
            subC=self.c2g(ilist=Lx_second_half,jlist=Ly_first_half)


            SA=self.von_Neumann_entropy_m(subA,fermion_idx=False)
            SC=self.von_Neumann_entropy_m(subC,fermion_idx=False)
            SAC=self.von_Neumann_entropy_m(torch.cat([subA,subC]),fermion_idx=False)
            return SA+SC-SAC
    
    def bipartite_mutual_information_quasi_1d(self,shift=(0,0),selfaverage=False,partition=4):
        assert self.replica==1, "Bipartite mutual information only works for one replica"
        if selfaverage:
            return torch.stack([self.bipartite_mutual_information_quasi_1d(shift=(i,0)) for i in range(self.Lx)]).mean()
        else:
            # treat it as a quasi-1D
            Ly_ = (np.arange(self.Ly))
            Lx_first_ = (np.arange(self.Lx//partition) + shift[0])%self.Lx
            subA=self.c2g(ilist=Lx_first_,jlist=Ly_)
            Lx_third_ = (np.arange(self.Lx//partition)+self.Lx//2 + shift[0])%self.Lx
            subC=self.c2g(ilist=Lx_third_,jlist=Ly_)

            SA=self.von_Neumann_entropy_m(subA,fermion_idx=False)
            SC=self.von_Neumann_entropy_m(subC,fermion_idx=False)
            SAC=self.von_Neumann_entropy_m(torch.cat([subA,subC]),fermion_idx=False)
            return SA+SC-SAC
    

    def mutual_information_quasi_1d_crossratio(self, a, shift=0, selfaverage=False, Gamma=None, layer=None):
        """
        Bipartite and tripartite mutual information for four contiguous intervals
        on the ring (y-direction), all spanning [0,Lx).
        A = [ly, ly+a),  B = [ly+a, ly+Ly//2),  C = [ly+Ly//2, ly+Ly//2+a),  D = complement

        BMI = S(A) + S(C) - S(AC)
        TMI = S(A)+S(B)+S(C) - S(AB)-S(AC)-S(BC) + S(ABC),  S(ABC)=S(D) by purity

        Returns (BMI, TMI, eta) where eta = z13*z24 / (z14*z23).
        """
        assert self.replica == 1, "Mutual information only works for one replica"
        assert 1 <= a < self.Ly // 2, f"a must be in [1, Ly//2), got a={a}"

        if selfaverage:
            results = [
                self.mutual_information_quasi_1d_crossratio(a, shift=ly, selfaverage=False, Gamma=Gamma, layer=layer)
                for ly in range(self.Ly)
            ]
            BMI = torch.stack([r[0] for r in results]).mean()
            TMI = torch.stack([r[1] for r in results]).mean()
        else:
            ly = shift
            Lx_ = np.arange(self.Lx)

            jlist_A = (np.arange(a) + ly) % self.Ly
            jlist_B = (np.arange(self.Ly // 2 - a) + ly + a) % self.Ly
            jlist_C = (np.arange(a) + ly + self.Ly // 2) % self.Ly
            jlist_D = (np.arange(self.Ly - self.Ly // 2 - a) + ly + self.Ly // 2 + a) % self.Ly

            subA = self.c2g(ilist=Lx_, jlist=jlist_A, layer=layer)
            subB = self.c2g(ilist=Lx_, jlist=jlist_B, layer=layer)
            subC = self.c2g(ilist=Lx_, jlist=jlist_C, layer=layer)
            subD = self.c2g(ilist=Lx_, jlist=jlist_D, layer=layer)

            SA = self.von_Neumann_entropy_m(subA, Gamma=Gamma, fermion_idx=False)
            SB = self.von_Neumann_entropy_m(subB, Gamma=Gamma, fermion_idx=False)
            SC = self.von_Neumann_entropy_m(subC, Gamma=Gamma, fermion_idx=False)
            SAB = self.von_Neumann_entropy_m(torch.cat([subA, subB]), Gamma=Gamma, fermion_idx=False)
            SAC = self.von_Neumann_entropy_m(torch.cat([subA, subC]), Gamma=Gamma, fermion_idx=False)
            SBC = self.von_Neumann_entropy_m(torch.cat([subB, subC]), Gamma=Gamma, fermion_idx=False)
            SABC = self.von_Neumann_entropy_m(subD, Gamma=Gamma, fermion_idx=False)

            BMI = SA + SC - SAC
            TMI = SA + SB + SC - SAB - SAC - SBC + SABC

        # Cross ratio (geometric, independent of shift)
        z = lambda dy: self.Ly / np.pi * np.sin(np.pi * dy / self.Ly)
        z13 = z(self.Ly // 2)
        z24 = z(self.Ly // 2)
        z14 = z(self.Ly // 2 + a)
        z23 = z(self.Ly // 2 - a)
        eta = (z13 * z24) / (z14 * z23)

        return BMI, TMI, eta

    def c2g(self,ilist,jlist,layer=None):
        if layer is not None:
            return torch.from_numpy(self.linearize_idx_span(ilist=ilist,jlist=jlist,layer=layer)).to(self.device)
        if self.layer == 1:
            return torch.from_numpy(self.linearize_idx_span(ilist = ilist,jlist=jlist,layer=0)).to(self.device)
        elif self.layer==2:
            return torch.hstack((
                torch.from_numpy(self.linearize_idx_span(ilist = ilist,jlist=jlist,layer=0)).to(self.device),
                torch.from_numpy(self.linearize_idx_span(ilist = ilist,jlist=jlist,layer=1)).to(self.device))
            )

    def xlogx(self,A):
        val,vec=torch.linalg.eigh(A)
        negative=val<=0
        val[negative]=0
        val_pos=val[~negative]
        val[~negative]=-val_pos*torch.log(val_pos)
        val=val+0j
        return vec@torch.diag(val)@vec.conj().T

    def C_m_selfaverage(self,n=1):
        """take the selfaverage of Gamma, by shifting all possible coordinates"""
        idx=(self.replica, self.layer, self.Lx*self.Ly,self.orbit,2)
        C_m_reshape=(self.C_m**n).view(idx*2)
        d_list =[1]*len(idx*2)
        Lxy=self.Lx*self.Ly
        d_list[2]=d_list[7]=Lxy
        kernel = torch.eye(Lxy,device=self.device).reshape(d_list)/(Lxy)
        return torch.fft.ifft2(torch.fft.fft2(C_m_reshape,dim=(2,7))*torch.fft.fft2(kernel,dim=(2,7)),dim=(2,7)).reshape(self.C_m.shape).real

    def _hamiltonian(self,mu, Delta=1., t=1):
        sigmax=torch.tensor([[0, 1.], [1., 0]],dtype=self.dtype_float,device=self.device)
        sigmay=torch.tensor([[0, -1j], [1j, 0]],dtype=self.dtype_complex,device=self.device)
        sigmaz=torch.tensor([[1., 0], [0, -1.]],dtype=self.dtype_float,device=self.device)
        hopx = torch.diag(torch.ones(self.Lx-1,dtype=self.dtype_float,device=self.device), -1)
        hopx[0, -1] = self.bcx
        hopy = torch.diag(torch.ones(self.Ly-1,dtype=self.dtype_float,device=self.device), -1)
        hopy[0, -1] = self.bcy
        hopxmat = torch.kron(hopx,torch.eye(self.Ly,dtype=self.dtype_float,device=self.device))
        hopymat = torch.kron(torch.eye(self.Lx,dtype=self.dtype_float,device=self.device),hopy)
        # hopxmat = torch.kron(torch.eye(self.Ly,dtype=self.dtype_float,device=self.device),hopx)
        # hopymat = torch.kron(hopy,torch.eye(self.Lx,dtype=self.dtype_float,device=self.device))
        if isinstance(mu, (int, float)):
            mu_mat = mu * torch.eye(self.Lx*self.Ly,dtype=self.dtype_float,device=self.device)
        else:
            mu_tensor = torch.as_tensor(mu, dtype=self.dtype_float, device=self.device)
            if mu_tensor.ndim == 1 and mu_tensor.shape[0] == self.Lx:
                mu_tensor = mu_tensor.repeat_interleave(self.Ly)
            mu_mat = torch.diag(mu_tensor)
        return ((torch.kron(hopxmat-hopxmat.T, sigmax)+torch.kron(hopymat-hopymat.T, sigmay))* 1j*Delta-t*torch.kron(hopxmat+hopxmat.T+hopymat+hopymat.T, sigmaz))/2+torch.kron(mu_mat, sigmaz)

    def bandstructure(self,mu):
        h = self._hamiltonian(mu).conj()
        val, vec = torch.linalg.eigh(h)
        sortindex = torch.argsort(val)
        val = val[sortindex]
        vec = vec[:, sortindex]
        return val, vec

    def covariance_matrix(self, mu):
        '''
        G_{ij}=<f_i^dagger f_j>
        '''
        val,vec=self.bandstructure(mu=mu)
        C_f = torch.einsum("ij,j,jk",vec ,(val<0).float(),vec.T.conj())
        C_m = self.get_C_m(C_f)
        return C_m
    


    



def amplitude(nshell,nkx=500,nky=500,tau=[0,1],mu=1,geometry = 'square', lower=True, C=1):
    """if gemoetry is square, then the shape is [i-nshell,i+nshell]x[j-nshell,j+nshell]"""
    
    kx = np.linspace(-np.pi,np.pi,nkx)
    ky = np.linspace(-np.pi,np.pi,nky)
    KX,KY = np.meshgrid(kx,ky, indexing='ij')
    offdiag=(np.sin(KX)-1j*np.sin(KY))**C
    dx = offdiag.real
    dy = -offdiag.imag
    dz = mu-np.cos(KX)-np.cos(KY)
    E = np.sqrt(dx**2+dy**2+dz**2)
    cos_theta = dz/E
    sin_theta_exp_iphi = (dx+1j*dy)/E
    tau = np.array(tau)/np.linalg.norm(tau)
    if lower:
        ak = (1-cos_theta)/2*tau[0] - 1/2*sin_theta_exp_iphi.conj()*tau[1]
        bk = - 1/2*sin_theta_exp_iphi*tau[0] + (1+cos_theta)/2*tau[1]
    else:
        ak = (1+cos_theta)/2*tau[0] + 1/2*sin_theta_exp_iphi.conj()*tau[1]
        bk = 1/2*sin_theta_exp_iphi*tau[0] + (1-cos_theta)/2*tau[1]

    # ak=-1/2*(dx-1j*dy)/E
    # bk=1/2+dz/(2*E)

    def a_nint(i,j):
        a_int=ak * np.exp(1j * (KX * i + KY*j))
        return np.trapz(np.trapz(a_int,kx),ky)/(2*np.pi)**2
    def b_nint(i,j):
        b_int=bk * np.exp(1j * (KX * i + KY*j))
        return np.trapz(np.trapz(b_int,kx),ky)/(2*np.pi)**2

    if geometry == 'square':
        i_list = np.arange(-nshell,nshell+1)
        j_list = np.arange(-nshell,nshell+1)
        ij_list = [(i,j) for i in i_list for j in j_list]
    elif geometry == 'diamond':
        ij_list=[(i,j) for i in range(-nshell,nshell+1) for j in range(-nshell+abs(i),nshell+1 -abs(i))]
    a_i = {(i,j):a_nint(i,j) for i,j in ij_list}
    b_i = {(i,j):b_nint(i,j) for i,j in ij_list}
    return a_i,b_i

def amplitude_fft(nkx=5000,nky=5000,tau=[0,1],mu=1, lower=True, C=1):
    """if gemoetry is square, then the shape is [i-nshell,i+nshell]x[j-nshell,j+nshell]"""
    
    # kx = np.linspace(-np.pi,np.pi,nkx,endpoint=False)
    # ky = np.linspace(-np.pi,np.pi,nky,endpoint=False)
    kx = np.linspace(0,2*np.pi,nkx,endpoint=False)
    ky = np.linspace(0,2*np.pi,nky,endpoint=False)
    KX,KY = np.meshgrid(kx,ky, indexing='ij')
    offdiag=(np.sin(KX)-1j*np.sin(KY))**C
    dx = offdiag.real
    dy = -offdiag.imag
    dz = mu-np.cos(KX)-np.cos(KY)
    E = np.sqrt(dx**2+dy**2+dz**2+1e-18)
    cos_theta = dz/E
    sin_theta_exp_iphi = (dx+1j*dy)/E
    tau = np.array(tau)/np.linalg.norm(tau)
    if lower:
        ak = (1-cos_theta)/2*tau[0] - 1/2*sin_theta_exp_iphi.conj()*tau[1]
        bk = - 1/2*sin_theta_exp_iphi*tau[0] + (1+cos_theta)/2*tau[1]
    else:
        ak = (1+cos_theta)/2*tau[0] + 1/2*sin_theta_exp_iphi.conj()*tau[1]
        bk = 1/2*sin_theta_exp_iphi*tau[0] + (1-cos_theta)/2*tau[1]

    a_i = np.fft.fft2(ak.conj())/(nkx*nky)
    b_i = np.fft.fft2(bk.conj())/(nkx*nky)
    # conjugation due to the opposite definition of exponent of FT pahse
    return a_i,b_i

def amplitude_fft_gpu(device,nkx=5000,nky=5000,tau=[0,1],mu=1, lower=True, C=1):
    """if gemoetry is square, then the shape is [i-nshell,i+nshell]x[j-nshell,j+nshell]"""
    
    # kx = np.linspace(-np.pi,np.pi,nkx,endpoint=False)
    # ky = np.linspace(-np.pi,np.pi,nky,endpoint=False)
    kx = torch.linspace(0.,2*torch.pi,nkx+1,device=device)[:-1]
    ky = torch.linspace(0.,2*torch.pi,nky+1,device=device)[:-1]
    KX,KY = torch.meshgrid(kx,ky, indexing='ij')
    offdiag=(torch.sin(KX)-1j*torch.sin(KY))**C
    dx = offdiag.real
    dy = -offdiag.imag
    dz = mu-torch.cos(KX)-torch.cos(KY)
    E = torch.sqrt(dx**2+dy**2+dz**2+1e-18)
    cos_theta = dz/E
    sin_theta_exp_iphi = (dx+1j*dy)/E
    tau = np.array(tau)/np.linalg.norm(tau)
    if lower:
        ak = (1-cos_theta)/2*tau[0] - 1/2*sin_theta_exp_iphi.conj()*tau[1]
        bk = - 1/2*sin_theta_exp_iphi*tau[0] + (1+cos_theta)/2*tau[1]
    else:
        ak = (1+cos_theta)/2*tau[0] + 1/2*sin_theta_exp_iphi.conj()*tau[1]
        bk = 1/2*sin_theta_exp_iphi*tau[0] + (1-cos_theta)/2*tau[1]

    a_i = torch.fft.fft2(ak.conj())/(nkx*nky)
    b_i = torch.fft.fft2(bk.conj())/(nkx*nky)
    # conjugation due to the opposite definition of exponent of FT pahse
    return a_i,b_i

def amplitude_fft_nshell(nshell,nkx=500,nky=500,tau=[0,1],mu=1,geometry = 'square', lower=True, C=1):
    if geometry == 'square':
        i_list = np.arange(-nshell,nshell+1)
        j_list = np.arange(-nshell,nshell+1)
        ij_list = [(i,j) for i in i_list for j in j_list]
    elif geometry == 'diamond':
        ij_list=[(i,j) for i in range(-nshell,nshell+1) for j in range(-nshell+abs(i),nshell+1 -abs(i))]
    a_i,b_i = amplitude_fft(nkx,nky,tau,mu, lower, C)
    a_i = {(i,j):a_i[(-i)%nkx,(j)%nky] for i,j in ij_list}
    b_i = {(i,j):b_i[(-i)%nkx,(j)%nky] for i,j in ij_list}
    return a_i,b_i

def amplitude_fft_nshell_gpu(nshell,device,nkx=500,nky=500,tau=[0,1],mu=1,geometry = 'square', lower=True, C=1):
    if geometry == 'square':
        i_list = np.arange(-nshell,nshell+1)
        j_list = np.arange(-nshell,nshell+1)
        ij_list = [(i,j) for i in i_list for j in j_list]
    elif geometry == 'diamond':
        ij_list=[(i,j) for i in range(-nshell,nshell+1) for j in range(-nshell+abs(i),nshell+1 -abs(i))]
    a_i,b_i = amplitude_fft_gpu(device,nkx,nky,tau,mu, lower, C)
    a_i = {(i,j):a_i[(-i)%nkx,(j)%nky].item() for i,j in ij_list}
    b_i = {(i,j):b_i[(-i)%nkx,(j)%nky].item() for i,j in ij_list}
    return a_i,b_i