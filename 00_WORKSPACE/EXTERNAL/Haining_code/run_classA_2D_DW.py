# This script is used to compute the domain wall dynamics of the 2D class A system
from GTN2_torch import *
import torch
import argparse
from tqdm import tqdm
import time
from utils_torch import *



def spiral_order(Lx, Ly):
    """Counter-clockwise spiral starting from (0,0) with PBC."""
    visited = set()
    order = []
    x, y = 0, 0
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # right, up, left, down
    visited.add((0, 0))
    order.append((0, 0))
    step_size = 1
    dir_idx = 0
    while len(order) < Lx * Ly:
        dx, dy = directions[dir_idx % 4]
        for _ in range(step_size):
            x = (x + dx) % Lx
            y = (y + dy) % Ly
            if (x, y) not in visited:
                visited.add((x, y))
                order.append((x, y))
            if len(order) >= Lx * Ly:
                break
        dir_idx += 1
        if dir_idx % 2 == 0:
            step_size += 1
    return order


def measure_feedback_layer_dw_line(gtn2_torch,overlap,geometry,truncate=False, mu=[1,3], order='raster'):
    """
    mu = [mu_inner, mu_outer]
    For square, inner is the region within the strip , typically reserved for topological
    """
    ilist = range(gtn2_torch.Lx)
    jlist = range(gtn2_torch.Ly)
    if order == 'spiral':
        ij_list = spiral_order(gtn2_torch.Lx, gtn2_torch.Ly)
    else:
        ij_list = [(i,j) for i in (ilist) for j in (jlist)]
    margin=0 if overlap else gtn2_torch.nshell
    inner_list = range(gtn2_torch.Lx//5+margin,gtn2_torch.Lx//5*4-margin)
    outer_list = range(gtn2_torch.Lx//5-margin,gtn2_torch.Lx//5*4+margin)
    if geometry == 'square':
        inner_region=set([(i,j) for i in inner_list for j in inner_list])
        outer_region=set([(i,j) for i in ilist for j in jlist if i not in outer_list or j not in outer_list])
    elif geometry == 'strip':
        inner_region=set([(i,j) for i in inner_list for j in jlist])
        outer_region=set([(i,j) for i in ilist for j in jlist if i not in outer_list])
    region_inner=inner_region if truncate else None
    region_outer=outer_region if truncate else None
    # for i,j in tqdm(ij_list,desc='measure with feedback'):
    for i,j in ij_list:
        if (i,j) in inner_region:
            gtn2_torch.measure_feedback(ij = [i,j],mu=mu[0],region=region_inner,tau=(1,1))
            gtn2_torch.measure_feedback(ij = [i,j],mu=mu[0],region=region_inner,tau=(1,-1))
        elif (i,j) in outer_region:
            gtn2_torch.measure_feedback(ij = [i,j],mu=mu[1],region=region_outer,tau=(1,1))
            gtn2_torch.measure_feedback(ij = [i,j],mu=mu[1],region=region_outer,tau=(1,-1))



def randomize(gtn2_torch,measure=True):
    # for i in tqdm(range(2*gtn2_torch.L+1,4*gtn2_torch.L,2),desc='randomize'):
    for i in range(2*gtn2_torch.L+1,4*gtn2_torch.L,2):
        # print([i, (i+1)%(2*gtn2_torch.L)+2*gtn2_torch.L])
        gtn2_torch.randomize([i, (i+1)%(2*gtn2_torch.L)+2*gtn2_torch.L])
    if measure:
        # for i in tqdm(range(2*gtn2_torch.L,4*gtn2_torch.L,2),desc='measure'):
        for i in range(2*gtn2_torch.L,4*gtn2_torch.L,2):
            gtn2_torch.measure_single_mode_Born([i,i+1],mode=[1])

def dummy(inputs):
    L,nshell,tf,truncate,seed=inputs
    gtn2_torch=GTN2_torch(Lx=L,Ly=L,history=False,random_init=False,random_U1=True,bcx=1,bcy=1,seed=seed,orbit=2,nshell=nshell,layer=2,replica=1,complex128=True,gpu=False)
    mu_list=[1,3]
    tau_list=[(1,1),(1,-1)]
    gtn2_torch.a_i={}
    gtn2_torch.b_i={}
    gtn2_torch.A_i={}
    gtn2_torch.B_i={}
    for mu in mu_list:
        for tau in tau_list:
            gtn2_torch.a_i[mu,tau],gtn2_torch.b_i[mu,tau] = amplitude_fft_nshell_gpu(gtn2_torch.nshell,gtn2_torch.device,tau=tau,geometry='square',lower=True,mu=mu,nkx=L,nky=L)
            gtn2_torch.A_i[mu,tau],gtn2_torch.B_i[mu,tau] = amplitude_fft_nshell_gpu(gtn2_torch.nshell,gtn2_torch.device,tau=tau,geometry='square',lower=False,mu=mu,nkx=L,nky=L)
    return gtn2_torch

def run(inputs):
    L,nshell,tf,truncate,seed=inputs
    gtn2_torch=GTN2_torch(Lx=L,Ly=L,history=False,random_init=False,random_U1=False,bcx=1,bcy=1,seed=seed,orbit=2,nshell=nshell,layer=2,replica=1,gpu=False)
    # mu_list=[1,3]
    # tau_list = [(1,1),(1,-1)]

    gtn2_torch.a_i = gtn2_dummy.a_i
    gtn2_torch.b_i = gtn2_dummy.b_i
    gtn2_torch.A_i = gtn2_dummy.A_i
    gtn2_torch.B_i = gtn2_dummy.B_i


    
    ilist=np.arange(0,gtn2_torch.Lx)
    jlist=np.arange(0,gtn2_torch.Ly)
    subregion_m = torch.hstack((
        torch.from_numpy(gtn2_torch.linearize_idx_span(ilist = ilist,jlist=jlist,layer=0)).cuda(),
        torch.from_numpy(gtn2_torch.linearize_idx_span(ilist = ilist,jlist=jlist,layer=1)).cuda())
    )
    # EC_list=[]
    # C_r_list=[]
    C_m_history[0] += gtn2_torch.C_m

    # EC_list.append( gtn2_torch.entanglement_contour(subregion_m,fermion_idx=False,Gamma=gtn2_torch.C_m).reshape((2,ilist.shape[0],jlist.shape[0],2,2)).sum(axis=(-1,-2))) 
    # C_r_list.append( gtn2_torch.local_Chern_marker(gtn2_torch.C_m,))


    for i in tqdm(range(tf*gtn2_torch.Lx)):
        print(i)
        measure_feedback_layer_dw_line(gtn2_torch,overlap=True,geometry='strip',truncate=truncate)
        randomize(gtn2_torch,measure=True)
        
        # EC_list.append( gtn2_torch.entanglement_contour(subregion_m,fermion_idx=False,Gamma=gtn2_torch.C_m).reshape((2,ilist.shape[0],jlist.shape[0],2,2)).sum(axis=(-1,-2))) 
        # C_r_list.append( gtn2_torch.local_Chern_marker(gtn2_torch.C_m,))
        C_m_history[i+1] += gtn2_torch.C_m
    
    # return {'EC_list':EC_list,'C_r_list':C_r_list,'C_m_history':C_m_history}
    return {'C_m_history':C_m_history}

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--L','-L',type=int)
    parser.add_argument('--nshell','-nshell',type=int,default=2)
    parser.add_argument('--es','-es',type=int,default=10)
    parser.add_argument('--seed0','-seed0',type=int,default=0)
    parser.add_argument('--tf','-tf',type=int,default=20)
    parser.add_argument('--truncate','-truncate',action='store_true')
    args=parser.parse_args()

    st=time.time()
    inputs=[(args.L, args.nshell, args.tf, args.truncate,seed+args.seed0) for seed in range(args.es)]
    # EC_list = []
    # C_r_list = []
    upper_list = []
    lower_list = []
    bulk_list = []
    gtn2_dummy=dummy(inputs[0])
    C_m_history = torch.zeros((args.tf*gtn2_dummy.Lx+1,*gtn2_dummy.C_m.shape),dtype=torch.float64,device=gtn2_dummy.device)

    for inp in inputs:
        rs= run(inp)
        # EC_list.append(rs['EC'])
        # C_r_list.append(rs['C_r'])
    
    for C_m in C_m_history:
        C_m.div_(args.es)
    #     C_m_sq = C_m**2
    #     upper_list.append( correlation_i_length(C_m_sq,replica=1,layer=2,Lx=gtn2_dummy.Lx,Ly=gtn2_dummy.Ly,i0=gtn2_dummy.Lx//5))
    #     lower_list.append(correlation_i_length(C_m_sq,replica=1,layer=2,Lx=gtn2_dummy.Lx,Ly=gtn2_dummy.Ly,i0=gtn2_dummy.Lx//5*4-1))
    #     bulk_list.append( correlation_i_length(C_m_sq,replica=1,layer=2,Lx=gtn2_dummy.Lx,Ly=gtn2_dummy.Ly,i0=gtn2_dummy.Lx//2))

    # fn=f'class_A_2D_L{args.L}_nshell{args.nshell}_es{args.es}_seed{args.seed0}_tf{args.tf}{"_truncate" if args.truncate else ""}.pt'
    fn=f'class_A_2D_L{args.L}_nshell{args.nshell}_es{args.es}_seed{args.seed0}_tf{args.tf}{"_truncate" if args.truncate else ""}_raw_Cm.pt'
    torch.save({
    # 'EC':EC_list,
    # 'C_r':C_r_list,
    # 'upper':upper_list,
    # 'lower':lower_list,
    # 'bulk':bulk_list,
    'C_m':C_m_history,
    'args':args},fn)
    print('Time elapsed: {:.4f}'.format(time.time()-st))



