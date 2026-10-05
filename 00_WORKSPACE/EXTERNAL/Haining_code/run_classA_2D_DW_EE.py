# This script is used to compute the entanglement entropy of the 2D class A system with domain wall
from GTN2_torch import *
import torch
import argparse
from tqdm import tqdm
import time
from utils_torch import *
from run_classA_2D_DW import measure_feedback_layer_dw_line, randomize

MODE_MAP = {
    'avoided':   {'overlap': False, 'truncate': False},
    'truncated': {'overlap': True,  'truncate': True},
    'overlap':   {'overlap': True,  'truncate': False},
}

def dummy(inputs):
    Lx,Ly,nshell,tf,mode,mu,order,seed=inputs
    gtn2_torch=GTN2_torch(Lx=Lx,Ly=Ly,history=False,random_init=False,random_U1=True,bcx=1,bcy=1,seed=seed,orbit=2,nshell=nshell,layer=2,replica=1,complex128=True,gpu=True)
    tau_list=[(1,1),(1,-1)]
    gtn2_torch.a_i={}
    gtn2_torch.b_i={}
    gtn2_torch.A_i={}
    gtn2_torch.B_i={}
    for m in mu:
        for tau in tau_list:
            gtn2_torch.a_i[m,tau],gtn2_torch.b_i[m,tau] = amplitude_fft_nshell_gpu(gtn2_torch.nshell,gtn2_torch.device,tau=tau,geometry='square',lower=True,mu=m,nkx=Lx,nky=Ly)
            gtn2_torch.A_i[m,tau],gtn2_torch.B_i[m,tau] = amplitude_fft_nshell_gpu(gtn2_torch.nshell,gtn2_torch.device,tau=tau,geometry='square',lower=False,mu=m,nkx=Lx,nky=Ly)
    return gtn2_torch

def run(inputs):
    Lx,Ly,nshell,tf,mode,mu,order,seed=inputs
    gtn2_torch=GTN2_torch(Lx=Lx,Ly=Ly,history=False,random_init=False,random_U1=False,bcx=1,bcy=1,seed=seed,orbit=2,nshell=nshell,layer=2,replica=1,gpu=True)

    gtn2_torch.a_i = amp_dicts['a_i']
    gtn2_torch.b_i = amp_dicts['b_i']
    gtn2_torch.A_i = amp_dicts['A_i']
    gtn2_torch.B_i = amp_dicts['B_i']

    overlap = MODE_MAP[mode]['overlap']
    truncate = MODE_MAP[mode]['truncate']

    for i in tqdm(range(tf*gtn2_torch.Lx)):
        measure_feedback_layer_dw_line(gtn2_torch,overlap=overlap,geometry='strip',truncate=truncate,mu=mu,order=order)
        randomize(gtn2_torch,measure=True)

    # Compute S_vN([0,Lx] x [0,a], layer=0) with self-averaging along j
    a_list = np.arange(1, Ly)
    S_list = []
    for a in a_list:
        S_avg = 0.0
        for j0 in range(Ly):
            sub = torch.from_numpy(gtn2_torch.linearize_idx_span(jlist=np.arange(a),ilist=np.arange(Lx),layer=0,shift=(0,j0)))
            S_avg += gtn2_torch.von_Neumann_entropy_m(sub,fermion_idx=False).item()
        S_list.append(S_avg / Ly)
    
    # Bipartite and tripartite mutual information with self-averaging
    a_list_mi = np.arange(1, Ly // 2)
    BMI_list, TMI_list, eta_list = [], [], []
    for a in tqdm(a_list_mi):
        BMI, TMI, eta = gtn2_torch.mutual_information_quasi_1d_crossratio(a, selfaverage=True, layer=0)
        BMI_list.append(BMI.item())
        TMI_list.append(TMI.item())
        eta_list.append(eta)
    return {'S_list': S_list, 'a_list_mi': a_list_mi, 'BMI_list': BMI_list, 'TMI_list': TMI_list, 'eta_list': eta_list}

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--Lx','-Lx',type=int)
    parser.add_argument('--Ly','-Ly',type=int)
    parser.add_argument('--nshell','-nshell',type=int,default=2)
    parser.add_argument('--mu','-mu',type=int,nargs=2,default=[1,3],help='[mu_inner, mu_outer]')
    parser.add_argument('--mode','-mode',type=str,default='overlap',choices=['avoided','truncated','overlap'])
    parser.add_argument('--es','-es',type=int,default=10)
    parser.add_argument('--seed0','-seed0',type=int,default=0)
    parser.add_argument('--tf','-tf',type=int,default=20)
    parser.add_argument('--order','-order',type=str,default='raster',choices=['raster','spiral'])
    args=parser.parse_args()

    st=time.time()
    inputs=[(args.Lx,args.Ly,args.nshell,args.tf,args.mode,args.mu,args.order,seed+args.seed0) for seed in range(args.es)]
    gtn2_dummy=dummy(inputs[0])
    amp_dicts={k:getattr(gtn2_dummy,k) for k in ('a_i','b_i','A_i','B_i')}
    del gtn2_dummy; torch.cuda.empty_cache()
    S_list_all, BMI_list_all, TMI_list_all = [], [], []
    eta_list = None
    for inp in inputs:
        rs = run(inp)
        S_list_all.append(rs['S_list'])
        BMI_list_all.append(rs['BMI_list'])
        TMI_list_all.append(rs['TMI_list'])
        eta_list = rs['eta_list']  # same across ensembles

    a_list = np.arange(1, args.Ly)
    a_list_mi = np.arange(1, args.Ly // 2)
    order_suffix = f'_{args.order}' if args.order != 'raster' else ''
    fn=f'class_A_2D_DW_Lx{args.Lx}_Ly{args.Ly}_nshell{args.nshell}_mu{"_".join(map(str,args.mu))}_es{args.es}_seed{args.seed0}_tf{args.tf}_{args.mode}{order_suffix}_EE.pt'
    torch.save({'S_list':S_list_all,'a_list':a_list,'a_list_mi':a_list_mi,'BMI_list':BMI_list_all,'TMI_list':TMI_list_all,'eta_list':eta_list,'args':args},fn)
    print('Time elapsed: {:.4f}'.format(time.time()-st))
