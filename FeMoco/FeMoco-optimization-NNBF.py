#!/usr/bin/env python


import os

os.environ["MAX_SORB"] = '192'

import sys
import torch
import time
import numpy as np
import torch.distributed as dist
import tempfile
import argparse

from time import ctime
from pyscf import gto, fci

from functools import partial
from loguru import logger
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP

from pynqs.utils import setup_seed, ElectronInfo
from pynqs.utils.pyscf_helper import read_integral, interface
from pynqs.distributed import get_rank, get_world_size
from pynqs.utils.loggings import dist_print
from pynqs.utils.enums import ElocMethod
from pynqs.ansatz import (
    RBMWavefunction,
    Graph_MPS_RNN,
    # MultiPsi_non_ar,
    # NNBWavefunction,
)
from pynqs.config import dtype_config
from pynqs.utils.public_function import random_str
from pynqs.optim import VMCOptimizer
from pynqs.libs.C_extension import check_sorb
from pynqs.utils.ci import unpack_ucisd, fci_revise
from pynqs import cuda_synchronize_config

cuda_synchronize_config.apply(use_synchronize=True)

# torch.set_default_dtype(torch.double)
torch.set_printoptions(precision=6)
print = partial(print, flush=True)


from pynqs.sample import MCMCParams, ElocParams, SampleParams
from pynqs.sample.base import aux_WF_Params

tmp_str = random_str()

device = "cuda"

seed = 222

mole_name = "FeMo-LMO"
pth_file = "./ham_femoco_LMO.pth"  # "./molecule/H6_1.0A_OAO.pth"  #  

n_layer = 1
hidden_num = 256
n_det = 1

# --------------------------------------------------------------------------

checkpoint = f"./checkpoint_femoco_LMO_g6trp614.pth"

max_iter = 15000
max_lr = 1e-3
def lr_sh(t): 
    if t<=2000: 
        return max_lr*t/2000
    elif t<=4000: 
        return max_lr
    elif t<=13000:
        return max_lr/(1+(t-4000)/1000)
    else:
        return max_lr/10/10**((t-13000)/2000)

full_label = f"{mole_name}-NNBF-h{hidden_num}-{tmp_str}"

# --------------------------------------------------------------------------

aux_params_file = "./mps_femoco_LMO_isweep3_d100.pth"
aux_graph_file = "./graph_femoco_LMO.graphml"
aux_dcut = 100

method_sample = "HybridMC"
burn_in = 3000
mcmc_param = MCMCParams(
    n_walker=8192,  # total n_walker for all ranks 
    therm_step=burn_in//2,
    n_sweep=burn_in//2,
    sample_interval=burn_in//2,
    propose_rule='S',
    starting="aux",
    aux_wf_params=None,
    alpha=2,
    compile_kwargs={"dynamic": True, "fullgraph": False},
)

eloc_params = ElocParams(
    method = ElocMethod.REDUCE,   #ElocMethod.SIMPLE,  /ElocMethod.SAMPLE_SPACE
    use_unique = True,
    use_LUT = True,
    eps = 1e-2,
    eps_sample = 1000,
    batch = 128,
    fp_batch = 20000,
)

opt_type = optim.AdamW
opt_params = {"lr": 1, "betas": (0.9, 0.999)}

use_sr = True
store_O_on_cpu = False


    


def clip_grad_scheduler(step):
    return 1.0
    # return torch.inf
    # return 1.0 / 10**(max(0,step-1500)/1000)

window = 100

e_ref = None





if __name__ == "__main__":

    if(device=="cpu" or store_O_on_cpu):
        backend = "gloo"
    elif(device=="cuda"):
        backend = "nccl"

    dist.init_process_group(backend)
    local_rank = int(os.environ["LOCAL_RANK"])
    # tmp_str = random_str()
    dtype_config.apply(use_complex=False, use_float64=False, device=device)
    dtype = dtype_config._default_dtype

    mcmc_param.n_walker = mcmc_param.n_walker // get_world_size()
    
    
    setup_seed(seed)
    if device == "cuda":
        torch.cuda.set_device(local_rank)
    logger.remove()
    logger.add(dist_print, format="{message}", enqueue=True, level="INFO")
    logger.add(f"./log/{full_label}.log", format="{message}", enqueue=True, level="INFO")
    rank = get_rank()


    e = torch.load(pth_file, map_location="cpu", weights_only=False)
    h1e = e["h1e"]
    h2e = e["h2e"]
    sorb = e["sorb"]
    noa = e["noa"]
    nob = e["nob"]
    ci_space = e["ci_space"]
    ecore = e["ecore"]
    nele = e["nele"]
    info_dict = {
        "h1e": h1e,
        "h2e": h2e,
        "onstate": ci_space,
        "ecore": ecore,
        "sorb": sorb,
        "nele": nele,
        "nob": nob,
        "noa": noa,
        "nva": (sorb - nele) // 2,
    }
    check_sorb(sorb, nele)
    e_lst = e["e_lst"]
    if rank == 0:
        logger.info(f"e_lst: {e_lst}")
    electron_info = ElectronInfo(info_dict, device=device)


    from pynqs.ansatz.backflow.NN_blocks import Embedding, MLP, Block_Sequential, ResNet_with_BN
    from pynqs.ansatz.backflow.WF_blocks import SAAM_NNBF, NNBF

    use_hole = True
    Ne = (sorb - nele) if use_hole else nele
    No = sorb
    NN_block = MLP(
        nqubits=No,
        n_layers=n_layer,
        hidden_shape=hidden_num,
        hidden_activation="silu",
        shape_output=(n_det, No, Ne),
        dtype=dtype,
        device=device,
        iscale=1e-1,
        params_file=checkpoint,
    )
    ansatz = NNBF(
        NN_block=NN_block,
        dtype=dtype,
        device=device,
        normalization=100.0,
        hole_representation=use_hole,
        params_file=checkpoint,
    )

    from pynqs.ansatz.rnn.graph_mps import Graph_MPS
    from pynqs.sample.base import ARParams, aux_WF_Params
    import networkx as nx
    graph_nn0 = nx.read_graphml(aux_graph_file)
    aux_wf = Graph_MPS(
        hilbert_local=4,
        nqubits=sorb,
        nele=nele,
        device=device,
        dcut=aux_dcut,
        graph=graph_nn0,
        rank_independent_sampling=True,
        params_file=aux_params_file,
        alpha_nele=noa,
    )
    if device == "cuda":
        aux_wf = DDP(aux_wf, device_ids=[local_rank], output_device=local_rank)
    else:
        aux_wf = DDP(aux_wf)

    aux_ar_params = ARParams(
        n_sample=mcmc_param.n_walker,
        use_dfs_sample=True,
        use_same_tree=False,
        min_batch=10000,
        min_tree_height=20,
    )

    aux_params = aux_WF_Params(
        aux_wf=aux_wf,
        aux_sampler_params=aux_ar_params
    )

    mcmc_param.aux_wf_params = aux_params

    
    if rank == 0:
        n_param = sum(map(torch.numel, ansatz.parameters()))
        logger.info(f"n_param: {n_param}")
        n_param_requires_grad = sum(p.numel() for p in ansatz.parameters() if p.requires_grad)
        logger.info(f"n_param_requires_grad: {n_param_requires_grad}")

    # ansatz = torch.compile(ansatz, fullgraph=True)
    if device == "cuda":
        model = DDP(ansatz, device_ids=[local_rank], output_device=local_rank)
    else:
        model = DDP(ansatz)



    from pynqs.sample import SampleParams

    sampler_param = SampleParams(
        debug_exact = False,  # exact optimization
        seed = seed,
        method_sample = method_sample,
        only_AD = False,
        eloc_params = eloc_params,
        params = mcmc_param,
        use_spin_flip = False,
    )


    # opt
    opt = opt_type(model.parameters(), **opt_params)

    from torch.optim.lr_scheduler import LambdaLR
    lr_sch_params = {"lr_lambda": lr_sh}
    lr_scheduler = LambdaLR(opt, **lr_sch_params)

    prefix = f"./tmp/{full_label}"

    vmc_opt_params = {
        "nqs": model,
        "opt": opt,
        "lr_scheduler": lr_scheduler,
        "sampler_param": sampler_param,
        # "only_sample": True,
        "electron_info": electron_info,
        "use_spin_raising": True,
        "spin_raising_coeff": 1,
        "only_output_spin_raising": True,
        "max_iter": max_iter,
        "interval": 10,
        "MAX_AD_DIM": 256,
        # "check_point": check_point,
        "read_model_only": True,
        "prefix": prefix,
        "use_clip_grad": True,
        "max_grad_norm": 1,
        "start_clip_grad": -1,
        "clip_grad_scheduler": clip_grad_scheduler,
        "clip_eloc": False,
        "sr": use_sr,
        "store_O_on_cpu": store_O_on_cpu,
    }

    if(e_ref is None):  e_ref = e_lst[0]
    opt_vmc = VMCOptimizer(**vmc_opt_params)
    opt_vmc.run()
    # opt_vmc.summary(e_ref, e_lst, prefix=prefix)

