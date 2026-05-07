#!/usr/bin/env python

import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
from matplotlib import cm

import os

os.environ["MAX_SORB"] = '128'

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

# torch.set_default_dtype(torch.double)
torch.set_printoptions(precision=6)
print = partial(print, flush=True)

from pynqs.sample import MCMCParams, ElocParams, SampleParams
from pynqs.sample.base import aux_WF_Params


hidden_num = 128
ansatz_str = "SAAM"


check_point = "./tmp/checkpoint_H50_SAAM_128.pth"

device = "cuda"

seed = 222

mole_name = f"H50"
pth_file = "./ham_h50_2Bohr_oao.pth"  

n_layer = 1
n_det = 1

tree = "./spin_25+25-_10terms.npz"

activation = "silu"
use_hole = True



exact_sample = False

pre_max_iter = 0

method_sample = "MCMC"

burn_in = 2500
mcmc_param = MCMCParams(
    n_walker=4096,  # total n_walker for all ranks 
    therm_step=burn_in//2,
    n_sweep=burn_in//2,
    sample_interval=burn_in//2,
    propose_rule='S',
    starting="random",
    alpha=1.5
)

eloc_params = ElocParams(
    method = ElocMethod.REDUCE,   #ElocMethod.SIMPLE,  /ElocMethod.SAMPLE_SPACE
    use_unique = True,
    use_LUT = True,
    eps = 0.001,
    eps_sample = 1000,
    batch = 1024,
    fp_batch =50000,
)

max_iter = 100
lr_sh = lambda t: min(1E-3, 1E-3/10**((t-3000)/1000))

opt_type = optim.AdamW
opt_params = {"lr": 1, "betas": (0.9, 0.999)}

use_sr = True
store_O_on_cpu = False

e_ref = None

def clip_grad_scheduler(step):
    return 1.0
    # return torch.inf
    # return 1.0 / 10**(max(0,step-1500)/1000)

window = 500

tmp_str = random_str()



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
    mcmc_total = torch.inf if exact_sample else get_world_size()*mcmc_param.n_walker*(mcmc_param.n_sweep//mcmc_param.sample_interval)
    full_label = f"{mole_name}({ansatz_str},{hidden_num})_{tmp_str}"
    
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
        "nva": sorb//2 - noa,
    }
    check_sorb(sorb, nele)
    e_lst = e["e_lst"]
    if rank == 0:
        logger.info(f"e_lst: {e_lst}")
    electron_info = ElectronInfo(info_dict, device=device)

    from pynqs.ansatz.backflow.NN_blocks import Embedding, MLP, Block_Sequential
    from pynqs.ansatz.backflow.WF_blocks import SAAM_NNBF, NNBF

    if ansatz_str == "SAAM":
        Ne = (sorb - nele) if use_hole else nele
        No = sorb//2
        embed = Embedding(
            nqubits=No,
            shape_output=No*3,
            size_dict=3,
            size_embed=3,
            convert=(lambda x: ((x+2)/2).to(torch.int)),
            dtype=dtype,
            device=device,
            params_file=check_point,
        )
        mlp = MLP(
            nqubits=No*3,
            n_layers=n_layer,
            hidden_shape=hidden_num,
            hidden_activation="silu",
            shape_output=(n_det, No, Ne),
            dtype=dtype,
            device=device,
            iscale=1e-1,
            params_file=check_point,
        )
        NN_block = Block_Sequential(embed, mlp)
        ansatz = SAAM_NNBF(
            noa=noa,
            tree=tree,
            NN_block=NN_block,
            dtype=dtype,
            device=device,
            normalization=100.0,
            hole_representation=use_hole,
            tree_dtype=dtype_config.real_dtype,
            params_file=check_point,
        )
    elif ansatz_str == "NNBF":
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
            params_file=check_point,
        )
        ansatz = NNBF(
            NN_block=NN_block,
            dtype=dtype,
            device=device,
            normalization=100.0,
            hole_representation=use_hole,
            params_file=check_point,
        )

    if rank == 0:
        n_param = sum(map(torch.numel, ansatz.parameters()))
        logger.info(f"n_param: {n_param}")

    # ansatz = torch.compile(ansatz, fullgraph=True)
    if device == "cuda":
        model = DDP(ansatz, device_ids=[local_rank], output_device=local_rank)
    else:
        model = DDP(ansatz)


    from pynqs.sample import SampleParams

    sampler_param = SampleParams(
        debug_exact = exact_sample,  # exact optimization
        seed = seed,
        method_sample = method_sample,
        only_AD = False,
        eloc_params = eloc_params,
        params = mcmc_param,
        use_spin_flip = False,
    )

    from pynqs.utils.public_function import SpinProjection

    # opt
    opt = opt_type(model.parameters(), **opt_params)

    from torch.optim.lr_scheduler import LambdaLR
    lr_sch_params = {"lr_lambda": lr_sh}
    lr_scheduler = LambdaLR(opt, **lr_sch_params)

    prefix = f"./tmp/{full_label}"

    from pynqs.property.RenyiS2 import PropertyRenyiS2

    property = PropertyRenyiS2(
        model,
        sampler_param,
        device,
        seed,
        electron_info,
    )

    property.eval(10)

