import multiprocessing
import os
import shutil
from multiprocessing import Process
from typing import List

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from lightning_fabric import Fabric
from lightning_fabric.strategies import DDPStrategy

from src.definitions import AppConfig
from src.training import train
from src.util import patch_environment


def train_batch_job(confs: List[AppConfig], rank=None):
    patch_environment()

    torch.set_float32_matmul_precision("high")

    local_rank = int(os.environ.get("SLURM_PROCID", 0)) if rank is None else rank
    os.environ["LOCAL_RANK"] = str(local_rank)

    conf = confs[0] if len(confs) == 1 else confs[local_rank]

    devices = torch.cuda.device_count()

    fabric = Fabric(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=devices if conf.batch_size == 1 else [local_rank % devices],
        precision="bf16-mixed",
        num_nodes=os.environ.get("SLURM_NNODES", 1),
        strategy=DDPStrategy() if conf.batch_size == 1 else "auto",
    )

    fabric.launch(train, conf)


def train_with_fabric(conf: AppConfig):
    torch.set_float32_matmul_precision("high")

    devices = torch.cuda.device_count()

    fabric = Fabric(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=devices if conf.batch_size == 1 else [conf.batch_idx % devices],
        precision="bf16-mixed",
        num_nodes=1,
        strategy=DDPStrategy() if conf.batch_size == 1 else "auto",
    )

    fabric.launch(train, conf)


def train_interactive_job(configs: List[AppConfig]):
    processes = [
        Process(target=train_with_fabric, args=(c,), daemon=False) for c in configs
    ]

    for p in processes:
        p.start()
    for p in processes:
        p.join()


@hydra.main(
    version_base=None,
    config_path="conf",
    config_name=None,
)
def main(conf: AppConfig):
    multiprocessing.set_start_method("spawn", force=True)

    hydra_conf = HydraConfig.get()
    conf.output_dir = hydra_conf.runtime.output_dir
    conf.overrides = hydra_conf.overrides.task
    conf.batch_size = hydra_conf.launcher.batch_size

    # keep hydra directory config but create
    # directory manually later with additional info
    shutil.rmtree(hydra_conf.runtime.output_dir, ignore_errors=True)

    is_submitting_jobs = "SLURM_JOBID" not in os.environ

    if hydra_conf.mode == RunMode.MULTIRUN:
        return conf, train_batch_job if is_submitting_jobs else train_interactive_job

    torch.set_float32_matmul_precision("high")
    train_batch_job([conf], rank=0)


if __name__ == "__main__":
    main()
