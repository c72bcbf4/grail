import logging
import os
import socket
import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

import submitit
from hydra._internal.core_plugins.basic_launcher import BasicLauncher
from hydra.core.config_store import ConfigStore
from hydra.core.utils import (
    JobReturn,
    JobStatus,
)
from hydra.types import HydraContext, TaskFunction
from omegaconf import DictConfig

log = logging.getLogger(__name__)


class CustomLauncher(BasicLauncher):
    def __init__(self, mode, strict, batch_size, conf):
        super().__init__()
        self.mode = mode
        self.strict = strict
        self.conf = conf
        self.batch_size = batch_size
        self.config: Optional[DictConfig] = None
        self.task_function: Optional[TaskFunction] = None
        self.hydra_context: Optional[HydraContext] = None

        if mode not in ["run", "submit"]:
            raise RuntimeError(f"invalid launcher mode '{self.mode}'")

        if "SLURM_JOBID" in os.environ and self.mode == "submit":
            log.warning(
                "detected slurm job. changing mode to 'run'. setting strict to 'false'."
            )
            self.mode = "run"
            self.strict = False

    def setup(
        self,
        *,
        hydra_context: HydraContext,
        task_function: TaskFunction,
        config: DictConfig,
    ) -> None:
        self.config = config
        self.hydra_context = hydra_context
        self.task_function = task_function

    def split(self, array, n):
        batch = 0
        batches = []
        tmp = []
        for i, (config, fn) in enumerate(array):
            config.batch = batch
            config.batch_idx = i
            tmp.append((config, fn))
            if len(tmp) == n:
                batches.append(tmp)
                tmp = []
                batch += 1

        # avoid partially configured jobs
        if tmp and not self.strict:
            batches.append(tmp)

        return batches

    def launch(
        self, job_overrides: Sequence[Sequence[str]], initial_job_idx: int
    ) -> Sequence[JobReturn]:
        results = super().launch(job_overrides, initial_job_idx)

        for i, job in enumerate(results):
            assert (
                job.status == JobStatus.COMPLETED
            ), f"failed get parameters for job {i}:\n{job.return_value}"

        configs = [r.return_value for r in results]

        batches = self.split(configs, self.batch_size)

        log.info(f"configuration yielded {len(batches)} batches")

        for batch in batches:
            params, fns = zip(*batch)
            fn = fns[0]

            if self.mode == "run":
                log.info("running job locally")
                fn(params)
                continue

            executor = submitit.AutoExecutor(folder="slurm")
            executor.update_parameters(**self.conf)

            job = executor.submit(fn, params)
            log.info(f"submitted job {job.job_id}")

        return results


@dataclass
class CustomLauncherConf:
    _target_: str = "hydra_plugins.plugins.CustomLauncher"
    mode: str = "run"
    strict: bool = True
    batch_size: int = 1
    conf: dict = field(default_factory=lambda: dict())


ConfigStore.instance().store(
    group="hydra/launcher",
    name="custom",
    node=CustomLauncherConf,
)
