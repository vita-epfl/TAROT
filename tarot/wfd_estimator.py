from .projectors import (
    ProjectionType,
    AbstractProjector,
    CudaProjector,
    BasicProjector,
    ChunkedCudaProjector,
)
from .gradient_computers import FunctionalGradientComputer, AbstractGradientComputer
from .savers import AbstractSaver, MmapSaver, ModelIDException
from .utils import get_num_params, get_parameter_chunk_sizes,vectorize
from .utils import get_matrix_mult
from typing import Iterable, Optional, Union
from pathlib import Path
from tqdm import tqdm
from torch import Tensor
import os
import logging
import numpy as np
import torch
import psutil
ch = torch


class WFDEstimator:

    def __init__(
        self,
        model: torch.nn.Module,
        task: str,
        candidate_set_size: int,
        target_set_size: int,
        save_dir: str = "./wfd_results",
        load_from_save_dir: bool = True,
        device: Union[str, torch.device] = "cuda",
        gradient_computer: AbstractGradientComputer = FunctionalGradientComputer,
        projector: Optional[AbstractProjector] = None,
        saver: Optional[AbstractSaver] = None,
        proj_dim: int = 2048,
        logging_level=logging.INFO,
        use_half_precision: bool = True,
        proj_max_batch_size: int = 32,
        projector_seed: int = 0,
        grad_wrt: Optional[Iterable[str]] = None,
        lambda_reg: float = 0.0,
    ) -> None:


        self.model = model
        self.task = task
        self.candidate_set_size = candidate_set_size
        self.target_set_size = target_set_size
        self.device = device
        self.dtype = ch.float16 if use_half_precision else ch.float32
        self.grad_wrt = grad_wrt
        self.lambda_reg = lambda_reg

        logging.basicConfig()
        self.logger = logging.getLogger("WFD Estimator")
        self.logger.setLevel(logging_level)

        self.num_params = get_num_params(self.model)
        if self.grad_wrt is not None:
            d = dict(self.model.named_parameters())
            self.num_params_for_grad = sum(
                [d[param_name].numel() for param_name in self.grad_wrt]
            )
        else:
            self.num_params_for_grad = self.num_params
        # inits self.projector
        self.proj_seed = projector_seed
        self.init_projector(
            projector=projector,
            proj_dim=proj_dim,
            proj_max_batch_size=proj_max_batch_size,
        )

        self.normalize_factor = ch.sqrt(
            ch.tensor(self.num_params_for_grad, dtype=ch.float32)
        )

        self.save_dir = Path(save_dir).resolve()
        self.load_from_save_dir = load_from_save_dir

        if type(self.task) is str:
            self.task = TASK_TO_MODELOUT[self.task]

        self.gradient_computer = gradient_computer(
            model=self.model,
            task=self.task,
            grad_dim=self.num_params_for_grad,
            dtype=self.dtype,
            device=self.device,
            grad_wrt=self.grad_wrt,
        )
        metadata = {
            "JL dimension": self.proj_dim,
            "JL matrix type": self.projector.proj_type,
            "candidate set size": self.candidate_set_size,
            "target set size": self.target_set_size,
        }

        if saver is None:
            saver = MmapSaver
        self.saver = saver(
            save_dir=self.save_dir,
            metadata=metadata,
            candidate_set_size=self.candidate_set_size,
            target_set_size=self.target_set_size,
            proj_dim=self.proj_dim,
            load_from_save_dir=self.load_from_save_dir,
            logging_level=logging_level,
            use_half_precision=use_half_precision,
        )

        self.ckpt_loaded = "no ckpt loaded"

    def init_projector(
        self,
        projector: Optional[AbstractProjector],
        proj_dim: int,
        proj_max_batch_size: int,
    ) -> None:


        self.projector = projector
        if projector is not None:
            self.proj_dim = self.projector.proj_dim
            if self.proj_dim == 0:  # using NoOpProjector
                self.proj_dim = self.num_params_for_grad

        else:
            using_cuda_projector = False
            self.proj_dim = proj_dim
            if self.device == "cpu":
                self.logger.info("Using BasicProjector since device is CPU")
                projector = BasicProjector
                # Sampling from bernoulli distribution is not supported for
                # dtype float16 on CPU; playing it safe here by defaulting to
                # normal projection, rather than rademacher
                proj_type = ProjectionType.normal
                self.logger.info("Using Normal projection")
            else:
                try:
                    import fast_jl

                    test_gradient = ch.ones(1, self.num_params_for_grad).cuda()
                    num_sms = ch.cuda.get_device_properties(
                        "cuda"
                    ).multi_processor_count
                    fast_jl.project_rademacher_8(
                        test_gradient, self.proj_dim, 0, num_sms
                    )
                    projector = CudaProjector
                    using_cuda_projector = True

                except (ImportError, RuntimeError, AttributeError) as e:
                    self.logger.error(f"Could not use CudaProjector.\nReason: {str(e)}")
                    self.logger.error("Defaulting to BasicProjector.")
                    projector = BasicProjector
                proj_type = ProjectionType.rademacher

            if using_cuda_projector:
                max_chunk_size, param_chunk_sizes = get_parameter_chunk_sizes(
                    self.model, proj_max_batch_size
                )
                self.logger.debug(
                    (
                        f"the max chunk size is {max_chunk_size}, ",
                        "while the model has the following chunk sizes",
                        f"{param_chunk_sizes}.",
                    )
                )

                if (
                    len(param_chunk_sizes) > 1
                ):  # we have to use the ChunkedCudaProjector
                    self.logger.info(
                        (
                            f"Using ChunkedCudaProjector with"
                            f"{len(param_chunk_sizes)} chunks of sizes"
                            f"{param_chunk_sizes}."
                        )
                    )
                    rng = np.random.default_rng(self.proj_seed)
                    seeds = rng.integers(
                        low=0,
                        high=500,
                        size=len(param_chunk_sizes),
                    )
                    projector_per_chunk = [
                        projector(
                            grad_dim=chunk_size,
                            proj_dim=self.proj_dim,
                            seed=seeds[i],
                            proj_type=ProjectionType.rademacher,
                            max_batch_size=proj_max_batch_size,
                            dtype=self.dtype,
                            device=self.device,
                        )
                        for i, chunk_size in enumerate(param_chunk_sizes)
                    ]
                    self.projector = ChunkedCudaProjector(
                        projector_per_chunk,
                        max_chunk_size,
                        param_chunk_sizes,
                        proj_max_batch_size,
                        self.device,
                        self.dtype,
                    )
                    return  # do not initialize projector below

            self.logger.debug(
                f"Initializing projector with grad_dim {self.num_params_for_grad}"
            )
            self.projector = projector(
                grad_dim=self.num_params_for_grad,
                proj_dim=self.proj_dim,
                seed=self.proj_seed,
                proj_type=proj_type,
                max_batch_size=proj_max_batch_size,
                dtype=self.dtype,
                device=self.device,
            )
            self.logger.debug(f"Initialized projector with proj_dim {self.proj_dim}")

    def load_checkpoint(
        self,
        checkpoint: Iterable[Tensor],
        model_id: int,
        _allow_featurizing_already_registered=False,
    ) -> None:

        if self.saver.model_ids.get(model_id) is None:
            self.saver.register_model_id(
                model_id, _allow_featurizing_already_registered
            )
        else:
            self.saver.load_current_store(model_id)

        self.model.load_state_dict(checkpoint)
        self.model.eval()

        self.gradient_computer.load_model_params(self.model)

        self._last_ind = 0
        self.ckpt_loaded = model_id

    def collect_grads(
        self,
        batch: Iterable[Tensor],
        inds: Optional[Iterable[int]] = None,
        num_samples: Optional[int] = None,
        is_target: bool = False,
    ) -> None:
        if num_samples is not None:
            inds = np.arange(self._last_ind, self._last_ind + num_samples)
            self._last_ind += num_samples
        else:
            num_samples = inds.reshape(-1).shape[0]

        mode = "target" if is_target else "candidate"
        grads = self.gradient_computer.compute_per_sample_grad(batch=batch)

        grads = self.projector.project(grads, model_id=0)
        grads /= self.normalize_factor

        self.saver.current_store[f"{mode}_grads"][inds] = (
            grads.to(self.dtype).cpu().clone().detach()
        )
        self.saver.current_store[f"{mode}_is_collected"][inds] = 1
        
        if self._last_ind == self.candidate_set_size:
            self._last_ind = 0
            print('finished grad collection')

    def get_xtx(self, grads: Tensor) -> Tensor:
        self.proj_dim = grads.shape[1]
        result = ch.zeros(
            self.proj_dim, self.proj_dim, dtype=self.dtype, device=self.device
        )
        blocks = ch.split(grads, split_size_or_sections=20_000, dim=0)

        for block in blocks:
            result += block.T.to(self.device) @ block.to(self.device)

        return result


    def get_wfd(self) -> Tensor:
        # find all directories within self.savedir
        model_ids = [d for d in os.listdir(self.save_dir) if os.path.isdir(os.path.join(self.save_dir, d))]
        print(model_ids)
        self.saver.load_current_store(model_ids[0])

        _g_mmp = self.saver.current_store["candidate_grads"]
        _g_target_mmp = self.saver.current_store["target_grads"]
        _g_cpu = ch.zeros([_g_mmp.shape[0],_g_mmp.shape[1]], device="cpu", dtype=torch.float16)
        _g_target_cpu = ch.zeros([_g_target_mmp.shape[0],_g_target_mmp.shape[1]], device="cpu", dtype=torch.float16)

        for j, model_id in enumerate(
            tqdm(model_ids, desc="Aggregating gradients for all model IDs..")
        ):
            self.saver.load_current_store(model_id)
            g = ch.as_tensor(self.saver.current_store["candidate_grads"], device='cpu')
            g_target = ch.as_tensor(
                self.saver.current_store["target_grads"], device='cpu'
            )
            _g_cpu += g
            _g_target_cpu += g_target

        _g_cpu.div_(len(model_ids))
        _g_target_cpu.div_(len(model_ids))
        num_c = g.shape[0]

        from .utils import get_matrix_mult

        all_features = torch.cat((g, g_target), 0).to(self.device)
        all_features.sub_(all_features.mean(0, keepdim=True))
        xtx_all = self.get_xtx(all_features)/all_features.shape[0]

        del g,g_target
        xtx_all += torch.eye(xtx_all.shape[0], device=xtx_all.device) * 1e-5
        cholesky_factor = torch.linalg.cholesky(xtx_all)
        whitening_matrix = torch.inverse(cholesky_factor).T

        whitening_matrix = whitening_matrix.to(torch.float16)
        all_features = all_features.to(torch.float16)

        all_features = get_matrix_mult(all_features, whitening_matrix)

        g = all_features[:num_c]
        g.div_(g.norm(2, -1, keepdim=True).clamp_min(1e-12))

        g_target = all_features[num_c:]
        g_target.div_(g_target.norm(2, -1, keepdim=True).clamp_min(1e-12))

        del all_features
        _scores_on_cpu = get_matrix_mult(features=g, target_grads=g_target).detach().cpu().numpy()

        # self.saver.load_current_store(list(model_ids.keys())[0], exp_name, num_targets)
        # _scores_mmap = self.saver.current_store[f"{exp_name}_scores"]
        # _scores_mmap[:] = _scores_on_cpu
        # self.saver.save_scores(exp_name)
        print('finished')
        return _scores_on_cpu,g,g_target


def get_loss_motion(
    model,
    weights,
    buffers,
    *batch,
) -> Tensor:
    keys = batch[-1]
    batch = batch[:-1]
    batch = {k: v.unsqueeze(0) for k, v in zip(keys, batch)}
    batch = {'batch_size': batch['obj_trajs'].shape[0], 'input_dict': batch}
    prediction, loss = ch.func.functional_call(model, (weights, buffers), batch)

    return loss.sum()/1000


TASK_TO_MODELOUT = {
    "motion_prediction": get_loss_motion
}
