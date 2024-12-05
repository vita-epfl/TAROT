from abc import ABC, abstractmethod
from typing import Iterable, Optional
from torch import Tensor
from .utils import get_num_params, parameters_to_vector
import logging
import torch

ch = torch


class AbstractGradientComputer(ABC):
    """Implementations of the GradientComputer class should allow for
    per-sample gradients.  This is behavior is enabled with three methods:

    - the :meth:`.load_model_params` method, well, loads model parameters. It can
      be as simple as a :code:`self.model.load_state_dict(..)`

    - the :meth:`.compute_per_sample_grad` method computes per-sample gradients
      of the chosen model output function with respect to the model's parameters.

    - the :meth:`.compute_loss_grad` method computes the gradients of the loss
      function with respect to the model output (which should be a scalar) for
      every sample.

    """

    @abstractmethod
    def __init__(
        self,
        model: torch.nn.Module,
        task,
        grad_dim: Optional[int] = None,
        dtype: Optional[torch.dtype] = torch.float16,
        device: Optional[torch.device] = "cuda",
    ) -> None:

        self.model = model
        self.loss_fn = task
        self.grad_dim = grad_dim
        self.dtype = dtype
        self.device = device

    @abstractmethod
    def load_model_params(self, model) -> None:
        ...

    @abstractmethod
    def compute_per_sample_grad(self, batch: Iterable[Tensor]) -> Tensor:
        ...

class FunctionalGradientComputer(AbstractGradientComputer):
    def __init__(
        self,
        model: torch.nn.Module,
        task,
        grad_dim: int,
        dtype: torch.dtype,
        device: torch.device,
        grad_wrt: Optional[Iterable[str]] = None,
    ) -> None:
        """Initializes attributes, and loads model parameters.

        Args:
            grad_wrt (list[str], optional):
                A list of parameter names for which to keep gradients.  If None,
                gradients are taken with respect to all model parameters.
                Defaults to None.
        """
        super().__init__(model, task, grad_dim, dtype, device)
        self.model = model
        self.num_params = get_num_params(self.model)
        self.load_model_params(model)
        self.grad_wrt = grad_wrt
        self.logger = logging.getLogger("GradientComputer")

    def load_model_params(self, model) -> None:
        """Given a a torch.nn.Module model, inits/updates the (functional)
        weights and buffers. See https://pytorch.org/docs/stable/func.html
        for more details on :code:`torch.func`'s functional models.

        Args:
            model (torch.nn.Module):
                model to load

        """
        self.func_weights = dict(model.named_parameters())
        self.func_buffers = dict(model.named_buffers())

    def compute_per_sample_grad(self, batch: Iterable[Tensor]) -> Tensor:
        """Uses functorch's :code:`vmap` (see
        https://pytorch.org/functorch/stable/generated/functorch.vmap.html#functorch.vmap
        for more details) to vectorize the computations of per-sample gradients.

        Doesn't use :code:`batch_size`; only added to follow the abstract method
        signature.

        Args:
            batch (Iterable[Tensor]):
                batch of data

        Returns:
            dict[Tensor]:
                A dictionary where each key is a parameter name and the value is
                the gradient tensor for that parameter.

        """
        # taking the gradient wrt weights (second argument of get_output, hence argnums=1)
        grads_loss = torch.func.grad(
            self.loss_fn, has_aux=False, argnums=1
        )
        #self.loss_fn(self.model, self.func_weights, self.func_buffers, *batch)
        grads = torch.func.vmap(
            grads_loss,
            in_dims=(None, None, None, *[0 if isinstance(x, torch.Tensor) else None for x in batch]),
            randomness="different",
        )(self.model, self.func_weights, self.func_buffers, *batch)

        if self.grad_wrt is not None:
            for param_name in list(grads.keys()):
                if param_name not in self.grad_wrt:
                    del grads[param_name]
        return grads

