from torch import Tensor
import torch
import numpy as np
from multiprocessing import Pool
import matplotlib.pyplot as plt
import os
ch = torch
import seaborn as sns
from sklearn.manifold import TSNE
from matplotlib.colors import TwoSlopeNorm
import wandb

def parameters_to_vector(parameters) -> Tensor:
    """
    Same as https://pytorch.org/docs/stable/generated/torch.nn.utils.parameters_to_vector.html
    but with :code:`reshape` instead of :code:`view` to avoid a pesky error.
    """
    vec = []
    for param in parameters:
        vec.append(param.reshape(-1))
    return ch.cat(vec)


def get_num_params(model: torch.nn.Module) -> int:
    return parameters_to_vector(model.parameters()).numel()


def is_not_buffer(ind, params_dict) -> bool:
    name = params_dict[ind]
    if (
        ("running_mean" in name)
        or ("running_var" in name)
        or ("num_batches_tracked" in name)
    ):
        return False
    return True


def vectorize(g, arr=None, device="cuda") -> Tensor:
    """
    records result into arr

    gradients are given as a dict :code:`(name_w0: grad_w0, ... name_wp:
    grad_wp)` where :code:`p` is the number of weight matrices. each
    :code:`grad_wi` has shape :code:`[batch_size, ...]` this function flattens
    :code:`g` to have shape :code:`[batch_size, num_params]`.
    """
    if arr is None:
        g_elt = g[list(g.keys())[0]]
        batch_size = g_elt.shape[0]
        num_params = 0
        for param in g.values():
            assert param.shape[0] == batch_size
            num_params += int(param.numel() / batch_size)
        arr = ch.empty(size=(batch_size, num_params), dtype=g_elt.dtype, device=device)

    pointer = 0
    for param in g.values():
        if len(param.shape) < 2:
            num_param = 1
            p = param.data.reshape(-1, 1)
        else:
            num_param = param[0].numel()
            p = param.flatten(start_dim=1).data

        arr[:, pointer : pointer + num_param] = p.to(device)
        pointer += num_param

    return arr


def get_output_memory(features: Tensor, target_grads: Tensor, target_dtype: type):
    output_shape = features.size(0) * target_grads.size(0)
    output_dtype_size = ch.empty((1,), dtype=target_dtype).element_size()

    return output_shape * output_dtype_size


def get_free_memory(device):
    reserved = ch.cuda.memory_reserved(device=device)
    allocated = ch.cuda.memory_allocated(device=device)

    free = reserved - allocated
    return free


def get_matrix_mult_standard(
    features: Tensor, target_grads: Tensor, target_dtype: type
):
    output = features @ target_grads.t()
    return output.clone().to(target_dtype)


def get_matrix_mult_blockwise(
    features: Tensor, target_grads: Tensor, target_dtype: type, bs: int
):
    s_features = features.shape[0]
    s_target_grads = target_grads.shape[0]

    bs = min(s_features, s_target_grads, bs)

    # Copy the data in a pinned memory location to allow non-blocking
    # copies to the GPU
    features = features.pin_memory()
    target_grads = target_grads.pin_memory()

    # precompute all the blocks we will have to compute
    slices = []
    for i in range(int(np.ceil(s_features / bs))):
        for j in range(int(np.ceil(s_target_grads / bs))):
            slices.append((slice(i * bs, (i + 1) * bs), slice(j * bs, (j + 1) * bs)))

    # Allocate memory for the final output.
    final_output = ch.empty(
        (s_features, s_target_grads), dtype=target_dtype, device="cpu"
    )

    # Output buffers pinned on the CPU to be able to collect data from the
    # GPU asynchronously
    # For each of our (2) cuda streams we need two output buffer, one
    # is currently written on with the next batch of result and the
    # second one is already finished and getting copied on the final output

    # If the size is not a multiple of batch size we need extra buffers
    # with the proper shapes
    outputs = [
        ch.zeros((bs, bs), dtype=target_dtype, device=features.device).pin_memory()
        for x in range(4)
    ]
    left_bottom = s_features % bs
    options = [outputs]  # List of buffers we can potentially use
    if left_bottom:
        outputs_target_gradsottom = [
            ch.zeros(
                (left_bottom, bs), dtype=target_dtype, device=features.device
            ).pin_memory()
            for x in range(4)
        ]
        options.append(outputs_target_gradsottom)
    left_right = s_target_grads % bs
    if left_right:
        outputs_right = [
            ch.zeros(
                (bs, left_right), dtype=target_dtype, device=features.device
            ).pin_memory()
            for x in range(4)
        ]
        options.append(outputs_right)
    if left_right and left_bottom:
        outputs_corner = [
            ch.zeros(
                (left_bottom, left_right), dtype=target_dtype, device=features.device
            ).pin_memory()
            for x in range(4)
        ]
        options.append(outputs_corner)

    streams = [ch.cuda.Stream() for x in range(2)]

    # The slice that was computed last and need to now copied onto the
    # final output
    previous_slice = None

    def find_buffer_for_shape(shape):
        for buff in options:
            if buff[0].shape == shape:
                return buff
        return None

    for i, (slice_i, slice_j) in enumerate(slices):
        with ch.cuda.stream(streams[i % len(streams)]):
            # Copy the relevant blocks from CPU to the GPU asynchronously
            features_i = features[slice_i, :].cuda(non_blocking=True)
            target_grads_j = target_grads[slice_j, :].cuda(non_blocking=True)

            output_slice = features_i @ target_grads_j.t()

            find_buffer_for_shape(output_slice.shape)[i % 4].copy_(
                output_slice, non_blocking=False
            )

        # Write the previous batch of data from the temporary buffer
        # onto the final one (note that this was done by the other stream
        # so we swap back to the other one
        with ch.cuda.stream(streams[(i + 1) % len(streams)]):
            if previous_slice is not None:
                output_slice = final_output[previous_slice[0], previous_slice[1]]
                output_slice.copy_(
                    find_buffer_for_shape(output_slice.shape)[(i - 1) % 4],
                    non_blocking=True,
                )

        previous_slice = (slice_i, slice_j)

    # Wait for all the calculations/copies to be done
    ch.cuda.synchronize()

    # Copy the last chunk to the final result (from the appropriate buffer)
    output_slice = final_output[previous_slice[0], previous_slice[1]]
    output_slice.copy_(
        find_buffer_for_shape(output_slice.shape)[i % 4], non_blocking=True
    )

    return final_output


def get_matrix_mult(
    features: Tensor,
    target_grads: Tensor,
    target_dtype: torch.dtype = None,
    batch_size: int = 8096,
    use_blockwise: bool = False,
) -> Tensor:
    """

    Computes features @ target_grads.T. If the output matrix is too large to fit
    in memory, it will be computed in blocks.

    Args:
        features (Tensor):
            The first matrix to multiply.
        target_grads (Tensor):
            The second matrix to multiply.
        target_dtype (torch.dtype, optional):
            The dtype of the output matrix. If None, defaults to the dtype of
            features. Defaults to None.
        batch_size (int, optional):
            The batch size to use for blockwise matrix multiplication. Defaults
            to 8096.
        use_blockwise (bool, optional):
            Whether or not to use blockwise matrix multiplication. Defaults to
            False.

    """
    if target_dtype is None:
        target_dtype = features.dtype

    if use_blockwise:
        return get_matrix_mult_blockwise(
            features.cpu(), target_grads.cpu(), target_dtype, batch_size
        )
    elif features.device.type == "cpu":
        return get_matrix_mult_standard(features, target_grads, target_dtype)

    output_memory = get_output_memory(features, target_grads, target_dtype)
    free_memory = get_free_memory(features.device)

    if output_memory < free_memory:
        return get_matrix_mult_standard(features, target_grads, target_dtype)
    else:
        return get_matrix_mult_blockwise(
            features.cpu(), target_grads.cpu(), target_dtype, batch_size
        )


def get_parameter_chunk_sizes(
    model: torch.nn.Module,
    batch_size: int,
):
    """The :class:`CudaProjector` supports projecting when the product of the
    number of parameters and the batch size is less than the the max value of
    int32. This function computes the number of parameters that can be projected
    at once for a given model and batch size.

    The method returns a tuple containing the maximum number of parameters that
    can be projected at once and a list of the actual number of parameters in
    each chunk (a sequence of paramter groups).  Used in
    :class:`ChunkedCudaProjector`.
    """
    param_shapes = []
    for p in model.parameters():
        param_shapes.append(p.numel())

    param_shapes = np.array(param_shapes)

    chunk_sum = 0
    max_chunk_size = np.iinfo(np.uint32).max // batch_size
    params_per_chunk = []

    for ps in param_shapes:
        if chunk_sum + ps >= max_chunk_size:
            params_per_chunk.append(chunk_sum)
            chunk_sum = 0

        chunk_sum += ps

    if param_shapes.sum() - np.sum(params_per_chunk) > 0:
        params_per_chunk.append(param_shapes.sum() - np.sum(params_per_chunk))

    return max_chunk_size, params_per_chunk



def set_dataloader_params(original_loader,key,value):
    from torch.utils.data import DataLoader
    loader_params = vars(original_loader).copy()
    loader_params[key] = value
    unwanted_keys = [key for key in loader_params.keys() if key.startswith('_')]
    for key in unwanted_keys:
        loader_params.pop(key, None)
    loader_params.pop("batch_sampler", None)
    loader_params.pop("sampler", None)
    updated_loader = DataLoader(**loader_params)
    return updated_loader

def qualitative_analysis(exp_name, selected_index, weight, scores, g, g_target, candidate_loader, target_loader, vis_func):
    wandb_image = {}
    plt.style.use('ggplot')
    wandb_image['tsne'] = tsne_visulization(weight, selected_index,g,g_target)
    wandb_image['influence_estimation'] = visualize_supportive_and_negative(candidate_loader, target_loader, scores, vis_func)
    wandb.init(project='tarot', name=exp_name)
    wandb.log(wandb_image)
    return



def tsne_visulization(weight, selected_index, g,g_target):


    sample_num_train = 2400
    sample_num_val = 500

    weight[:] = 1
    potential = np.zeros(g.shape[0])
    potential[:] = -1
    potential[selected_index] = weight

    print(np.mean(potential), np.max(potential), np.min(potential))
    # Normalize the potential values for gradient color mapping with a center at 0
    from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap

    colors = ['#EAD2AC', '#4281A4']
    cmap = LinearSegmentedColormap.from_list('custom_coolwarm', colors)
    norm = TwoSlopeNorm(vmin=min(potential), vcenter=np.mean(potential), vmax=max(potential))

    # Create the custom colormap
    g = np.array(g.cpu())
    g_target = np.array(g_target.cpu())
    g_random_sample = np.random.choice(len(g), min(len(g), sample_num_train), replace=False)
    g = g[g_random_sample]
    potential = potential[g_random_sample]
    g_target_random_sample = np.random.choice(len(g_target), min(len(g_target), sample_num_val), replace=False)
    g_target = g_target[g_target_random_sample]

    all_gradient = np.concatenate([g, g_target], axis=0)

    print('start to fit')
    tsne = TSNE(n_jobs=16, perplexity=50, early_exaggeration=20)
    tsne_points = tsne.fit_transform(all_gradient)
    print('fit done')
    train_tsne = tsne_points[:len(g)]
    val_tsne = tsne_points[len(g):]

    size = 30
    alpha = 1
    # Create the plot
    plt.figure(figsize=(10, 8))
    # Plot train points with gradient color based on potential
    train_tsne = train_tsne[potential > -1]
    potential = potential[potential > -1]
    plt.scatter(train_tsne[:, 0], train_tsne[:, 1],
                                label='Candidate Points', s=size, alpha=alpha, c=potential,cmap=cmap, norm=norm,zorder=1)
    # Plot validation points in a different solid color (colorblind-friendly)
    plt.scatter(val_tsne[:, 0], val_tsne[:, 1],
                color="#FE938C", label='Target Points', s=size, alpha=1,zorder=0)

    plt.axis('off')

    return wandb.Image(plt)

def visualize_supportive_and_negative(train_loader, val_loader, scores, vis_func):
    total_num = len(val_loader.dataset)
    sample_interval = max(total_num//30,1)
    image_list = []
    for i in range(0, total_num, sample_interval):
        fig, ax = plt.subplots(1,5, figsize=(20, 4))

        val_data = val_loader.dataset[i]
        _ = vis_func(ax[0], val_data)

        top_scores = scores[:, i].argsort()[-2:][::-1]
        bottom_scores = scores[:, i].argsort()[:2][::-1]

        for ii, train_im_ind in enumerate(top_scores):
            _ = vis_func(ax[ii+1], train_loader.dataset[train_im_ind])

        for ii, train_im_ind in enumerate(bottom_scores):
            _ = vis_func(ax[ii+3], train_loader.dataset[train_im_ind])

        plt.tight_layout()
        #plt.show()
        image_list.append(wandb.Image(plt))
    return image_list