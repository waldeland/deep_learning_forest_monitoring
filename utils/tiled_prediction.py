import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm


def tiled_prediction(
    data,
    net,
    patch_size=None,
    patch_overlap=None,
    apply_classifier=False,
    apply_softmax=False,
    batch_size=32,
    precision="float",
):
    """
    Chops up a large image in smaller patches and run each patch through a segmentation network. The output is stitched
    together from the predicted patches.
    Args:
        data (np.array): The large image (np.array 2D (single channel) or 3D (multiple channels) )
        net (torch.nn.Module): A pytorch segmentation model (input size must be equal to output size)
        patch_size ([int,int]): Size of patches
        patch_overlap ([int,int]): How much overlap there should be between patches
        apply_classifier (bool): Apply argmax across of ouput channels
        apply_softmax (bool): Apply softmax across of ouput channels
        batch_size(int): number of samples pr batch
        make_input_divisable_with (int): If patch_size is None, then it might be required to pad the input to make the dimensions with the model
        precision (str): 'half' or 'single'

    Returns:
         Predictions for large image (np.array 2D (single channel) or 3D (multiple channels))
    """

    # Functions to convert between B x C x H x W format and W x H x C format
    def hwc_to_bchw(x):
        if not isinstance(x, list):
            x = [x]
        x = [np.expand_dims(x_, 0) for x_ in x]
        x = np.concatenate(x, 0)
        x = np.moveaxis(x, -1, 1)
        return x

    def bcwh_to_hwc(x):
        x = np.moveaxis(x, 1, -1)
        return [x[i, :, :, :] for i in range(x.shape[0])]

    # Some preprocessing
    if len(data.shape) == 2:
        data = np.expand_dims(data, -1)

    if type(patch_size) == int:
        patch_size = [patch_size, patch_size]

    # Add padding to avoid trouble when removing the overlap later
    data = np.pad(
        data,
        [
            [patch_overlap[0], patch_overlap[0]],
            [patch_overlap[1], patch_overlap[1]],
            [0, 0],
        ],
        "constant",
    )

    # Loop through patches identified by upper-left pixel
    upper_left_x0 = np.arange(
        0, data.shape[0] - patch_overlap[0], patch_size[0] - patch_overlap[0] * 2
    )
    upper_left_x1 = np.arange(
        0, data.shape[1] - patch_overlap[1], patch_size[1] - patch_overlap[1] * 2
    )

    predictions = []

    batched_data = []
    batched_x0 = []
    batched_x1 = []
    batched_pad_val_0 = []
    batched_pad_val_1 = []

    bar = tqdm(desc = 'Predicting', total = len(upper_left_x0)*len(upper_left_x1))
    bar.update(0)

    for x0 in upper_left_x0:
        for x1 in upper_left_x1:
            # Cut out a small patch of the data
            data_patch = data[x0 : x0 + patch_size[0], x1 : x1 + patch_size[1], :]

            # Pad with zeros if we are at the edges
            pad_val_0 = patch_size[0] - data_patch.shape[0]
            pad_val_1 = patch_size[1] - data_patch.shape[1]

            if pad_val_0 > 0:
                data_patch = np.pad(
                    data_patch, [[0, pad_val_0], [0, 0], [0, 0]], "constant"
                )

            if pad_val_1 > 0:
                data_patch = np.pad(
                    data_patch, [[0, 0], [0, pad_val_1], [0, 0]], "constant"
                )

            # Add to batch:
            batched_data.append(data_patch)
            batched_x0.append(x0)
            batched_x1.append(x1)
            batched_pad_val_0.append(pad_val_0)
            batched_pad_val_1.append(pad_val_1)

            if len(batched_data) == batch_size or (
                x0 == upper_left_x0[-1] and x1 == upper_left_x1[-1]
            ):
                bar.update(len(batched_data))

                # Run it through model
                with torch.no_grad():
                    batched_data = np_to_var(
                        hwc_to_bchw(batched_data), gpu_no_of_var(net)
                    )
                    batched_data = (
                        batched_data.float()
                        if precision == "float"
                        else batched_data.half()
                    )
                    out_patches_torch = net(batched_data)

                    # Softmax
                    if apply_softmax:
                        out_patches_torch = F.softmax(out_patches_torch, dim=1)

                    # Argmax for classifications
                    if apply_classifier:
                        _, out_patches_torch = torch.max(
                            out_patches_torch, dim=1, keepdims=True
                        )

                out_patches = bcwh_to_hwc(var_to_np(out_patches_torch))
                del out_patches_torch  # Make sure output is flushed from GPU

                # Make output array (We do this here since it will then be agnostic to the number of output channels)
                if len(predictions) == 0:
                    predictions = np.concatenate(
                        [
                            data[
                                : -(patch_overlap[0] * 2),
                                : -(patch_overlap[1] * 2),
                                0:1,
                            ]
                            * 0
                        ]
                        * out_patches[0].shape[2],
                        -1,
                    )

                # Loop through samples in batch
                for i in range(len(batched_data)):

                    # Remove eventual padding related to edges
                    out_patch = out_patches[i][
                        0 : patch_size[0] - batched_pad_val_0[i],
                        0 : patch_size[1] - batched_pad_val_1[i],
                        :,
                    ]

                    # Remove eventual padding related to overlap between data_patches
                    out_patch = out_patch[
                        patch_overlap[0] : -patch_overlap[0],
                        patch_overlap[1] : -patch_overlap[1],
                        :,
                    ]

                    # Insert output_patch in out array
                    predictions[
                        batched_x0[i] : batched_x0[i] + out_patch.shape[0],
                        batched_x1[i] : batched_x1[i] + out_patch.shape[1],
                        :,
                    ] = out_patch

                # Empty batch-lists
                batched_data = []
                batched_x0 = []
                batched_x1 = []
                batched_pad_val_0 = []
                batched_pad_val_1 = []

    return predictions


def put_on_gpu_like(cpu_var, gpu_var):
    """
    Take a variable and put on same gpu as gpu_var
    Args:
        cpu_var (torch.autograd.Variable): Variable to put on GPU
        gpu_var (torch.autograd.Variable): Variable to get GPU number from

    Returns:
        Variable

    """
    gpu_no = gpu_no_of_var(gpu_var)
    if type(gpu_no) == int:
        cpu_var = cpu_var.cuda(gpu_no)
    return cpu_var


def gpu_no_of_var(var):
    """
    Get the GPU-no of a variable
    Args:
        var (torch.autograd.Variable):

    Returns:
        (int) if variable is on GPU
        (false) if variable is not on GPU

    """
    try:
        is_cuda = next(var.parameters()).is_cuda
    except:
        is_cuda = var.is_cuda

    if is_cuda:
        try:
            return next(var.parameters()).get_device()
        except:
            return var.get_device()
    else:
        return False


# Take a pytorch variable and make numpy


def var_to_np(var, delete_var=False):
    tmp_var = var
    if type(tmp_var) in [np.array, np.ndarray]:
        return tmp_var

    # If input is list we do this for all elements
    if type(tmp_var) == type([]):
        out = []
        for v in tmp_var:
            out.append(var_to_np(v))
        return out

    try:
        tmp_var = tmp_var.cpu()
    except:
        None
    try:
        tmp_var = tmp_var.data
    except:
        None
    try:
        tmp_var = tmp_var.numpy()
    except:
        None

    if type(tmp_var) == tuple:
        tmp_var = tmp_var[0]

    if delete_var:
        del var
    return tmp_var


# Take a numpy variable and make a pytorch variable
def np_to_var(var, gpu_no=False, volatile=False):
    # If input is list we do this for all elements
    if type(var) == type([]):
        out = []
        for v in var:
            out.append(np_to_var(v))
        return out

    # Make numpy object
    if type(var) in [type(0), type(0.0), np.float64]:
        var = np.array([var])
    # Make tensor
    if type(var) in [np.ndarray]:
        var = torch.from_numpy(var.astype("float32")).float()
    # Make Variable
    if "Tensor" in str(type(var)):
        var = Variable(var, volatile=volatile)
    # Put on GPU
    if type(gpu_no) not in (int, bool):
        try:
            gpu_no = gpu_no_of_var(gpu_no)
        except:
            pass

    if type(gpu_no) == int:
        var = var.cuda(int(gpu_no))

    return var
