import collections

import torch
import torch.multiprocessing as mp
import numpy as np

global_data = {}


def to_array(output_type, output):
    if output_type == "list":
        output = [to_array("array", i) for i in output]
        return output
    else:
        arr, shape = output
        if isinstance(arr, torch.Tensor):
            return arr
        else:
            return np.ctypeslib.as_array(arr).reshape(shape)


def init_worker(i, output_type_, output_):
    global input, fn, output, output_type
    output_type = output_type_
    input = global_data[i][0]
    fn = global_data[i][1]
    output = to_array(output_type, output_)
    # XXX: https://github.com/pytorch/pytorch/issues/17199#issuecomment-493672631
    # XXX: or using intel openmp: https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#intel-openmp-runtime-library-libiomp
    torch.set_num_threads(1)


def process_data_wrapper(i):
    result = fn(input[i])

    if output_type == "list":
        for j in range(len(output)):
            output[j][i] = result[j]
    else:
        output[i] = result


def initialize_output(size, output_t):
    if isinstance(output_t, np.ndarray):
        return mp.Array(np.ctypeslib.as_ctypes_type(output_t.dtype), size * np.prod(output_t.shape).item(), lock=False), (size, *output_t.shape)
    elif isinstance(output_t, torch.Tensor):
        return torch.zeros(size, *output_t.shape, dtype=output_t.dtype).share_memory_(), (size, *output_t.shape)
    elif isinstance(output_t, int):
        return mp.Array("q", size, lock=False), (size,)
    elif isinstance(output_t, float):
        return mp.Array("d", size, lock=False), (size,)
    elif isinstance(output_t, bool):
        return mp.Array("b", size, lock=False), (size,)
    else:
        raise Exception(f"Cannot handle type {type(output_t)} for output")


def map(fn, input, num_workers=None):
    # Input has to be a iterable
    input = list(input)
    # Test the function output
    # Output has to be either a numpy array or primitive type (int, float, bool)
    # List or tuple are allowed if they contain only supported types
    output_t = fn(input[0])

    # Preallocate the output buffer in the main process
    if isinstance(output_t, collections.abc.Sequence):
        output_type = "list"
        output = []
        for i in output_t:
            output.append(initialize_output(len(input), i))
    else:
        output_type = "array"
        output = initialize_output(len(input), output_t)

    # Utilize the linux COW to share the data between processes
    # The input data is read only, so the memory is shared between processes
    while True:
        i = np.random.randint(0, 0xffffffff)
        if i not in global_data:
            global_data[i] = (input, fn)
            break

    try:
        ctx = mp.get_context("fork")
        with ctx.Pool(ctx.cpu_count() if num_workers is None else num_workers, initializer=init_worker, initargs=(i, output_type, output)) as pool:
            pool.map(process_data_wrapper, list(range(len(input))))
        output = to_array(output_type, output)
        return output

    finally:
        del global_data[i]
