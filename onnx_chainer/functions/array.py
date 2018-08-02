import chainer
import numpy as np
from onnx import helper
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE

from onnx_chainer import mapping


def convert_Cast(func, input_names, output_names, parameters):
    onnx_op_name = mapping.operators[func.__class__.__name__]
    typ = func.type if isinstance(func.type, np.dtype) else np.dtype(func.type)
    return helper.make_node(
        onnx_op_name, input_names, output_names,
        to=mapping.TENSOR_TYPE_TO_NAME[NP_TYPE_TO_TENSOR_TYPE[typ]]
    ),


def convert_Concat(func, input_names, output_names, parameters):
    onnx_op_name = mapping.operators[func.__class__.__name__]
    return helper.make_node(
        onnx_op_name, input_names, output_names,
        axis=func.axis
    ),


def convert_Copy(func, input_names, output_names, parameters):
    onnx_op_name = mapping.operators[func.__class__.__name__]
    return helper.make_node(
        onnx_op_name, input_names, output_names
    ),


def convert_Depth2Space(
        func, input_names, output_names, parameters):
    onnx_op_name = mapping.operators[func.__class__.__name__]
    return helper.make_node(
        onnx_op_name, input_names, output_names,
        blocksize=func.r
    ),


def convert_GetItem(func, input_names, output_names, parameters):
    onnx_op_name = mapping.operators[func.__class__.__name__]
    return helper.make_node(
        onnx_op_name, input_names, output_names,
        slice=func.slices
    ),


def convert_Pad(func, input_names, output_names, parameters):
    onnx_op_name = mapping.operators[func.__class__.__name__]

    if func.mode not in ['constant', 'reflect', 'edge']:
        raise ValueError(
            '{} mode is not supported in ONNX\'s Pad operation'.format(
                func.mode))

    if 'constant_values' in func.keywords:
        values = func.keywords['constant_values']
        if not isinstance(values, int) and len(values) > 1:
            raise ValueError(
                'ONNX doesn\'t support multiple constant values for Pad '
                'operation')
        elif not isinstance(values, int):
            values = values[0]

        node = helper.make_node(
            onnx_op_name, input_names, output_names,
            mode=func.mode,
            pads=func.pad_bw.tolist(),
            value=values
        )
    else:
        node = helper.make_node(
            onnx_op_name, input_names, output_names,
            mode=func.mode,
            pads=func.pad_bw.ravel().tolist(),
        )

    return node,


def convert_Reshape(func, input_names, output_names, parameters):
    onnx_op_name = mapping.operators[func.__class__.__name__]

    # TODO(mitmul): This part is needed for opset_version > 1
    # # Add tiles and axis to graph
    # shape = np.asarray(func.shape, dtype=np.int64)
    # shape_param = chainer.Parameter(shape)
    # parameters.append(shape_param)
    # input_names.append(str(id(shape_param)))

    return helper.make_node(
        onnx_op_name, input_names, output_names,
        shape=func.shape
    ),


def convert_Space2Depth(func, input_names, output_names, parameters):
    onnx_op_name = mapping.operators[func.__class__.__name__]
    return helper.make_node(
        onnx_op_name, input_names, output_names,
        blocksize=func.r
    ),


def convert_SplitAxis(func, input_names, output_names, parameters):
    onnx_op_name = mapping.operators[func.__class__.__name__]

    if func.indices is not None:
        indices_or_sections = func.indices
    else:
        indices_or_sections = func.sections

    if hasattr(indices_or_sections, '__iter__'):
        split = []
        prev_i = 0
        for i in indices_or_sections:
            split.append(i - prev_i)
            prev_i = i
    else:
        length = func.inputs[0].shape[func.axis] // indices_or_sections
        split = [length for _ in range(indices_or_sections)]

    return helper.make_node(
        onnx_op_name, input_names, output_names,
        axis=func.axis,
        split=split
    ),


def convert_Squeeze(func, input_names, output_names, parameters):
    onnx_op_name = mapping.operators[func.__class__.__name__]

    if func.axis is None:
        axis = []
        for s in func.inputs[0].shape:
            if s == 1:
                axis.append(s)
    else:
        axis = func.axis

    return helper.make_node(
        onnx_op_name, input_names, output_names,
        axes=axis
    ),


def convert_Tile(func, input_names, output_names, parameters):
    onnx_op_name = mapping.operators[func.__class__.__name__]

    # Add tiles and axis to graph
    if isinstance(func.reps, int):
        func.reps = [func.reps]
    tiles = np.asarray(func.reps, dtype=np.float32)

    tiles_param = chainer.Parameter(tiles)
    parameters.append(tiles_param)
    input_names.append(str(id(tiles_param)))

    # In operater version = 1, axis also should be given
    axis = np.array([i for i, _ in enumerate(func.reps)], dtype=np.float32)
    axis_param = chainer.Parameter(axis)
    parameters.append(axis_param)
    input_names.append(str(id(axis_param)))

    node = helper.make_node(onnx_op_name, input_names, output_names)
    return node,


def convert_Transpose(func, input_names, output_names, parameters):
    onnx_op_name = mapping.operators[func.__class__.__name__]

    if func.axes is None:
        node = helper.make_node(onnx_op_name, input_names, output_names)
    else:
        node = helper.make_node(
            onnx_op_name, input_names, output_names,
            perm=func.axes
        )

    return node,


# This is workaround implementation for ONNX not having operator
# corresponding to F.where. This implementation approximates
# F.where(cond, x1, x2) as (cond * x1) + ((1-cond) * x2) but the
# behavior on NaN and infinities are different.
def convert_Where(func, input_names, output_names, parameters):
    typ = func.inputs[1].dtype if isinstance(
        func.inputs[1].dtype, np.dtype) else np.dtype(func.inputs[1].dtype)

    one = chainer.Parameter(np.array(1, dtype=typ))
    parameters.append(one)

    n1_out_name = gensym()
    n2_out_name = gensym()
    n3_out_name = gensym()
    n4_out_name = gensym()
    n5_out_name = gensym()

    n1 = helper.make_node(
        "Cast", [input_names[0]], [n1_out_name],
        to=mapping.TENSOR_TYPE_TO_NAME[NP_TYPE_TO_TENSOR_TYPE[typ]]
    )
    n2 = helper.make_node("Neg", [n1_out_name], [n2_out_name])
    n3 = helper.make_node("Add", [n2_out_name, str(id(one))], [n3_out_name])
    n4 = helper.make_node("Mul", [n1_out_name, input_names[1]], [n4_out_name])
    n5 = helper.make_node("Mul", [n3_out_name, input_names[2]], [n5_out_name])
    n6 = helper.make_node("Add", [n4_out_name, n5_out_name], [output_names[0]])

    return n6, n5, n4, n3, n2, n1


dummy_objects = []


def gensym():
    o = object()
    dummy_objects.append(o)
    return str(id(o))
