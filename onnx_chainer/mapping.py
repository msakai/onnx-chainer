TENSOR_TYPE_TO_NAME = {
    0: 'UNDEFINED',
    1: 'FLOAT',
    2: 'UINT8',
    3: 'INT8',
    4: 'UINT16',
    5: 'INT16',
    6: 'INT32',
    7: 'INT64',
    8: 'STRING',
    9: 'BOOL',
    10: 'FLOAT16',
    11: 'DOUBLE',
    12: 'UINT32',
    13: 'UINT64',
    14: 'COMPLEX64',
    15: 'COMPLEX128',
}

operators = {
    # Activation
    'ELU': 'Elu',
    'HardSigmoid': 'HardSigmoid',
    'LeakyReLU': 'LeakyRelu',
    'LogSoftmax': 'LogSoftmax',
    'PReLUFunction': 'PRelu',
    'ReLU': 'Relu',
    'Sigmoid': 'Sigmoid',
    'Softmax': 'Softmax',
    'Softplus': 'Softplus',
    'Tanh': 'Tanh',

    # Array
    'Cast': 'Cast',
    'Concat': 'Concat',
    'Copy': 'Identity',
    'Depth2Space': 'DepthToSpace',
    'Pad': 'Pad',
    'Reshape': 'Reshape',
    'Space2Depth': 'SpaceToDepth',
    'SplitAxis': 'Split',
    'Squeeze': 'Squeeze',
    'Tile': 'Tile',
    'Transpose': 'Transpose',
    'Where': 'Where',

    # Connection
    'Convolution2DFunction': 'Conv',
    'ConvolutionND': 'Conv',
    'Deconvolution2DFunction': 'ConvTranspose',
    'DeconvolutionND': 'ConvTranspose',
    'EmbedIDFunction': 'Gather',
    'LinearFunction': 'Gemm',

    # Math
    'Add': 'Add',
    'AddConstant': 'Add',
    'Absolute': 'Abs',
    'Div': 'Div',
    'Mul': 'Mul',
    'Neg': 'Neg',
    'PowVarConst': 'Pow',
    'Sub': 'Sub',
    'Clip': 'Clip',
    'Exp': 'Exp',
    'Identity': 'Identity',
    'MatMul': 'Gemm',
    'Maximum': 'Max',
    'Minimum': 'Min',
    'Sqrt': 'Sqrt',
    'Sum': 'ReduceSum',

    # Noise
    'Dropout': 'Dropout',

    # Pooling
    'AveragePooling2D': 'AveragePool',
    'AveragePoolingND': 'AveragePool',
    'MaxPooling2D': 'MaxPool',
    'MaxPoolingND': 'MaxPool',

    # Normalization
    'BatchNormalization': 'BatchNormalization',
    'FixedBatchNormalization': 'BatchNormalization',
    'LocalResponseNormalization': 'LRN',

}
