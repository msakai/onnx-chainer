import unittest

import numpy as np

import chainer
from chainer import testing
import chainer.links as L
import onnx_chainer
from onnx_chainer.testing import test_mxnet

MXNET_SUPPORT = {
    'Convolution2D': True,
    'ConvolutionND': False,
    'DilatedConvolution2D': True,
    'Deconvolution2D': True,
    'EmbedID': False,
    'Linear': True,
}


@testing.parameterize(
    {'link': L.Convolution2D, 'in_shape': (1, 3, 5, 5), 'in_type': np.float32,
     'args': [None, 3, 3, 1, 1]},
    {'link': L.Convolution2D, 'in_shape': (1, 3, 5, 5), 'in_type': np.float32,
     'args': [None, 3, 3, 1, 1, True]},

    {'link': L.ConvolutionND, 'in_shape': (1, 3, 5), 'in_type': np.float32,
     'args': [1, 3, 4, 3, 1, 0]},
    {'link': L.ConvolutionND, 'in_shape': (1, 3, 5), 'in_type': np.float32,
     'args': [1, 3, 4, 3, 1, 0, True]},
    {'link': L.ConvolutionND, 'in_shape': (1, 3, 5, 5, 5),
     'in_type': np.float32, 'args': [3, 3, 4, 3, 1, 0]},

    {'link': L.DilatedConvolution2D, 'in_shape': (1, 3, 5, 5),
     'in_type': np.float32, 'args': [None, 3, 3, 1, 1, 2]},
    {'link': L.DilatedConvolution2D, 'in_shape': (1, 3, 5, 5),
     'in_type': np.float32, 'args': [None, 3, 3, 1, 1, 2, True]},

    {'link': L.Deconvolution2D, 'in_shape': (1, 3, 5, 5),
     'in_type': np.float32, 'args': [None, 3, 4, 2, 0]},
    {'link': L.Deconvolution2D, 'in_shape': (1, 3, 5, 5),
     'in_type': np.float32, 'args': [None, 3, 4, 2, 0, True]},

    {'link': L.EmbedID, 'in_shape': (1, 10), 'in_type': np.int32,
     'args': [5, 8]},

    {'link': L.Linear, 'in_shape': (1, 10), 'in_type': np.float32,
     'args': [None, 8]},
    {'link': L.Linear, 'in_shape': (1, 10), 'in_type': np.float32,
     'args': [None, 8, True]},
)
class TestConnections(unittest.TestCase):

    def setUp(self):

        class Model(chainer.Chain):

            def __init__(self, link, args):
                super(Model, self).__init__()
                with self.init_scope():
                    self.l1 = link(*args)

            def __call__(self, x):
                return self.l1(x)

        self.model = Model(self.link, self.args)
        self.x = np.zeros(self.in_shape, dtype=self.in_type)
        self.fn = self.link.__name__ + '.onnx'

    def test_compatibility(self):
        if MXNET_SUPPORT[self.link.__name__]:
            test_mxnet.check_compatibility(self.model, self.x, self.fn)
        else:
            onnx_chainer.export(self.model, self.x)
