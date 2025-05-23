    
from keras.layers.core import Layer
import theano.tensor as T
#from keras.engine.topology import Layer
import keras.backend as K
class LRN(Layer):

    def __init__(self, alpha=0.001,k=1,beta=0.75,n=7, **kwargs):
        self.alpha = alpha
        self.k = k
        self.beta = beta
        self.n = n
        super(LRN, self).__init__(**kwargs)
    
    def call(self, x, mask=None):
        b, ch, r, c = x.shape
        half_n = self.n // 2 # half the local region
        input_sqr = T.sqr(x) # square the input
        extra_channels = T.alloc(0., b, ch + 2*half_n, r, c) # make an empty tensor with zero pads along channel dimension
        input_sqr = T.set_subtensor(extra_channels[:, half_n:half_n+ch, :, :],input_sqr) # set the center to be the squared input
        scale = self.k # offset for the scale
        norm_alpha = self.alpha / self.n # normalized alpha
        for i in range(self.n):
            scale += norm_alpha * input_sqr[:, i:i+ch, :, :]
        scale = scale ** self.beta
        x = x / scale
        return x

    def get_config(self):
        config = {"alpha": self.alpha,
                  "k": self.k,
                  "beta": self.beta,
                  "n": self.n}
        base_config = super(LRN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class PoolHelper(Layer):
    
    def __init__(self, **kwargs):
        super(PoolHelper, self).__init__(**kwargs)
    
    def call(self, x, mask=None):
        return x[:,:,1:,1:]
    
    def get_config(self):
        config = {}
        base_config = super(PoolHelper, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Round(Layer):

    def __init__(self, **kwargs):
        super(Round, self).__init__(**kwargs)

    def call(self, x, mask=None):
        return K.switch(x>0.2,x,0)

#        return K.round(x)

    def get_config(self):
        config = {}
        base_config = super(Round, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))    



#class ZeroMaskedEntries(Layer):
#   
#
#
#
#    def compute_mask(self, x, mask=None):
#        if not self.mask_zero:
#            return None
#        else:
#            return K.greater(x, 0.3)
#   
#
#    def __init__(self, **kwargs):
#        self.support_mask = True
#        super(ZeroMaskedEntries, self).__init__(**kwargs)
#
#    def build(self, input_shape):
#        self.output_dim = input_shape[1]
#        self.repeat_dim = input_shape[2]
#
#    def call(self, x, mask=None):
#        mask = K.cast(mask, 'float32')
#        mask = K.repeat(mask, self.repeat_dim)
#        mask = K.permute_dimensions(mask, (0, 2, 1))
#        return x * mask
#
#    def compute_mask(self, input_shape, input_mask=None):
#        return None