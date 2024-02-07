import uuid
import tensorflow.compat.v2 as tf

from keras import activations
from keras import backend
from keras import constraints
from keras import initializers
from keras import regularizers
from keras.engine import base_layer
from keras.engine.input_spec import InputSpec
from keras.layers.rnn import gru_lstm_utils
from keras.layers.rnn import rnn_utils
from keras.layers.rnn.base_rnn import RNN
from keras.layers.rnn.dropout_rnn_cell_mixin import DropoutRNNCellMixin
from keras.utils import tf_utils

# isort: off
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export

@tf.function
def split_uneven(x, n, axis=0):
    shape = tf.shape(x)[axis] 
    size_of_splits = tf.math.ceil(shape / n)
    new_size = tf.cast(size_of_splits * (n-1), tf.int32)

    ls = tf.split(x[:new_size], (n-1), axis)
    ls.append(x[new_size:])
    return ls


@tf.function
def split_first(x, n, axis=0):
    shape = tf.shape(x)[axis] 
    size_of_splits = tf.math.ceil(shape / n)
    new_shape = tf.cast(size_of_splits * n, tf.int32)

    needed = new_shape - shape

    addition = tf.zeros(tf.concat([[needed], tf.shape(x)[1:]], 0))
    x_new = tf.concat([x, addition], 0)
    
    return tf.reshape(x_new, tf.concat([[n, -1], tf.shape(x)[1:]], 0))


RECURRENT_DROPOUT_WARNING_MSG = ("recurrent dropout is not supported!")

@keras_export("keras.layers.GRUCell", v1=[])
class GRU_Cell_Variational(DropoutRNNCellMixin, base_layer.BaseRandomLayer):
    def __init__(
        self,
        units,
        kernel_prior_fn,
        kernel_posterior_fn,
        recurrent_kernel_prior_fn,
        recurrent_kernel_posterior_fn,
        bias_prior_fn,
        bias_posterior_fn,
        activation="tanh",
        recurrent_activation="sigmoid",
        use_bias=True,
        scale = None,
        kernel_initializer="glorot_uniform",
        recurrent_initializer="orthogonal",
        bias_initializer="zeros",
        kernel_regularizer=None,
        recurrent_regularizer=None,
        bias_regularizer=None,
        kernel_constraint=None,
        recurrent_constraint=None,
        bias_constraint=None,
        n_samples = 4,
        dropout=0.0,
        kl_weight=None,
        kl_use_exact=False,
        reset_after=True,
        sampling = 'once',
        **kwargs,
    ):
        if units < 0:
            raise ValueError(
                f"Received an invalid value for argument `units`, "
                f"expected a positive integer, got {units}."
            )
        # By default use cached variable under v2 mode, see b/143699808.
        if tf.compat.v1.executing_eagerly_outside_functions():
            self._enable_caching_device = kwargs.pop(
                "enable_caching_device", True
            )
        else:
            self._enable_caching_device = kwargs.pop(
                "enable_caching_device", False
            )

        super().__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias
        self.scale = scale
        self.sampling = sampling
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1.0, max(0.0, dropout))
        self.n_samples = n_samples

        self.kernel_prior_fn = kernel_prior_fn
        self.kernel_posterior_fn = kernel_posterior_fn

        self.recurrent_kernel_prior_fn = recurrent_kernel_prior_fn
        self.recurrent_kernel_posterior_fn = recurrent_kernel_posterior_fn

        self.bias_prior_fn = bias_prior_fn
        self.bias_posterior_fn = bias_posterior_fn

        self.reset_after = reset_after
        self.state_size = self.units
        self.output_size = self.units

        self.kl_penalty = 0
        self._kl_divergence_fn = _make_kl_divergence_penalty(
            kl_use_exact, weight=kl_weight)

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        super().build(input_shape)
        input_dim = input_shape[-1]

        self.kernel_shape = (input_dim, self.units * 3)
        self.recurrent_kernel_shape = (self.units, self.units * 3)
        self._kernel_prior = self.kernel_prior_fn(
            shape=self.kernel_shape,
            name="kernel_prior",
            scale = self.scale,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            n_samples = self.n_samples
        )

        self._kernel_posterior = self.kernel_posterior_fn(
            shape=self.kernel_shape,
            name="kernel_posterior",
            scale = self.scale,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            n_samples = self.n_samples
        )

        self._recurrent_kernel_prior = self.recurrent_kernel_prior_fn(
            shape=self.recurrent_kernel_shape,
            name="recurrent_kernel_prior",
            scale = self.scale,
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
            n_samples = self.n_samples
        )

        self._recurrent_kernel_posterior = self.recurrent_kernel_posterior_fn(
            shape=self.recurrent_kernel_shape,
            name="recurrent_kernel_posterior",
            scale = self.scale,
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
            n_samples = self.n_samples
        )          

        if self.use_bias:
            if not self.reset_after:
                self.bias_shape = (3 * self.units,)
            else:
                # separate biases for input and recurrent kernels
                # Note: the shape is intentionally different from CuDNNGRU
                # biases `(2 * 3 * self.units,)`, so that we can distinguish the
                # classes when loading and converting saved weights.
                self.bias_shape = (2, 3 * self.units)

            self._bias_prior = self.bias_prior_fn(
                shape=self.bias_shape,
                name="bias_prior",
                scale = self.scale,
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                n_samples = self.n_samples
            )
            self._bias_posterior = self.bias_posterior_fn(
                shape=self.bias_shape,
                name="bias_prior",
                scale = self.scale,
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                n_samples = self.n_samples
            )

        else:
            self.bias = None
        self.built = True

    def get_kernel(self, inputs):
        # Get prior and posterior for kernel and recurrent kernel
        q_kernel = self._kernel_posterior(tf.random.normal([5,]))
        r_kernel = self._kernel_prior(tf.random.normal([5,]))

        q_recurrent_kernel = self._recurrent_kernel_posterior(tf.random.normal([5,]))
        r_recurrent_kernel = self._recurrent_kernel_prior(tf.random.normal([5,]))

        # Calculate KL Divergence
        self.KL_kernel = self._kl_divergence_fn(q_kernel, r_kernel)
        self.KL_rec_kernel = self._kl_divergence_fn(q_recurrent_kernel, r_recurrent_kernel)
        
        self.kl_penalty = tf.reduce_mean(self.KL_kernel) + tf.reduce_mean(self.KL_rec_kernel)
        self.add_loss(self.kl_penalty)

        # Make posterior into useable weights
        self.kernel = tf.convert_to_tensor(q_kernel)
        self.recurrent_kernel = tf.convert_to_tensor(q_recurrent_kernel)

        # Repeat for bias
        if self.use_bias:
            q_bias = self._bias_posterior(tf.random.normal([5,]))
            r_bias = self._bias_prior(tf.random.normal([5,]))
            self.KL_bias = self._kl_divergence_fn(q_bias, r_bias)
            self.add_loss(tf.reduce_mean(self.KL_bias))
            
            self.kl_penalty += self.KL_bias
            self.bias = tf.convert_to_tensor(q_bias)     
        else:
            self.bias=None


    def call(self, inputs, states, training=None):
        h_tm1 = (
            states[0] if tf.nest.is_nested(states) else states
        )  # previous memory

        # dropout stuff, not tested
        dp_mask = self.get_dropout_mask_for_cell(inputs, training, count=3)
        if 0.0 < self.dropout < 1.0:
            inputs = inputs * dp_mask[0]

        # get kernel once the first time the layer is called
        if not self.initialised:
            self.get_kernel(inputs)
            self.initialised = True
        if self.sampling == 'always':
            self.get_kernel(inputs)
            

        if self.use_bias:
            if not self.reset_after:
                input_bias, recurrent_bias = self.bias, None
            else:
                input_bias, recurrent_bias = tf.unstack(self.bias, axis = -2)

        matrix_x = backend.dot(inputs, self.kernel)
        if self.use_bias:
        # biases: bias_z_i, bias_r_i, bias_h_i
            matrix_x = backend.bias_add(matrix_x, input_bias)


        x_z, x_r, x_h = tf.split(matrix_x, 3, axis=-1)

        if self.reset_after:
            matrix_inner = backend.dot(h_tm1, self.recurrent_kernel)
            if self.use_bias:
                matrix_inner = backend.bias_add(matrix_inner, recurrent_bias)

        else:
            matrix_inner = backend.dot(h_tm1, self.recurrent_kernel[:, : 2 * self.units])

        recurrent_z, recurrent_r, recurrent_h = tf.split(
            matrix_inner, [self.units, self.units, -1], axis=-1
        )

        z = self.recurrent_activation(x_z + recurrent_z)
        r = self.recurrent_activation(x_r + recurrent_r)

        if self.reset_after:
            recurrent_h = r * recurrent_h
        else:
            recurrent_h = backend.dot(r * h_tm1, self.recurrent_kernel[:, 2 * self.units :])


        hh = self.activation(x_h + recurrent_h)
        # previous and candidate state mixed by update gate
        h = z * h_tm1 + (1 - z) * hh
        new_state = [h] if tf.nest.is_nested(states) else h
        return h, new_state

    def get_config(self):
        config = {
            "units": self.units,
            "activation": activations.serialize(self.activation),
            "recurrent_activation": activations.serialize(
                self.recurrent_activation
            ),
            "use_bias": self.use_bias,
            "kernel_initializer": initializers.serialize(
                self.kernel_initializer
            ),
            "recurrent_initializer": initializers.serialize(
                self.recurrent_initializer
            ),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "kernel_regularizer": regularizers.serialize(
                self.kernel_regularizer
            ),
            "recurrent_regularizer": regularizers.serialize(
                self.recurrent_regularizer
            ),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "recurrent_constraint": constraints.serialize(
                self.recurrent_constraint
            ),
            "bias_constraint": constraints.serialize(self.bias_constraint),
            "dropout": self.dropout,
            "reset_after": self.reset_after,
        }
        config.update(rnn_utils.config_for_enable_caching_device(self))
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        state = rnn_utils.generate_zero_filled_state_for_cell(
            self, inputs, batch_size, dtype
        )
        self.initialised = False
        return state

def _make_kl_divergence_penalty(
    use_exact_kl=False,
    test_points_reduce_axis=(),  # `None` == "all"; () == "none".
    test_points_fn=tf.convert_to_tensor,
    weight=None):
    """Creates a callable computing `KL[a,b]` from `a`, a `tfd.Distribution`."""

    if use_exact_kl:
        kl_divergence_fn = kullback_leibler.kl_divergence
    else:
        def kl_divergence_fn(distribution_a, distribution_b):
            z = test_points_fn(distribution_a)
            return tf.reduce_mean(
                distribution_a.log_prob(z) - distribution_b.log_prob(z),
                axis=test_points_reduce_axis)

    # Closure over: kl_divergence_fn, weight.
    def _fn(distribution_a, distribution_b):
        """Closure that computes KLDiv as a function of `a` as in `KL[a, b]`."""
        with tf.name_scope('kldivergence_loss'):
            kl = kl_divergence_fn(distribution_a, distribution_b)
            if weight is not None:
                kl = tf.cast(weight, dtype=kl.dtype) * kl
            # Losses appended with the model.add_loss and are expected to be a single
            # scalar, unlike model.loss, which is expected to be the loss per sample.
            # Therefore, we reduce over all dimensions, regardless of the shape.
            # We take the sum because (apparently) Keras will add this to the *post*
            # `reduce_sum` (total) loss.
            # TODO(b/126259176): Add end-to-end Keras/TFP test to ensure the API's
            # align, particularly wrt how losses are aggregated (across batch
            # members).
            return kl

    return _fn

    