import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
from tqdm import tqdm


def KL_annealing(step, reset_pos=10000, split=0.5, lower=0.0, upper=1.0, type='linear'):
    # Fu, Hao, et al. "Cyclical annealing schedule: A simple approach to mitigating kl vanishing." arXiv preprint arXiv:1903.10145 (2019).

    # step: An integer representing the current step in the training process.
    # reset_pos: An integer specifying the position at which to reset the step counter if it exceeds this value. Default is 10000.
    # split: A float value between 0 and 1 representing the fraction of the reset cycle at which the annealing weight switches from its lower to upper bound. Default is 0.5.
    # lower: A float specifying the lower bound of the annealing weight. Default is 0.0.
    # upper: A float specifying the upper bound of the annealing weight. Default is 1.0.
    # type: A string indicating the type of annealing to use. It can be one of 'linear', 'sigmoid', or 'cosine'. Default is 'linear'.

    while step > reset_pos:
        step -= reset_pos

    if step >= int(reset_pos * split):
        return tf.constant(upper, dtype=tf.float32)
    frac = step / (int(reset_pos * split))

    if type == 'linear':
        return tf.constant(frac, dtype=tf.float32) * (upper - lower) + lower
    if type == 'sigmoid':
        return tf.constant(lower, dtype=tf.float32) + (upper - lower) / (1 + tf.exp(-10 * (frac - 0.5)))
    if type == 'cosine':
        return tf.constant(lower, dtype=tf.float32) + 0.5 * (1 - tf.cos(np.pi * frac)) * (upper - lower)

def split_uneven(x, n, axis=0):
    # splits tensor into a list of n subtensors. 
    # Required because sometimes the input doesn't split evenly into n.

    shape = tf.shape(x)[axis] 
    size_of_splits = tf.math.floor(shape / n)
    new_size = tf.cast(size_of_splits * n, tf.int32)

    ls = tf.split(x[:new_size], n, axis)
    ls.append(x[new_size:])
    return ls


    
def fit(_model, dataset, optimizer, loss_fn, epochs = 1, reset_pos=2000, split_steps = 3, prediction_steps =3):
    # _model: The machine learning model to be trained.
    # dataset: The dataset used for training.
    # optimizer: The optimizer used for optimizing the model's parameters.
    # loss_fn: The loss function used for calculating the loss during training.
    # epochs: The number of epochs (iterations over the entire dataset) for training. Default is 1.
    # reset_pos: An integer indicating the position at which to reset the KL annealing. Default is 2000.
    # split_steps: An integer indicating the number of steps to split. Default is 3.
    # prediction_steps: An integer indicating the number of steps for prediction. Default is 3.
    @tf.function
    def train_step(_model, x, y, loss_fn, optimizer, kl_w=1, prediction_steps=3, split_steps=3):
        # _model: The machine learning model to be trained.
        # x: The input data batch.
        # y: The corresponding labels batch.
        # loss_fn: The loss function used for calculating the loss.
        # optimizer: The optimizer used for optimizing the model's parameters.
        # kl_w: The weight for the KL divergence term. Default is 1.
        # prediction_steps: The number of prediction steps used for computing the predictive distribution parameters. Default is 3, higher is better but slower.
        # split_steps: The number of steps to split the input data and labels - don't want to use the same samples for every run in a batch. Default is 3, higher is better but slower.

        with tf.GradientTape() as tape:
            nll = []
            for xn, yn in zip(split_uneven(x, split_steps), split_uneven(y, split_steps)):
                pred = [_model(xn, training = True) for _ in range(prediction_steps)]

                means = tf.convert_to_tensor([p.mean() for p in pred])
                vars = tf.convert_to_tensor([p.variance() for p in pred])

                mean = tf.reduce_mean(means, 0)
                std = tf.math.sqrt(tf.reduce_mean(vars, 0) + tf.math.reduce_variance(means, 0))

                y_hat = tfp.distributions.Normal(mean, std)
                nll.append(loss_fn(yn, y_hat))
            
            nll = tf.reduce_mean(tf.concat(nll, 0))
            kl = kl_w*tf.reduce_mean(tf.concat([_model.layers[0].KL_rec_kernel,
                                            _model.layers[0].KL_kernel,
                                            _model.layers[0].KL_bias, 
                                            _model.layers[2].kl_penalty
                                            ], 0))

            loss_value = kl + nll
        grads = tape.gradient(loss_value, _model.trainable_weights)
        optimizer.apply_gradients(zip(grads, _model.trainable_weights))
        return nll, kl

    history = []
    for epoch in range(epochs):
        nll_total = 0
        kl_total = 0
        steps_counter = 0

        with tqdm(dataset, unit="batch") as tepoch:
            step = 1
            for x_batch_train, y_batch_train in tepoch:
                kl_w = KL_annealing(steps_counter, reset_pos=reset_pos, split=0.5, lower = 0.0, upper = 1.0, type = 'cosine')
                tepoch.set_description(f"Epoch {epoch+1}")

                nll_value, kl_value = train_step(_model, 
                                                 x_batch_train, 
                                                 y_batch_train, 
                                                 loss_fn, 
                                                 optimizer, 
                                                 kl_w = kl_w, 
                                                 prediction_steps=prediction_steps, 
                                                 split_steps=split_steps
                                                )

                nll_total += tf.reduce_mean(nll_value).numpy()
                kl_total += tf.reduce_mean(kl_value).numpy()

                tepoch.set_postfix(nll=nll_total/step, kl=kl_total/step)

                step+=1
                steps_counter += 1
        history.append({'nll': nll_total / (step + 1), 'kl': kl_total / (step + 1)})
    return _model, history