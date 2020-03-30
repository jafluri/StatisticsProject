import tensorflow_probability as tfp
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import GPyOpt


def create_dataset():
    #TODO: implement
    pass

def eval_dataset():
    # TODO: implement
    pass

class DeepGM():

    def __init__(self, param_prior, summary_dim, summary_generator, data_simulator, n_mixtures=1, param_dim=2,
                 eps=1e-37, cholesky=None):
        """
        This class provides you with a deep gaussian mixture model
        :param param_prior: prior of the model parameter (from tensorflow_probability)
        :param summary_dim: dimension of the summary statistics
        :param summary_generator: a function that can be used to generate summary statistics from the data_simulator
        :param data_simulator: A function that simulates data from the model, must be compatible with param_prior
        :param n_mixtures: number of mixtures to use
        :param param_dim: dimension of the paramter
        :param eps: a small value to avoid log of zero
        :param cholesky: a matrix at is applied to each summary for normalization (None -> identity)
        """

        self.param_prior = param_prior
        self.summary_dim = summary_dim
        self.summary_generator = summary_generator
        self.data_simulator = data_simulator
        self.n_mixtures = n_mixtures
        self.param_dim = param_dim

        if self.summary_dim > 25:
            raise ValueError("The dimension of the summary should be less than 25")

        # output dimension
        self.out_dim = self.summary_dim + 1 + int(self.summary_dim * (self.summary_dim + 1) / 2)
        self.out_dim = self.out_dim * n_mixtures

        # create a simple network
        self.model = tf.keras.Sequential([tf.keras.layers.Dense(8,
                                                                activation=tf.nn.relu,
                                                                # input shape required
                                                                input_shape=(self.param_dim,)),
                                          tf.keras.layers.Dense(16, activation=tf.nn.relu),
                                          tf.keras.layers.Dense(32, activation=tf.nn.relu),
                                          tf.keras.layers.Dense(64, activation=tf.nn.relu),
                                          tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                          tf.keras.layers.Dense(256, activation=tf.nn.relu),
                                          tf.keras.layers.Dense(self.out_dim)
                                          ])

        # optimizer
        self.optimizer = tf.keras.optimizers.Adam(1e-4)

        # to avoid 0 stuff
        self.eps = eps

        if cholesky is None:
            self.cholesky = tf.eye(self.summary_dim)
        else:
            self.cholesky = tf.constant(cholesky, dtype=tf.float32)


    def loss_function(self, batch_size, n_same):

        # get the summaries
        summaries, params = self._gen_batch(batch_size, n_same)

        # eval loss
        loss_val = self._loss_function_summaries(summaries, params)

        return loss_val

    def _gen_batch(self, batch_size, n_same):
        # sample from the param space
        params = self.param_prior.sample(batch_size)
        summaries, params = self._gen_batch_from_prams(params, n_same)

        return summaries, params

    def _gen_batch_from_prams(self, params, n_same):
        params = tf.expand_dims(params, axis=1)
        params = tf.multiply(params, tf.ones((1, n_same, 1)))
        params = tf.reshape(params, shape=(-1, self.param_dim))

        # generate the data
        data = self.data_simulator(params)
        # calculate the summaries
        summaries = self.summary_generator(data)

        return summaries, params

    def _loss_function_summaries(self, summaries, params):

        # feed to the network
        output = self.model(params)

        # evaluate the probability
        probs = self._eval_prob(output, summaries)

        # get the kl diver
        kl_div = tf.scalar_mul(-1.0, tf.reduce_mean(tf.math.log(tf.maximum(probs, self.eps))))

        return kl_div

    def _eval_prob(self, network_output, summaries):
        # whiten summaries
        summaries = tf.einsum("ij,aj->ai", self.cholesky, summaries)

        # split the output
        means, weights, cholesky = tf.split(network_output,
                                            num_or_size_splits=[self.summary_dim *
                                                                self.n_mixtures,
                                                                self.n_mixtures,
                                                                int(self.summary_dim *
                                                                    (self.summary_dim + 1) / 2) *
                                                                self.n_mixtures],
                                            axis=1)

        # normalize weights
        weights = tf.math.softmax(weights)

        # split the means and the cholesky part
        mean_splits = tf.split(means, num_or_size_splits=self.n_mixtures, axis=1)
        cholesky_splits = tf.split(cholesky, num_or_size_splits=self.n_mixtures, axis=1)

        probs = []
        for mean, cholesky in zip(mean_splits, cholesky_splits):
            # diff of mean and summary
            diff = tf.subtract(mean, summaries)

            # make upper triang matrix L^T
            upper_triang = tfp.math.fill_triangular(cholesky, upper=True)

            # Get diagonal
            diag = tf.linalg.diag_part(upper_triang)

            # get determinant, add a small number such that the diag is never zero and
            # take the log of it
            log_det = tf.reduce_sum(tf.math.log(tf.maximum(tf.square(diag), self.eps)), axis=1, keepdims=True)

            # get norm(L^T*res) (second part of the likelihood loss)
            norm = tf.reduce_sum(tf.square(tf.einsum('ijk,ik->ij', upper_triang, diff)),
                                 axis=1, keepdims=True)

            # expoential part
            log_exp = tf.scalar_mul(-0.5, norm)

            # sig^-1 = LTL^T the calculated determinant was from
            # the inverse covariance matrix
            #log_det = tf.math.log(tf.add(det, tf.constant(1e-42, dtype=tf.float32)))
            log_fac = tf.constant(self.summary_dim*np.log(2*np.pi), dtype=tf.float32)
            log_det = tf.scalar_mul(0.5, tf.subtract(log_det, log_fac))

            # append to list everything 2d
            probs.append(tf.exp(tf.add(log_exp, log_det)))

        # get actual probs
        probs = tf.concat(probs, axis=1)

        # get the weighted probs
        probs = tf.reduce_sum(tf.multiply(weights, probs), axis=1)

        return probs

    def train_step(self, batch_size, n_same):
        # tape and eval the loss function
        with tf.GradientTape() as tape:
            loss_val = self.loss_function(batch_size, n_same)
        # apply gradients
        self.optimizer.apply_gradients(zip(tape.gradient(loss_val, self.model.trainable_variables),
                                           self.model.trainable_variables))
        return loss_val

    def train_step_dset(self, summaries, params):
        # tape and eval the loss function
        with tf.GradientTape() as tape:
            loss_val = self._loss_function_summaries(summaries, params)
        # apply gradients
        self.optimizer.apply_gradients(zip(tape.gradient(loss_val, self.model.trainable_variables),
                                           self.model.trainable_variables))
        return loss_val

    def optimize(self, n_steps=10000, batch_size=4, n_same=512, progress=True, use_dset=True,
                 dset_design="random", cache_after=None):
        """
        Optimize the deep Gaussian mixture model
        :param n_steps: number of steps
        :param batch_size: number of different parameter sets per step
        :param n_same: number of simulated maps per different parameter set
        :param progress: show progress of the training
        :param use_dset: use the tf.Dataset API, this can in some cases lead to a speed up
        :param dset_design: random -> draw the parameter in the dataset random, latin -> create a latin hypercube
        :param cache_after: cache the dataset after this many iterations (repeat the dataset afer that) this can
        lead to a massive speed up once everything is cache. However, it can lead to overfitting.
        """


        if not use_dset:
            # graph everything
            @tf.function
            def single_step():
                return self.train_step(batch_size, n_same)

            # loop
            if progress:
                with tqdm(range(n_steps)) as pbar:
                    for i in pbar:
                        loss_val = single_step()
                        pbar.set_postfix(loss_val=loss_val.numpy(), refresh=False)
            else:
                for i in range(n_steps):
                    loss_val = single_step()
                    if i % 100 == 0:
                        print(loss_val)

        else:
            # create a batch generator
            def data_gen():
                # number of steps
                if cache_after is not None:
                    gen_steps = cache_after
                else:
                    gen_steps = n_steps
                if dset_design == "random":
                    for i in range(gen_steps):
                        summaries, params = self._gen_batch(batch_size, n_same)
                        yield summaries, params
                elif dset_design == "latin":
                    # create efficent latin hyper cube
                    space = GPyOpt.Design_space(space=[{'name': 'var', 'type': 'continuous',
                                                        'domain': (l, h)} for l, h in zip(self.param_prior.low,
                                                                                          self.param_prior.high)])
                    params = GPyOpt.experiment_design.initial_design('latin', space, gen_steps*batch_size)
                    # create a dset
                    params = np.asarray(params, dtype=np.float32)
                    params = np.split(params, axis=0, indices_or_sections=gen_steps)
                    # cycle
                    for param in params:
                        summaries, params = self._gen_batch_from_prams(param, n_same)
                        yield summaries, params

                else:
                    raise ValueError("Design of the Dataset <{}> is not supported...".format(dset_design))

            # create a tf dataset
            dset = tf.data.Dataset.from_generator(data_gen, output_types=(tf.float32, tf.float32),
                                                  output_shapes=(tf.TensorShape([batch_size*n_same, self.summary_dim]),
                                                                 tf.TensorShape([batch_size*n_same, self.param_dim])))

            if cache_after is not None:
                # cache and shuffle
                dset = dset.cache()
                print("Starting cache...")
                dset = dset.repeat()
                print("Filling shuffle buffer...")
                dset = dset.shuffle(100)

            # prefetch
            print("Prefetching data...")
            print("This may take a while... (depending on cache size)", flush=True)
            dset.prefetch(10)

            # the train function
            @tf.function
            def single_step(summaries, params):
                return self.train_step_dset(summaries, params)

            # loop
            counter = 0
            if progress:
                with tqdm(dset, total=n_steps) as pbar:
                    for summaries, params in pbar:
                        loss_val = single_step(summaries, params)
                        pbar.set_postfix(loss_val=loss_val.numpy(), refresh=False)
                        counter += 1
                        if counter == n_steps:
                            break
            else:
                for summaries, params in dset:
                    loss_val = single_step(summaries, params)
                    if counter % 100 == 0:
                        print(loss_val)
                    counter += 1
                    if counter == n_steps:
                        break

    def prob_from_observation(self, params, observation):
        """
        Evaluates the unnormalized probability of a set of params given the summary statistics of a observation
        :param params: 2d array of parameters, each column contains a set
        :param observation: array of the summary statistics of the observation
        :return: the unnormalized probability of p(param | observation)
        """
        observation = np.asarray(np.atleast_2d(observation), dtype=np.float32)

        # feed to the network
        output = self.model(params)

        # evaluate the probability
        probs = self._eval_prob(output, observation)

        return probs

    def save_model(self, path):
        """
        Save the trainable parameter of the neural network
        :param path: path to save file (no suffix!)
        """
        self.model.save_model(path)

    def load_model(self, path):
        """
        Load the trainable parameter of the neural network
        :param path: path to save file (same as used for save_model)
        """
        self.model.load_weights(path)
