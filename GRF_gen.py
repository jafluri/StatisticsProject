# imports
#--------
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

# Hyper params
#-------------
# seeds
seed_real = 37
seed_imag = 42

# normalization
skip_norm = False

# number of parameter
n_params = 2

# get lpix
npix = 64
angle = 50.0
lpix = 360.0 / angle

# l space
#--------

def rfftfreq(n,d=1.0):
    """
    FFT frequency for for rfft
    """
    val = 1.0/(n*d)
    n_half = n//2 + 1
    results = np.arange(0, n_half, dtype=int)
    return results * val

lx = rfftfreq(npix) * npix * lpix
ly = np.fft.fftfreq(npix) * npix * lpix
# Compute the multipole moment of each FFT pixel
l = tf.constant(np.sqrt(lx[np.newaxis, :] ** 2 + ly[:, np.newaxis] ** 2), dtype=tf.float32)

def random_number_gen(batch_size):
    """
    :param batch_size: batch size
    :return: real and imag fft parts with symmetry properties enforces
    """
    shape_1 = (batch_size, l.shape[0], l.shape[1]-2)
    shape_2 = (batch_size, l.shape[0], 1)


    # Generate the random numbers with given shape and pixel narmalization
    real_part_batch_1 = tf.random.normal(shape=shape_1, mean=0.0, stddev=1.0, dtype=tf.float32,
                                         seed=seed_real)
    real_part_batch_2 = tf.random.normal(shape=shape_2, mean=0.0, stddev=1.0, dtype=tf.float32,
                                         seed=seed_real+1)
    real_part_batch_3 = tf.random.normal(shape=shape_2, mean=0.0, stddev=1.0, dtype=tf.float32,
                                         seed=seed_real+2)

    # real permutation matrix
    n = l.shape[0]
    real_perm_matrix = np.eye(n)
    real_perm_matrix[n//2+1:] = real_perm_matrix[1:n//2][::-1]
    # remove constant term
    real_perm_matrix[0] = 0
    real_perm_matrix = tf.convert_to_tensor(real_perm_matrix, dtype=tf.float32)

    # enforce constraints on the stripes left and right
    real_part_batch_2 = tf.einsum('ij,ajk->aik', real_perm_matrix, real_part_batch_2)
    real_part_batch_3 = tf.einsum('ij,ajk->aik', real_perm_matrix, real_part_batch_3)

    # concat
    real_part_batch = tf.concat([real_part_batch_2, real_part_batch_1, real_part_batch_3], axis=-1)

    # imaginary part
    # Generate the random numbers with given shape and pixel narmalization
    imag_part_batch_1 = tf.random.normal(shape=shape_1, mean=0.0, stddev=1.0, dtype=tf.float32,
                                         seed=seed_imag)
    imag_part_batch_2 = tf.random.normal(shape=shape_2, mean=0.0, stddev=1.0, dtype=tf.float32,
                                         seed=seed_imag + 1)
    imag_part_batch_3 = tf.random.normal(shape=shape_2, mean=0.0, stddev=1.0, dtype=tf.float32,
                                         seed=seed_imag + 2)

    # imag permutation matrix
    n = l.shape[0]
    imag_perm_matrix = np.eye(n)
    imag_perm_matrix[n // 2 + 1:] = -imag_perm_matrix[1:n // 2][::-1]
    imag_perm_matrix[0] = 0
    imag_perm_matrix[n//2] = 0
    imag_perm_matrix = tf.convert_to_tensor(imag_perm_matrix, dtype=tf.float32)

    # enforce constraints on the stripes left and right
    imag_part_batch_2 = tf.einsum('ij,ajk->aik', imag_perm_matrix, imag_part_batch_2)
    imag_part_batch_3 = tf.einsum('ij,ajk->aik', imag_perm_matrix, imag_part_batch_3)

    # concat
    imag_part_batch = tf.concat([imag_part_batch_2, imag_part_batch_1, imag_part_batch_3], axis=-1)

    # normalize
    real_part_batch = tf.scalar_mul(lpix / (2.0 * np.pi), real_part_batch)
    imag_part_batch = tf.scalar_mul(lpix / (2.0 * np.pi), imag_part_batch)

    return real_part_batch, imag_part_batch

def artificial_power_func(params, l, l_min=50, l_max=3000):
    """
    Power spectrum function from Yaniv (Semester project)
    Fisher info is not dependent on params (constant over whole space)
    Constraints are not degenerate

    Good prior values
    "a_prior", [1.0, 2.0], "Prior range for the amplitude")
    "s_prior", [-2.0, -1.0], "Prior range for the slope"

    params = [a, s]
    :param params: params (shape[-1]==2)
    :param l: the l values to evaluate the spectrum (2d shape)
    :param l_min: normalization factor (works for 5 deg maps)
    :param l_max: normalization factor (works for 5 deg maps)
    :return: powerspectrum evaluated at the input params, shape [prod(params.shape[:-1]], l.shape]
    """
    # get a and s
    a, s = tf.split(params, num_or_size_splits=2, axis=-1)
    a = tf.reshape(a, shape=[-1])
    s = tf.reshape(s, shape=[-1])

    # exp of a
    exp_a = tf.exp(tf.reshape(a, shape=(-1,1,1)))

    # some prep
    prep_l = tf.divide(tf.subtract(tf.expand_dims(tf.scalar_mul(2.0, l), axis=0), l_min + l_max), l_max - l_min)

    # exponent of s
    prep_s = tf.reshape(s, shape=(-1,1,1))

    # finalize
    return tf.multiply(exp_a, tf.exp(tf.multiply(prep_s, prep_l)))


tf_l_inter = np.load("grid_l.npy")
tf_interpol_data = (np.load("TF_power_spectra.npy") + 2.5e-9).astype(np.float32)
tf_log_l_min = np.array([np.min(np.log(tf_l_inter))], dtype=np.float32)
tf_log_l_max = np.array([np.max(np.log(tf_l_inter))], dtype=np.float32)

def power_func(params, l):
    # test the tf interpolator
    tf_pl = tfp.math.batch_interp_regular_nd_grid(x=params,
                                                  x_ref_min=np.array([0.1, 0.4], dtype=np.float32),
                                                  x_ref_max=np.array([0.5, 1.4], dtype=np.float32),
                                                  y_ref=tf_interpol_data,
                                                  axis=-3)
    log_l = tf.math.log(l)
    tf_pl = tfp.math.batch_interp_regular_1d_grid(x=log_l,
                                                  x_ref_min=tf_log_l_min,
                                                  x_ref_max=tf_log_l_max,
                                                  y_ref=tf.expand_dims(tf_pl, 1),
                                                  fill_value_below=None,
                                                  fill_value_above=None,
                                                  axis=-1)
    return tf_pl

def power_spectrum(params, l, no_noise=True):
    """
    This function retruns the power spectrum for a given set of params at given l
    :param params: 1 or 2 dimensional array. Either 1d array of length 2 containing exactly 1 parameter set or
    a 2 dimensional array where each column is a parameter set
    :param l: 1 or 2 dimensional array containing the l values that should be evaluated
    :param no_noise: Do not add observational noise to the power spectrum
    :return: The weak power spectrum evaluated at the given model parameter and l. The first dimension corresponds
    to the different parameter.
    """
    if not isinstance(l, np.ndarray):
        raise ValueError("l needs to be at least a 1d array...")

    # get the dims and make it 2d
    param_dim = params.ndim
    params = np.atleast_2d(params)
    l_dim = l.ndim
    l = np.atleast_2d(l)

    # eval tf power spectrum and get the numpy value
    tf_pl = power_func(params.astype(np.float32), l.astype(np.float32))
    spectrum = tf_pl.numpy()

    if no_noise:
        spectrum -= 2.5e-9

    # squeeze
    if l_dim == 1 and param_dim == 1:
        return np.squeeze(spectrum, axis=(0,1))
    if l_dim == 1:
        return np.squeeze(spectrum, axis=1)
    return spectrum

def base_generator(params, real_part_batch, imag_part_batch):

    # get the power spectrum
    Pl = power_func(params, l)

    # scale the random numers
    real_part = tf.multiply(tf.sqrt(tf.scalar_mul(0.5, Pl)), real_part_batch)
    imag_part = tf.multiply(tf.sqrt(tf.scalar_mul(0.5, Pl)), imag_part_batch)

    # Get map in real space and return
    if not skip_norm:
        real_part = tf.scalar_mul(l.shape[0] ** 2, real_part)
        imag_part = tf.scalar_mul(l.shape[0] ** 2, imag_part)

    # cast to complex
    ft_map = tf.add(tf.cast(real_part, dtype=tf.complex64),
                    tf.scalar_mul(1j, tf.cast(imag_part, dtype=tf.complex64)))


    # ifft and cast
    m = tf.signal.irfft2d(ft_map)

    # apply transorm
    m = 200.0*m   
    m = tf.where(m < 0, (tf.exp(m) - 1.0), m)
    m = m/200.0
 
    # give it a channel dimension (batch_size, npix, npix, 1)
    m = tf.expand_dims(m, axis=-1)
    return m

@tf.function(input_signature=[tf.TensorSpec(shape=[None, n_params], dtype=tf.float32)])
def GRF_generator(params):

    # get shape as tensor
    batch_size = tf.shape(params)[0]

    # generate random numbers with given shape
    real_part_batch, imag_part_batch = random_number_gen(batch_size)

    # generate the maps
    maps = base_generator(params, real_part_batch, imag_part_batch)

    return maps

fiducial_param = np.array([[0.25, 0.84]])
obs = GRF_generator(fiducial_param)

def get_true_log_prob(observed_GRF):

    # handle the observation
    observed_GRF = tf.convert_to_tensor(observed_GRF, dtype=tf.float32)
    observed_GRF = tf.reshape(observed_GRF, shape=(1, npix, npix))

    # undo non-lin transform
    observed_GRF = observed_GRF*200.0
    observed_GRF = tf.where(observed_GRF < 0, tf.math.log(observed_GRF + 1.0), observed_GRF)
    observed_GRF = observed_GRF/200.0

    # convert to Fourier space
    ft_map = tf.signal.rfft2d(observed_GRF)

    # get real and imag
    real_part = tf.math.real(ft_map)
    imag_part = tf.math.imag(ft_map)

    if not skip_norm:
        real_part = tf.divide(real_part, l.shape[0] ** 2)
        imag_part = tf.divide(imag_part, l.shape[0] ** 2)


    # get rid of pixnorm
    real_part = tf.scalar_mul((2.0 * np.pi) / lpix, real_part)
    imag_part = tf.scalar_mul((2.0 * np.pi) / lpix, imag_part)

    # masking
    n = npix
    mask_real = np.ones((1, n, n // 2 + 1))
    mask_real[:, :n // 2, 0] = 0
    mask_real[:, n // 2 + 1:, -1] = 0
    mask_real = tf.constant(mask_real, dtype=tf.float32)

    mask_imag = np.ones((1, n, n // 2 + 1))
    mask_imag[:, :n // 2 + 1, 0] = 0
    mask_imag[:, :n // 2 + 1, -1] = 0
    mask_imag = tf.constant(mask_imag, dtype=tf.float32)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, n_params], dtype=tf.float32)])
    def log_prob(param):
        spectra = power_func(param, l)

        tf_pdf_real = tf.scalar_mul(-1.0, tf.divide(tf.square(real_part), spectra))
        tf_pdf_real = tf.subtract(tf_pdf_real, tf.scalar_mul(0.5, tf.math.log(tf.scalar_mul(np.pi, spectra))))
        tf_log_pdf_real = tf.reduce_sum(tf.multiply(tf_pdf_real, mask_real), axis=(1, 2))

        # handle the imag part
        tf_pdf_imag = tf.scalar_mul(-1.0, tf.divide(tf.square(imag_part), spectra))
        tf_pdf_imag = tf.subtract(tf_pdf_imag, tf.scalar_mul(0.5, tf.math.log(tf.scalar_mul(np.pi, spectra))))
        tf_log_pdf_imag = tf.reduce_sum(tf.multiply(tf_pdf_imag, mask_imag), axis=(1, 2))

        return tf.add(tf_log_pdf_real, tf_log_pdf_imag)

    return log_prob
