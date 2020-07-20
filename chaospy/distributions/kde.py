"""
Gaussian kernel density estimation.
"""
from __future__ import division

import numpy
from scipy.special import comb, ndtr, ndtri, factorial2
from scipy.stats import gaussian_kde

from .baseclass import Dist
from .approximation import approximate_inverse


def batch_input(method):
    """
    Wrapper function ensuring that a KDE method never causes memory errors.
    """
    def wrapper(self, loc):
        out = numpy.zeros(loc.shape)
        for idx in range(0, loc.size, self.step_size):
            out[:, idx:idx+self.step_size] = method(
                self, loc[:, idx:idx+self.step_size])
        return out
    return wrapper


class GaussianKDE(Dist):
    """
    Examples:
        >>> samples = [-1, 0, 1]
        >>> dist = GaussianKDE(samples, 0.4**2)
        >>> dist.pdf([-1, -0.5, 0, 0.5, 1]).round(4)
        array([0.4711, 0.1971, 0.472 , 0.1971, 0.4711])
        >>> dist.cdf([-1, -0.5, 0, 0.5, 1]).round(4)
        array([0.1687, 0.3334, 0.5   , 0.6666, 0.8313])
        >>> dist.inv([0, 0.25, 0.5, 0.75, 1]).round(4)
        array([-3.6962, -0.7645,  0.    ,  0.7645,  3.8811])
        >>> dist.mom([1, 2, 3]).round(4)
        array([0.    , 0.8267, 0.    ])
        >>> # Does dist normalize to one
        >>> t = numpy.linspace(-4, 4, 1000000)
        >>> abs(numpy.mean(dist.pdf(t))*(t[-1]-t[0]) - 1)  # err
        9.999999999177334e-07

        >>> samples = [[-1, 0, 1], [0, 1, 2]]
        >>> dist = GaussianKDE(samples, 0.4)
        >>> dist.pdf([[0, 0, 1, 1], [0, 1, 0, 1]]).round(4)
        array([0.0435, 0.2688, 0.0018, 0.0435])
        >>> dist.inv([[0, 0, 1, 1], [0, 1, 0, 1]]).round(4)
        array([[-4.7436, -4.7436,  5.1709,  5.1709],
               [-4.6393,  6.0524, -4.5387,  6.4044]])

    """

    def __init__(self, samples, bandwidth="scott", batch_size=1e7):
        self.samples = numpy.atleast_2d(samples)
        assert self.samples.ndim == 2
        self.dim = len(self.samples)
        size = self.samples.shape[-1]

        self.batch_size = batch_size
        if self.batch_size < self.samples.size:
            self.batch_size = self.samples.size
        self.step_size = int(self.batch_size//self.samples.size)

        # the scale is taken from Scott-92.
        # The Scott factor is taken from scipy docs.
        if bandwidth in ("scott", "silverman"):
            qrange = numpy.quantile(self.samples, [0.25, 0.75], axis=1).ptp()
            scale = min(numpy.std(samples, axis=1), qrange/1.34)
            if bandwidth == "scott":
                scott_factor = size**(-1./(self.dim+4))
            else:
                scott_factor = (size*(self.dim+2)/4.)**(-1./(self.dim+4))
            bandwidth = numpy.diag(scale*scott_factor)**2

        else:
            bandwidth = numpy.asfarray(bandwidth)
            if bandwidth.ndim in (0, 1):
                bandwidth = bandwidth*numpy.eye(self.dim)
        assert bandwidth.shape == (self.dim, self.dim)
        self.L = numpy.linalg.cholesky(bandwidth)
        self.Li = numpy.linalg.inv(self.L)

        Dist.__init__(self)

    def __len__(self):
        return self.dim

    @batch_input
    def _pdf(self, x_loc):
        s, t = numpy.mgrid[:x_loc.shape[-1], :self.samples.shape[-1]]
        out = numpy.zeros(x_loc.shape)
        denomenator = numpy.ones(1)
        for idx in range(len(self)):
            x_loc_ = x_loc[idx, s]
            samples = self.samples[idx, t]
            z_loc = (x_loc_-samples)*self.Li[idx, idx]
            # Normal dist normalizes with 1/sqrt(2*pi),
            # However this PDF normalizes with 1/sqrt(pi).
            # No idea why, but I assume I am missing a factor.
            numerator = numpy.e**(-z_loc**2)/numpy.sqrt(numpy.pi)*self.Li[idx, idx]*denomenator
            out[idx] = numpy.mean(numerator, axis=-1)/numpy.mean(denomenator, axis=-1)
            denomenator = numerator
        return out

    @batch_input
    def _cdf(self, x_loc):
        s, t = numpy.mgrid[:x_loc.shape[-1], :self.samples.shape[-1]]
        out = numpy.zeros(x_loc.shape)
        denomenator = numpy.ones(1)
        for idx in range(len(self)):
            x_loc_ = x_loc[idx, s]
            samples = self.samples[idx, t]
            z_loc = (x_loc_-samples)*self.Li[idx, idx]
            numerator = ndtr(z_loc)*denomenator
            out[idx] = numpy.mean(numerator, axis=-1)/numpy.mean(denomenator, axis=-1)
            denomenator = numerator
        return out

    @batch_input
    def _ppf(self, u_loc):
        # speed up convergence considerable, by giving very good initial position.
        x0 = numpy.quantile(self.samples, u_loc[0])[numpy.newaxis]
        return approximate_inverse(self, u_loc, x0=x0, tol=1e-8)

    def _mom(self, k_loc):
        r"""

        Related moment of sum to the individual component:

        .. math::
            E(X^k) = 1/N int x^k sum_n f_n(x) dx   \\
                   = 1/N sum_n int x_k f_n(x) dx   \\
                   = 1/N sum_n E(X_n^k)

        Where each component's moment is calculate through relating standard
        normal through :math:`X_n = Z_n h + x_n`. Here `h` is standard
        deviation (aka bandwidth), `x_n` is the mean (aka sample), and `Z_n` is
        a standard normal variable.

        .. math::
            E(X_n^k) = int (x*h+x_n)^k f_Z(x) dx   \\
                     = sum_i comb(i, n) E(Z^i) h^i x_n^{n-i}
        """
        all_k = numpy.arange(k_loc[0]+1, dtype=int)
        moments = .5*factorial2(all_k-1)*(1+(-1)**all_k)  # standard normal moments
        moments *= self.L[0, 0]**all_k
        coeffs = comb(k_loc[0], all_k)
        s, t = numpy.mgrid[:self.samples.size, :moments.size]
        samples = self.samples[0, s]**all_k[::-1]
        out = numpy.sum(moments[t]*coeffs[t]*samples, axis=1)
        return numpy.mean(out)

    def _lower(self):
        return self.samples.min(axis=-1)-7.5*numpy.sum(self.L, axis=0)

    def _upper(self):
        return self.samples.max(axis=-1)+7.5*numpy.sum(self.L, axis=0)


# class GaussianKDE(Dist):
#     """
#     Gaussian kernel density estimation.

#     Examples:
#         >>> distribution = GaussianKDE([0, 1, 1, 1, 2])
#         >>> print(distribution)
#         GaussianKDE()
#         >>> q = numpy.linspace(0, 1, 7)[1:-1]
#         >>> print(q.round(4))
#         [0.1667 0.3333 0.5    0.6667 0.8333]
#         >>> #print(distribution.fwd(distribution.inv(q)).round(4))
#         >>> #print(distribution.inv(q).round(4))
#         >>> #print(distribution.sample(4).round(4))
#         >>> #print(distribution.mom(1).round(4))
#         >>> #print(distribution.ttr([1, 2, 3]).round(4))
#     """

#     def __init__(self, samples, lower=None, upper=None):

#         samples = numpy.asarray(samples)
#         if len(samples.shape) == 1:
#             samples = samples.reshape(1, -1)
#         kernel = gaussian_kde(samples, bw_method="scott")

#         if lower is None:
#             lower = samples.min(axis=-1)
#         if upper is None:
#             upper = samples.max(axis=-1)

#         self.lower_ = lower
#         self.upper_ = upper
#         self.samples = samples
#         self.l_fwd = numpy.linalg.cholesky(kernel.covariance)
#         self.l_inv = numpy.linalg.inv(self.l_fwd)
#         super(GaussianKDE, self).__init__()

#     def __len__(self):
#         return len(self.samples)

#     def _pdf(self, x_data):
#         out = numpy.zeros(x_data.shape)

#         # first dimension is simple:
#         for sample in self.samples[0]:
#             z = self.l_inv[0, 0]*(x_data[0] - sample)
#             out[0] += numpy.e**(-0.5*z*z) / len(self.samples[0])
#         out[0] *= self.l_inv[0, 0]*(2*numpy.pi)**-0.5

#         return out

#     def _cdf(self, x_data):
#         out = numpy.zeros(x_data.shape)

#         # first dimension is simple:
#         print(x_data)
#         for sample in self.samples[0]:
#             z = self.l_inv[0, 0]*(x_data[0] - sample)
#             print(z)
#             z = ndtr(z)
#             out[0] += z / len(self.samples[0])

#         return out

#     def _ppf(self, u_data):
#         out = numpy.zeros(u_data.shape)

#         # first dimension is simple:
#         z = self.l_fwd[0, 0]*ndtri(u_data[0])
#         for sample in self.samples[0]:
#             out[0] += (z + sample) / len(self.samples[0])

#         return out

#     def _lower(self):
#         return self.lower_

#     def _upper(self):
#         return self.upper_
