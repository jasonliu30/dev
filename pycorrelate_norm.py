"""
Functions to compute linear correlation on discrete signals (uniformly
sampled in time) **or** on point-processes (e.g. timestamps of events).
"""

import numpy as np
import numba

@numba.njit
def ucorrelate(t: np.ndarray, u: np.ndarray, maxlag=None,scale=False,start=0):

    """Compute correlation of two signals defined at uniformly-spaced points.
    Altered to normalize calculation by Albert Pietraszkiewicz 3-16-22
     Altered to add start point by Albert Pietraszkiewicz 3-29-22
    The correlation is defined only for positive lags (including zero).
    The input arrays represent signals defined at uniformily-spaced
    points. This function is equivalent to :func:`numpy.correlate`, but can
    efficiently compute correlations on a limited number of lags.

    Note that binning point-processes with uniform bins, provides
    signals that can be passed as argument to this function.

    Arguments:
        t (array): first signal to be correlated
        u (array): second signal to be correlated
        maxlag (int): number of lags where correlation is computed.
            If None, computes all the lags where signals overlap
            `min(t.size, t.size) - 1`.
        scale (boolean): if True, the correlation at each lag will be divided by sqrt((sum of xi**2)*(sum of yi**2)) , this makes the correlation between -1 and +1
        start (int): the starting point of the lags to calculate.Lags will be calculated from start to start + maxlag
         if start=5 and maxlag=10 lags from 5 to 15 are calculated
    Returns:
        Array contained the correlation at different lags.
        The size of this array is equal to the input argument `maxlag`
        (if defined) or to `min(t.size, u.size) - 1`.

    Example:

        Correlation of two signals `t` and `u`::

            >>> t = np.array([1, 2, 0, 0])
            >>> u = np.array([0, 1, 1])
            >>> pycorrelate.ucorrelate(t, u)
            array([2, 3, 1])

        The same result can be obtained with numpy swapping `t` and `u` and
        restricting the results only to positive lags::

            >>> np.correlate(u, t, mode='full')[t.size - 1:]
            array([2, 3, 1])
    """


    if maxlag is None:
        maxlag = u.size

    maxlag=maxlag+start
    maxlag = int(min(u.size, maxlag))
    C = np.zeros(maxlag-start, dtype=np.double)
    #if scale:
      #  tmax = min(u.size, t.size)
       # umax = tmax
        #denominator=np.sqrt(( u[0:umax]**2).sum())*np.sqrt((t[:tmax]**2).sum())
    for lag in range(start,maxlag):
        tmax = min(u.size - lag, t.size)
        umax = min(u.size, t.size + lag)
        C[lag-start] = (t[:tmax] * u[lag:umax]).sum()
        if scale:
            denominator=np.sqrt(( u[lag:umax]**2).sum())*np.sqrt((t[:tmax]**2).sum())
            # print('denom[', lag-start, ']: ', u[lag:umax], t[:tmax], (u[lag:umax]**2).sum(), (t[:tmax]**2).sum(), denominator)
            if denominator==0:
                C[lag-start]=0
            else:
                C[lag-start]=C[lag-start]/denominator
    return C
