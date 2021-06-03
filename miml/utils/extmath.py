"""
Extended math utilities.
"""

import warnings

import mipylib.numeric as np

def fast_logdet(A):
    """Compute log(det(A)) for A symmetric.
    Equivalent to : np.log(nl.det(A)) but more robust.
    It returns -Inf if det(A) is non positive or is not defined.
    Parameters
    ----------
    A : array-like
        The matrix.
    """
    sign, ld = np.linalg.slogdet(A)
    if not sign > 0:
        return -np.inf
    return ld

def svd_flip(u, v, u_based_decision=True):
    """Sign correction to ensure deterministic output from SVD.
    Adjusts the columns of u and the rows of v such that the loadings in the
    columns in u that are largest in absolute value are always positive.
    Parameters
    ----------
    u : ndarray
        u and v are the output of `linalg.svd` or
        :func:`~sklearn.utils.extmath.randomized_svd`, with matching inner
        dimensions so one can compute `np.dot(u * s, v)`.
    v : ndarray
        u and v are the output of `linalg.svd` or
        :func:`~sklearn.utils.extmath.randomized_svd`, with matching inner
        dimensions so one can compute `np.dot(u * s, v)`.
        The input v should really be called vt to be consistent with scipy's
        ouput.
    u_based_decision : bool, default=True
        If True, use the columns of u as the basis for sign flipping.
        Otherwise, use the rows of v. The choice of which variable to base the
        decision on is generally algorithm dependent.
    Returns
    -------
    u_adjusted, v_adjusted : arrays with the same dimensions as the input.
    """
    if u_based_decision:
        # columns of u, rows of v
        max_abs_cols = np.argmax(np.absolute(u), axis=0)
        signs = np.sign(u[max_abs_cols, range(u.shape[1])])
        u *= signs
        v *= signs[:, np.newaxis]
    else:
        # rows of v, columns of u
        max_abs_rows = np.argmax(np.absolute(v), axis=1)
        signs = np.sign(v[range(v.shape[0]), max_abs_rows])
        u *= signs
        v *= signs[:, np.newaxis]
    return u, v

def stable_cumsum(arr, axis=None, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum.
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat.
    axis : int, default=None
        Axis along which the cumulative sum is computed.
        The default (None) is to compute the cumsum over the flattened array.
    rtol : float, default=1e-05
        Relative tolerance, see ``np.allclose``.
    atol : float, default=1e-08
        Absolute tolerance, see ``np.allclose``.
    """
    out = np.cumsum(arr, axis=axis, dtype=np.float64)
    expected = np.sum(arr, axis=axis, dtype=np.float64)
    if not np.all(np.isclose(out.take(-1, axis=axis), expected, rtol=rtol,
                             atol=atol, equal_nan=True)):
        warnings.warn('cumsum was found to be unstable: '
                      'its last element does not correspond to sum',
                      RuntimeWarning)
    return out