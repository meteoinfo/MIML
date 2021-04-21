# -*- coding: utf-8 -*-

from org.meteothink.miml.util import InterpUtil

import mipylib.numeric as np

__all__ = [
    'interp1d','interp2d','griddata'
]

class interp1d(object):
    '''
    Interpolate a 1-D function.

    :param x: (*array_like*) A 1-D array of real values.
    :param y: (*array_like*) A 1-D array of real values. The length of y must be equal to the length of x.
    :param kind: (*boolean*) Specifies the kind of interpolation as a string ('kriging'). Default is ‘kriging’.
    '''
    def __init__(self, x, y, kind='kriging', **kwargs):
        if isinstance(x, list):
            x = np.array(x)
        if isinstance(y, list):
            y = np.array(y)
        if kind == 'kriging':
            beta = kwargs.pop('beta', 1.5)
            self._func = InterpUtil.getKriging1D(x.asarray(), y.asarray(), beta)
        else:
            self._func = InterpUtil.getInterpFunc(x.asarray(), y.asarray(), kind)

    def __call__(self, x):
        '''
        Evaluate the interpolate values.

        :param x: (*array_like*) Points to evaluate the interpolate at.
        '''
        if isinstance(x, list):
            x = np.array(x)
        if isinstance(x, np.NDArray):
            x = x.asarray()
        r = InterpUtil.evaluate(self._func, x)
        if isinstance(r, float):
            return r
        else:
            return np.NDArray(r)

class interp2d(object):
    '''
    Interpolate over a 2-D grid.

    x, y and z are arrays of values used to approximate some function f: z = f(x, y).
    This class returns a function whose call method uses spline interpolation to find
    the value of new points.

    If x and y represent a regular grid, consider using RectBivariateSpline.

    :param x: (*array_like*) 1-D arrays of x coordinate in strictly ascending order.
    :param y: (*array_like*) 1-D arrays of y coordinate in strictly ascending order.
    :param z: (*array_like*) 2-D array of data with shape (x.size,y.size).
    :param kind: (*boolean*) Specifies the kind of interpolation as a string ('kriging').
        Default is ‘kriging’.
    '''
    def __init__(self, x, y, z, kind='kriging', **kwargs):
        if isinstance(x, list):
            x = np.array(x)
        if isinstance(y, list):
            y = np.array(y)
        if isinstance(z, list):
            z = np.array(z)
        if kind == 'kriging':
            if z.ndim == 2:
                if x.ndim == 1:
                    x, y = np.meshgrid(x, y)
                x = x.reshape(-1)
                y = y.reshape(-1)
                z = z.reshape(-1)
            beta = kwargs.pop('beta', 1.5)
            self._func = InterpUtil.getKriging2D(x.asarray(), y.asarray(), z.asarray(), beta)

    def __call__(self, x, y):
        '''
        Evaluate the interpolate vlaues.

        :param x: (*array_like*) X to evaluate the interpolant at.
        :param y: (*array_like*) Y to evaluate the interpolant at.
        '''
        if isinstance(x, list):
            x = np.array(x)
        if isinstance(x, np.NDArray):
            x = x.asarray()
        if isinstance(y, list):
            y = np.array(y)
        if isinstance(y, np.NDArray):
            y = y.asarray()
        r = InterpUtil.evaluate(self._func, x, y)
        if isinstance(r, float):
            return r
        else:
            return np.NDArray(r)

def griddata(points, values, xi=None, **kwargs):
    '''
    Interpolate scattered data to grid data.

    :param points: (*list*) The list contains x and y coordinate arrays of the scattered data.
    :param values: (*array_like*) The scattered data array.
    :param xi: (*list*) The list contains x and y coordinate arrays of the grid data. Default is ``None``,
        the grid x and y coordinate size were both 500.
    :param method: (*string*) The interpolation method. [idw | cressman | nearest | inside_mean | inside_min
        | inside_max | inside_sum | inside_count | surface | barnes]
    :param fill_value: (*float*) Fill value, Default is ``nan``.
    :param pointnum: (*int*) Only used for 'idw' method. The number of the points to be used for each grid
        value interpolation.
    :param radius: (*float*) Used for 'idw', 'cressman' and 'neareast' methods. The searching raduis. Default
        is ``None`` in 'idw' method, means no raduis was used. Default is ``[10, 7, 4, 2, 1]`` in cressman
        method.
    :param centerpoint: (*boolean*) The grid points located at center or border of grid. Default
        is True (pont at center of grid).
    :param convexhull: (*boolean*) If the convexhull will be used to mask result grid data. Default is ``False``.

    :returns: (*array*) Interpolated grid data (2-D array)
    '''
    method = kwargs.pop('method', 'idw')
    x_s = points[0]
    y_s = points[1]

    if xi is None:
        xn = 500
        yn = 500
        x_g = np.linspace(x_s.min(), x_s.max(), xn)
        y_g = np.linspace(y_s.min(), y_s.max(), yn)
    else:
        x_g = xi[0]
        y_g = xi[1]

    if isinstance(values, np.NDArray):
        values = values.asarray()

    if method == 'kriging':
        beta = kwargs.pop('beta', 1.5)
        r = InterpUtil.gridDataKriging(x_s.asarray(), y_s.asarray(), values, x_g.asarray(), y_g.asarray(), beta)
    else:
        return None

    return np.NDArray(r), x_g, y_g
