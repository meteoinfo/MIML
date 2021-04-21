package org.meteothink.miml.util;

import org.meteoinfo.ndarray.Array;
import org.meteoinfo.ndarray.DataType;
import org.meteoinfo.ndarray.IndexIterator;
import org.meteoinfo.ndarray.math.ArrayUtil;
import smile.interpolation.KrigingInterpolation1D;
import smile.interpolation.KrigingInterpolation2D;

public class InterpUtil {
    /**
     * Get Kriging interpolation 1D class
     * @param x X array
     * @param y Y array
     * @param beta Beta value
     * @return Kriging interpolation 1D class
     */
    public static KrigingInterpolation1D getKriging1D(Array x, Array y, double beta) {
        double[] xd = (double[]) ArrayUtil.copyToNDJavaArray_Double(x);
        double[] yd = (double[]) ArrayUtil.copyToNDJavaArray_Double(y);
        KrigingInterpolation1D krigingInterpolation1D = new KrigingInterpolation1D(xd, yd, beta);

        return krigingInterpolation1D;
    }

    /**
     * Make interpolation function for grid data
     *
     * @param x X data
     * @param y Y data
     * @param z Z data
     * @param beta Beta value
     * @return Interpolation function
     */
    public static KrigingInterpolation2D getKriging2D(Array x, Array y, Array z, double beta) {
        double[] xd = (double[]) ArrayUtil.copyToNDJavaArray_Double(x);
        double[] yd = (double[]) ArrayUtil.copyToNDJavaArray_Double(y);
        double[] zd = (double[]) ArrayUtil.copyToNDJavaArray_Double(z);
        KrigingInterpolation2D krigingInterpolation2D = new KrigingInterpolation2D(xd, yd, zd, beta);

        return krigingInterpolation2D;
    }

    /**
     * Compute the value of the function
     *
     * @param func The function
     * @param x Input data
     * @return Function value
     */
    public static double evaluate(KrigingInterpolation1D func, Number x) {
        return func.interpolate(x.doubleValue());
    }

    /**
     * Compute the value of the function
     *
     * @param func The function
     * @param x Input data
     * @return Function value
     */
    public static Array evaluate(KrigingInterpolation1D func, Array x) {
        Array r = Array.factory(DataType.DOUBLE, x.getShape());
        IndexIterator xIter = x.getIndexIterator();
        for (int i = 0; i < r.getSize(); i++) {
            r.setDouble(i, func.interpolate(xIter.getDoubleNext()));
        }

        return r;
    }

    /**
     * Compute the value of the function
     *
     * @param func The function
     * @param x Input x data
     * @param y Input y data
     * @return Function value
     */
    public static Array evaluate(KrigingInterpolation2D func, Array x, Array y) {
        Array r = Array.factory(DataType.DOUBLE, x.getShape());
        IndexIterator xIter = x.getIndexIterator();
        IndexIterator yIter = y.getIndexIterator();
        for (int i = 0; i < r.getSize(); i++) {
            r.setDouble(i, func.interpolate(xIter.getDoubleNext(), yIter.getDoubleNext()));
        }

        return r;
    }

    /**
     * Compute the value of the function
     *
     * @param func The function
     * @param x Input x data
     * @param y Input y data
     * @return Function value
     */
    public static double evaluate(KrigingInterpolation2D func, Number x, Number y) {
        return func.interpolate(x.doubleValue(), y.doubleValue());
    }

    /**
     * Interpolation with Kriging2D method
     *
     * @param x_s scatter X array
     * @param y_s scatter Y array
     * @param a scatter value array
     * @param X grid X array
     * @param Y grid Y array
     * @param beta Beta
     * @return interpolated grid data
     */
    public static Array gridDataKriging(Array x_s, Array y_s, Array a,
                                        Array X, Array Y, double beta) {
        X = X.copyIfView();
        Y = Y.copyIfView();
        double[] xd = (double[]) ArrayUtil.copyToNDJavaArray_Double(x_s);
        double[] yd = (double[]) ArrayUtil.copyToNDJavaArray_Double(y_s);
        double[] ad = (double[]) ArrayUtil.copyToNDJavaArray_Double(a);

        int rowNum, colNum, pNum;
        colNum = (int)X.getSize();
        rowNum = (int)Y.getSize();
        pNum = (int)x_s.getSize();
        Array r = Array.factory(DataType.DOUBLE, new int[]{rowNum, colNum});
        int i, j;
        double w, gx, gy, v;
        boolean match;

        //Construct Kriging2D interpolation
        KrigingInterpolation2D ki2d = new KrigingInterpolation2D(xd, yd, ad, beta);

        //---- Do interpolation
        for (i = 0; i < rowNum; i++) {
            gy = Y.getDouble(i);
            for (j = 0; j < colNum; j++) {
                gx = X.getDouble(j);
                r.setDouble(i * colNum + j, ki2d.interpolate(gx, gy));
            }
        }

        return r;
    }
}
