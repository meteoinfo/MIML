package org.meteothink.miml.nd4j;

import org.meteoinfo.ndarray.math.ArrayUtil;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.meteoinfo.ndarray.Array;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.NDArrayUtil;

public class Nd4jUtil {
    /**
     * Convert MeteoInfo Array to ND4J INDArray
     * @param a MeteoInfo Array object
     * @return ND4J INDArray
     */
    public static INDArray toNDArray(Array a) {
        Object data;
        if (a.getIndexPrivate().isFastIterator())
            data = a.getStorage();
        else
            data = a.copyTo1DJavaArray();
        INDArray r = null;
        if (data instanceof int[]) {
            long[] shape = new long[a.getRank()];
            for (int i = 0; i < a.getRank(); i++)
                shape[i] = a.getShape()[i];
            r = Nd4j.create((int[])data, shape, DataType.INT);
        } else if (data instanceof float[])
            r = Nd4j.create((float[])data, a.getShape(), 'c');
        else if (data instanceof double[])
            r = Nd4j.create((double[])data, a.getShape(), 'c');
        
        return r;
    }

    /**
     * Convert ND4J INDArray to MeteoInfo Array
     * @param a ND4J INDArray object
     * @return MeteoInfo Array
     */
    public static Array toArray(INDArray a) {
        DataType dt = a.dataType();
        long[] lshape = a.shape();
        int n = lshape.length;
        int[] shape = new int[n];
        for (int i = 0; i < n; i++)
            shape[i] = (int)lshape[i];
        Array r = null;
        switch (dt) {
            case DOUBLE:
                if (n == 1) {
                    double[] data = a.toDoubleVector();
                    r = Array.factory(org.meteoinfo.ndarray.DataType.DOUBLE, shape, data);
                } else if (n == 2) {
                    double[][] data = a.toDoubleMatrix();
                    r = ArrayUtil.array(data, org.meteoinfo.ndarray.DataType.DOUBLE);
                }
                break;
            case FLOAT:
                if (n == 1) {
                    float[] data = a.toFloatVector();
                    return Array.factory(org.meteoinfo.ndarray.DataType.FLOAT, shape, data);
                } else if (n == 2) {
                    float[][] data = a.toFloatMatrix();
                    r = ArrayUtil.array(data, org.meteoinfo.ndarray.DataType.FLOAT);
                }
                break;
            case INT:
                if (n == 1) {
                    int[] data = a.toIntVector();
                    return Array.factory(org.meteoinfo.ndarray.DataType.INT, shape, data);
                } else if (n == 2) {
                    int[][] data = a.toIntMatrix();
                    r = ArrayUtil.array(data, org.meteoinfo.ndarray.DataType.INT);
                }
                break;
            case LONG:
                if (n == 1) {
                    long[] data = a.toLongVector();
                    return Array.factory(org.meteoinfo.ndarray.DataType.LONG, shape, data);
                } else if (n == 2) {
                    long[][] data = a.toLongMatrix();
                    r = ArrayUtil.array(data, org.meteoinfo.ndarray.DataType.LONG);
                }  
                break;
        }
        
        return r;
    }

    /**
     * Creates an outcome matrix from the specified inputs
     *
     * @param index       the index of the label
     * @param numOutcomes the number of possible outcomes
     * @return a binary label matrix used for supervised learning
     */
    public static INDArray toNDMatrix(Array index, int numOutcomes) {
        INDArray ret = Nd4j.create(index.getSize(), numOutcomes);
        for (int i = 0; i < ret.rows(); i++) {
            int[] nums = new int[(int) numOutcomes];
            nums[index.getInt(i)] = 1;
            ret.putRow(i, NDArrayUtil.toNDArray(nums));
        }

        return ret;
    }

    /**
     * Creates an outcome matrix from the specified inputs
     *
     * @param index       the index of the label
     * @param numOutcomes the number of possible outcomes
     * @return a binary label matrix used for supervised learning
     */
    public static Array toMatrix(Array index, int numOutcomes) {
        int n = index.getShape()[0];
        Array ret = Array.factory(index.getDataType(), new int[]{n, numOutcomes});
        for (int i = 0; i < n; i++) {
            ret.setInt(i * numOutcomes + index.getInt(i), 1);
        }

        return ret;
    }

}
