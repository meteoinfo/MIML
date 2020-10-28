/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.meteothink.miml.util;

import org.meteoinfo.ndarray.Array;
import org.meteoinfo.ndarray.IndexIterator;
import smile.classification.DataFrameClassifier;
import smile.classification.SoftClassifier;
import smile.clustering.CentroidClustering;
import smile.data.DataFrame;
import smile.data.Tuple;
import smile.data.type.DataType;
import smile.data.type.DataTypes;
import smile.data.type.StructField;
import smile.data.type.StructType;
import smile.math.distance.Distance;

import java.util.ArrayList;
import java.util.List;

/**
 *
 * @author Yaqiang Wang
 */
public class SmileUtil {
    /**
     * Returns the proximity matrix of a dataset for given distance function.
     *
     * @param data Input data
     * @param distance The distance
     * @param half If true, only the lower half of matrix is allocated to save space.
     * @return Proximity matrix
     */
    public static double[][] proximity(double[][] data, Distance distance, boolean half) {
        int n = data.length;
        double[][] proximity;
        if (half) {
            proximity = new double[n][];
            for (int i = 0; i < n; i++) {
                proximity[i] = new double[i + 1];
                for (int j = 0; j < i; j++) {
                    proximity[i][j] = distance.d(data[i], data[j]);
                }
            }
        } else {
            proximity = new double[n][n];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < i; j++) {
                    proximity[i][j] = distance.d(data[i], data[j]);
                    proximity[j][i] = proximity[i][j];
                }
            }
        }

        return proximity;
    }

    /**
     * Convert Array to DataFrame
     * @param x Array
     * @return DataFrame
     */
    public static DataFrame toDataFrame(Array x) {
        //x = x.copyIfView();
        int[] shape = x.getShape();
        int nRow = shape[0];
        int nCol = x.getRank() == 1 ? 1 : shape[1];
        StructField[] fields = new StructField[nCol];
        for (int i = 0; i < fields.length; i++) {
            fields[i] = new StructField("V" + String.valueOf(i + 1),
                    DataType.of(x.getDataType().getClassType()));
        }
        StructType schema = DataTypes.struct(fields);
        List<Tuple> rows = new ArrayList<>();
        IndexIterator iter = x.getIndexIterator();
        for (int i = 0; i < nRow; i++) {
            Object[] row = new Object[nCol];
            for (int j = 0; j < nCol; j++) {
                row[j] = iter.getObjectNext();
            }
            rows.add(Tuple.of(row, schema));
        }
        schema = schema.boxed(rows);
        return DataFrame.of(rows, schema);
    }

    /**
     * Convert Array to DataFrame
     * @param x Array x
     * @param y Array y
     * @return DataFrame
     */
    public static DataFrame toDataFrame(Array x, Array y) {
        //x = x.copyIfView();
        int[] shape = x.getShape();
        int nRow = shape[0];
        int nCol = x.getRank() == 1 ? 1 : shape[1];
        StructField[] fields = new StructField[nCol + 1];
        fields[0] = new StructField("class", DataType.of(y.getDataType().getClassType()));
        for (int i = 0; i < nCol; i++) {
            fields[i + 1] = new StructField("V" + (i + 1),
                    DataType.of(x.getDataType().getClassType()));
        }
        StructType schema = DataTypes.struct(fields);
        List<Tuple> rows = new ArrayList<>();
        IndexIterator iter = x.getIndexIterator();
        IndexIterator yIter = y.getIndexIterator();
        for (int i = 0; i < nRow; i++) {
            Object[] row = new Object[nCol + 1];
            row[0] = yIter.getObjectNext();
            for (int j = 0; j < nCol; j++) {
                row[j + 1] = iter.getObjectNext();
            }
            rows.add(Tuple.of(row, schema));
        }
        schema = schema.boxed(rows);
        return DataFrame.of(rows, schema);
    }

    /**
     * Get predict probability.
     * @param classifier The SoftClassifier.
     * @param x Input data.
     * @param k The number of classes
     * @return Predict probability.
     */
    public static double[][] predictProbability(SoftClassifier classifier, double[][] x, int k) {
        int n = x.length;
        double[] posteriori;
        double[][] probability = new double[n][k];
        boolean isTuple = classifier instanceof DataFrameClassifier;
        if (isTuple) {            
            Tuple tuple;
            for (int i = 0; i < n; i++) {
                tuple = Tuple.of(x[i], null);
                posteriori = new double[k];
                classifier.predict(tuple, posteriori);
                probability[i] = posteriori;
            }
        } else {
            for (int i = 0; i < n; i++) {
                posteriori = new double[k];
                classifier.predict(x[i], posteriori);
                probability[i] = posteriori;
            }
        }

        return probability;
    }

    /**
     * Cluster predict new data array
     * @param clustering The cluster model
     * @param x New data array
     * @return Predicted index array
     */
    public static Array clusterPredict(CentroidClustering clustering, double[][] x) {
        Array r = Array.factory(org.meteoinfo.ndarray.DataType.INT, new int[]{x.length});
        for (int i = 0; i < x.length; i++) {
            r.setInt(i, clustering.predict(x[i]));
        }

        return r;
    }
}
