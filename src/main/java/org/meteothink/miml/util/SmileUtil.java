/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.meteothink.miml.util;

import org.meteoinfo.ndarray.IndexIterator;
import smile.data.DataFrame;
import smile.data.Tuple;
import smile.data.type.DataType;
import smile.data.type.DataTypes;
import smile.data.type.StructField;
import smile.data.type.StructType;
import smile.math.distance.Distance;
import org.meteoinfo.ndarray.Array;

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
            fields[i] = new StructField("x" + String.valueOf(i + 1),
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
        for (int i = 0; i < nCol; i++) {
            fields[i] = new StructField("x" + String.valueOf(i + 1),
                    DataType.of(x.getDataType().getClassType()));
        }
        fields[nCol] = new StructField("class", DataType.of(y.getDataType().getClassType()));
        StructType schema = DataTypes.struct(fields);
        List<Tuple> rows = new ArrayList<>();
        IndexIterator iter = x.getIndexIterator();
        IndexIterator yIter = y.getIndexIterator();
        for (int i = 0; i < nRow; i++) {
            Object[] row = new Object[nCol + 1];
            for (int j = 0; j < nCol; j++) {
                row[j] = iter.getObjectNext();
            }
            row[nCol] = yIter.getObjectNext();
            rows.add(Tuple.of(row, schema));
        }
        schema = schema.boxed(rows);
        return DataFrame.of(rows, schema);
    }
}
