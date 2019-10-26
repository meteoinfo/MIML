/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.meteothink.miml.util;

import smile.math.distance.Distance;

/**
 *
 * @author Yaqiang Wang
 */
public class SmileUtil {
    /**
     * eturns the proximity matrix of a dataset for given distance function.
     *
     * @param data Input data
     * @param distance The distance
     * @param half If true, only the lower half of matrix is allocated to save space.
     * @return Proximity maxtrix
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
}
