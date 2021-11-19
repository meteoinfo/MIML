package org.meteothink.miml.dl4j;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.MiniBatchFileDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.IOException;

public class DL4jUtil {

    /**
     * Fit MultiLayerNetwork with epochs
     * @param model The network model
     * @param data The input data
     * @param labels The input labels
     * @param epochs The epochs
     * @param printStride The print stride
     */
    public static void fit(MultiLayerNetwork model, INDArray data, INDArray labels, int epochs,
                           int printStride) {
        DataSet dataSet = new DataSet(data, labels);
        int ppi = 0;
        for(int i = 0; i < epochs; i++) {
            if (i == ppi || i == epochs - 1){
                System.out.println(String.format("Epoch %d", i + 1));
                ppi += printStride;
            }
            model.fit(dataSet);
        }
    }

    /**
     * Fit MultiLayerNetwork with epochs
     * @param model The network model
     * @param data The input data
     * @param labels The input labels
     * @param epochs The epochs
     * @param batchSize The batch size
     * @param printStride The print stride
     */
    public static void fit(MultiLayerNetwork model, INDArray data, INDArray labels, int epochs,
                           int batchSize, int printStride) throws IOException {
        DataSet dataSet = new DataSet(data, labels);
        DataSetIterator iter = new MiniBatchFileDataSetIterator(dataSet, batchSize);
        int ppi = 0;
        for(int i = 0; i < epochs; i++) {
            if (i == ppi || i == epochs - 1){
                System.out.println(String.format("Epoch %d", i + 1));
                ppi += printStride;
            }
            model.fit(iter);
        }
    }
}
