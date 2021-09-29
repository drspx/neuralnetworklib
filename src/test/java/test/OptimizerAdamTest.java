package test;

import net.neuralnetwork.*;
import org.junit.jupiter.api.Test;

public class OptimizerAdamTest {

    @Test
    public void adamTest() {
        DenseLayer dense1 = new DenseLayer(2, 3);
        ActivationReLU activation1 = new ActivationReLU();
        DenseLayer dense2 = new DenseLayer(3, 3);
        ActivationSoftmaxLossCategoricalCrossentropy lossActivation = new ActivationSoftmaxLossCategoricalCrossentropy();
        OptimizerAdam optimizer = new OptimizerAdam();

        dense1.weights = getWeights1();
        dense2.weights = getWeights2();

        for (int i = 0; i < 1; i++) {
            dense1.forward(Tools.spiralDataX());
            activation1.forward(dense1.output);
            dense2.forward(activation1.output);
            double loss = lossActivation.forward(dense2.output, Tools.spiralDatay());
            double accuracy = Tools.accuracy(lossActivation.output, Tools.spiralDatay());

            lossActivation.backward(lossActivation.output, Tools.spiralDatay());
            dense2.backward(lossActivation.dInputs);
            activation1.backward(dense2.dInputs);
            dense1.backward(activation1.dInputs);

            optimizer.preUpdate();
            optimizer.update(dense1);
            optimizer.update(dense2);
            optimizer.postUpdate();

        }
        TestTools.assertMatrix(getWeightd1After(), dense1.weights);
        TestTools.assertMatrix(getBiasd1After(), dense1.bias);

        TestTools.assertMatrix(getWeightd2After(), dense2.weights);
        TestTools.assertMatrix(getBiasd2After(), dense2.bias, 10000);
    }

    private static double[][] getWeights1() {
        return new double[][]{{-0.01306527, 0.01658131, -0.00118164},
                {-0.00680178, 0.00666383, -0.0046072}};
    }

    private static double[][] getWeights2() {
        return new double[][]{{-0.01334258, -0.01346717, 0.00693773},
                {-0.00159573, -0.00133702, 0.01077744},
                {-0.01126826, -0.00730678, -0.0038488}};
    }

    private static double[][] getWeightd1After() {
        return new double[][]{{-0.06303358, -0.03335498, -0.05107621},
                {-0.05677427, -0.04288756, 0.04524218}};
    }

    private static double[][] getBiasd1After() {
        return new double[][]{{0.04998614, -0.0499483, 0.04995182}};
    }

    private static double[][] getWeightd2After() {
        return new double[][]{{-0.06325086, -0.06342067, 0.05690686},
                {0.04828199, 0.04859333, -0.03917815},
                {0.0386376, -0.05724858, 0.04599925}};
    }

    private static double[][] getBiasd2After() {
        return new double[][]{{0.04953828, 0.04947705, -0.04975159}};
    }

}
