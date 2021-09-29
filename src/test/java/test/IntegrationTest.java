package test;

import net.neuralnetwork.*;
import org.junit.jupiter.api.Test;

public class IntegrationTest {

    @Test
    public void backpropagationCompleteNetwork() {
        double[][] dense1Weights = new double[][]{
                {-0.01306527, 0.01658131, -0.00118164},
                {-0.00680178, 0.00666383, -0.0046072}};
        double[][] dense1Biases = new double[][]{{0., 0., 0.}};

        double[][] expectedDWeights1AfterSpiral = new double[][]{
                {1.5766357e-04, 7.8368583e-05, 4.7324400e-05},
                {1.8161038e-04, 1.1045573e-05, -3.3096312e-05}};
        double[][] expectedDBiases1AfterSpiral = new double[][]{
                {-3.60553473e-04, 9.66117223e-05, -1.03671395e-04}};

        double[][] dense2Weights = new double[][]{
                {-0.01334258, -0.01346717, 0.00693773},
                {-0.00159573, -0.00133702, 0.01077744},
                {-0.01126826, -0.00730678, -0.0038488}};
        double[][] dense2Biases = new double[][]{{0., 0., 0.}};

        double[][] expectedDWeights2AfterSpiral = new double[][]{
                {5.44109462e-05, 1.07411419e-04, -1.61822361e-04},
                {-4.07913431e-05, -7.16780924e-05, 1.12469446e-04},
                {-5.30112993e-05, 8.58172934e-05, -3.28059905e-05}};
        double[][] expectedDBiases2AfterSpiral = new double[][]{
                {-1.0729185e-05, -9.4610732e-06, 2.0027859e-05}};


        DenseLayer dense1 = new DenseLayer(2, 3);
        ActivationReLU activation1 = new ActivationReLU();
        DenseLayer dense2 = new DenseLayer(3, 3);
        ActivationSoftmaxLossCategoricalCrossentropy lossActivation = new ActivationSoftmaxLossCategoricalCrossentropy();

        dense1.weights = dense1Weights;
        dense1.bias = dense1Biases;
        dense2.weights = dense2Weights;
        dense2.bias = dense2Biases;

        //forward
        dense1.forward(Tools.spiralDataX());
        activation1.forward(dense1.output);
        dense2.forward(activation1.output);
        lossActivation.forward(dense2.output, Tools.convertToOneHot(Tools.spiralDatay(), 3));

        //backpropagation
        lossActivation.backward(lossActivation.output, Tools.convertToOneHot(Tools.spiralDatay(), 3));
        dense2.backward(lossActivation.dInputs);
        activation1.backward(dense2.dInputs);
        dense1.backward(activation1.dInputs);

        TestTools.assertMatrix(expectedDWeights1AfterSpiral, dense1.dWeights);
        TestTools.assertMatrix(expectedDBiases1AfterSpiral, dense1.dBiases);
        TestTools.assertMatrix(expectedDWeights2AfterSpiral, dense2.dWeights);
        TestTools.assertMatrix(expectedDBiases2AfterSpiral, dense2.dBiases);
    }

    @Test
    public void backpropagationOptimizer() {
        double[][] dense1Weights = new double[][]{
                {-0.01306527, 0.01658131, -0.00118164},
                {-0.00680178, 0.00666383, -0.0046072}};
        double[][] dense1Biases = new double[][]{{0., 0., 0.}};
        double[][] expectedWeights1AfterSpiralOptimized = new double[][]{
                {-0.01322293, 0.01650294, -0.00122896},
                {-0.00698339, 0.00665279, -0.0045741}};
        double[][] expectedBiases1AfterSpiralOptimized = new double[][]{
                {3.60553473e-04, -9.66117223e-05, 1.03671395e-04}};

        double[][] dense2Weights = new double[][]{
                {-0.01334258, -0.01346717, 0.00693773},
                {-0.00159573, -0.00133702, 0.01077744},
                {-0.01126826, -0.00730678, -0.0038488}};
        double[][] dense2Biases = new double[][]{{0., 0., 0.}};
        double[][] expectedWeights2AfterSpiralOptimized = new double[][]{
                {-0.013397, -0.01357459, 0.00709955},
                {-0.00155494, -0.00126534, 0.01066497},
                {-0.01121525, -0.00739259, -0.00381599}};
        double[][] expectedBiases2AfterSpiralOptimized = new double[][]{
                {1.0729185e-05, 9.4610732e-06, -2.0027859e-05}};


        DenseLayer dense1 = new DenseLayer(2, 3);
        ActivationReLU activation1 = new ActivationReLU();
        DenseLayer dense2 = new DenseLayer(3, 3);
        ActivationSoftmaxLossCategoricalCrossentropy lossActivation = new ActivationSoftmaxLossCategoricalCrossentropy();

        dense1.weights = dense1Weights;
        dense1.bias = dense1Biases;
        dense2.weights = dense2Weights;
        dense2.bias = dense2Biases;

        TestTools.assertMatrix(dense1Weights, dense1.weights);
        TestTools.assertMatrix(dense1Biases, dense1.bias);
        TestTools.assertMatrix(dense2Weights, dense2.weights);
        TestTools.assertMatrix(dense2Biases, dense2.bias);

        //forward
        dense1.forward(Tools.spiralDataX());
        activation1.forward(dense1.output);
        dense2.forward(activation1.output);
        double loss = lossActivation.forward(dense2.output, Tools.convertToOneHot(Tools.spiralDatay(), 3));

        //backpropagation
        lossActivation.backward(lossActivation.output, Tools.convertToOneHot(Tools.spiralDatay(), 3));
        dense2.backward(lossActivation.dInputs);
        activation1.backward(dense2.dInputs);
        dense1.backward(activation1.dInputs);

        OptimizerSGD.updateParams(dense1);
        OptimizerSGD.updateParams(dense2);

        TestTools.assertMatrix(expectedWeights1AfterSpiralOptimized, dense1.weights);
        TestTools.assertMatrix(expectedBiases1AfterSpiralOptimized, dense1.bias);
        TestTools.assertMatrix(expectedWeights2AfterSpiralOptimized, dense2.weights);
        TestTools.assertMatrix(expectedBiases2AfterSpiralOptimized, dense2.bias);
    }

}
