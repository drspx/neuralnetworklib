package com.test;

import com.net.DenseLayer;
import com.net.Main;
import org.junit.jupiter.api.Test;

public class DenseLayerTest {

    @Test
    public void inputTest() {

        double[] input1 = {1, 2, 3, 2.5};
        double[] input2 = {2., 5., -1., 2};
        double[] input3 = {-1.5, 2.7, 3.3, -0.8};
        double[][] inputs = {input1, input2, input3};

        double[] weights11 = {0.2, 0.8, -0.5, 1.0};
        double[] weights12 = {0.5, -0.91, 0.26, -0.5};
        double[] weights13 = {-0.26, -0.27, 0.17, 0.87};
        double[][] weights1 = {weights11, weights12, weights13};
        double[] biases1 = {2.0, 3.0, 0.5};

        double[] weights21 = {0.1, -0.14, 0.5};
        double[] weights22 = {-0.5, 0.12, -0.33};
        double[] weights23 = {-0.44, 0.73, -0.13};
        double[][] weights2 = {weights21, weights22, weights23};
        double[] biases2 = {-1, 2, -0.5};

        DenseLayer layer1 = new DenseLayer(2, 3);
        layer1.forward(Main.spiralDataX());
        DenseLayer layer2 = new DenseLayer(3, 2);
        layer2.forward(layer1.output);
//        Matrix.printMatrix(layer2.output);

    }
}
