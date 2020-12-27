package net.neuralnetwork;

import java.util.Arrays;

public class Loss {

    public static double calculate(CategoricalCrossEntropyLoss loss, double[][] input, double[] y) {
        double[] losses = loss.forward(input, y);
        return Arrays.stream(losses).average().orElse(-1);
    }

    public static double calculate(CategoricalCrossEntropyLoss loss, double[][] input, double[][] y) {
        double[] losses = loss.forward(input, y);
        return Arrays.stream(losses).average().orElse(-1);
    }

}
