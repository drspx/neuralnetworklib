package com.net;

public class OptimizerSGD {
    private double learningRate = 1;

    public OptimizerSGD() {
    }

    public OptimizerSGD(double learningRate) {
        this.learningRate = learningRate;
    }

    public static void updateParams(DenseLayer layer) {
        double learningRate = 1;
        layer.weights = Matrix.applyFunction((n, m) -> -learningRate * n+m, layer.dWeights, layer.weights);
        layer.bias = (Matrix.applyFunction((n, m) -> -learningRate * n+m, layer.dBiases, layer.bias));
    }
}
