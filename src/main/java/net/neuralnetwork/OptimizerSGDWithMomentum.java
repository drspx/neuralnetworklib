package net.neuralnetwork;

public class OptimizerSGDWithMomentum {

    public static double learningRate = 1;
    public static double currentLearningRate = learningRate;
    public static double decay = 0.001;
    public static double iterations = 0;


    public static void preUpdateParameters() {
        if (decay != 0) {
            currentLearningRate = learningRate * (1 / (1 + decay * iterations));
        }
    }

    public static void postUpdateParameters() {
        ++iterations;
    }


    public static double momentum = 1;
    private static double[] weightMomentum;
    private static double[][] biasMomentum;


    public static void updateParamsWithMomentum(DenseLayer layer) {
//        preUpdateParameters();

        if (layer.weightMomentums == null) {
            layer.weightMomentums = Matrix.zeroesLike(layer.weights);
            layer.biasMomentums = Matrix.zeroesLike(layer.bias);
        }


        double[][] scalar = Matrix.scalar(momentum, layer.weightMomentums);
        double[][] scalar1 = Matrix.scalar(currentLearningRate, layer.dWeights);
        //momentum * layer.weightMomentum - currentLearningRate * layer.dWeights
        layer.weightMomentums = Matrix.applyFunction((a, b) -> (a - b), scalar, scalar1);


        layer.weights = Matrix.applyFunction((n, m) -> -currentLearningRate * n + m, layer.dWeights, layer.weights);
        layer.bias = (Matrix.applyFunction((n, m) -> -currentLearningRate * n + m, layer.dBiases, layer.bias));

    }


}
