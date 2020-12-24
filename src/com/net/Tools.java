package com.net;

import java.util.Arrays;

import static com.net.Matrix.copyOf;

public class Tools {

    public static int argmax(double[] list) {
        int indexOfMax = 0;
        for (int i = 0; i < list.length; i++) {
            indexOfMax = list[i] > list[indexOfMax] ? i : indexOfMax;
        }
        return indexOfMax;
    }

    public static int[] argmax(double[][] list) {
        int[] ints = Arrays.stream(list).map(Tools::argmax).mapToInt(Integer::intValue).toArray();
        return ints;
    }

    public static double accuracy(int[] actual, double[] expected) {
        return accuracyHelper(Arrays.stream(actual).asDoubleStream().toArray(), expected);
    }

    public static double accuracy(double[] actual, double[] expected) {
        return accuracyHelper(actual, expected);
    }

    public static double accuracy(double[][] actual, double[] expected) {
        return accuracyHelper(Arrays.stream(actual).mapToDouble(Tools::argmax).toArray(), expected);
    }

    private static double accuracyHelper(double[] actual, double[] expected) {
        double[] correctPredictions = copyOf(expected);
        for (int i = 0; i < actual.length; i++) {
            correctPredictions[i] = actual[i] == expected[i] ? 1 : 0;
        }
        return Arrays.stream(correctPredictions).average().orElse(-1.0);

    }

    public static double[][] convertToOneHot(double[] ys, int variables) {
        int max = Arrays.stream(ys).mapToInt(n -> (int) n).max().orElse(-1);
        double[][] zeros = new double[ys.length][variables];
        for (int i = 0; i < ys.length; i++) {
            zeros[i][(int) ys[i]] = 1.0;
        }
        return zeros;
    }


}
