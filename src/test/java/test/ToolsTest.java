package test;

import net.neuralnetwork.*;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.stream.Stream;

public class ToolsTest {


    @Test
    public void argMaxMatrixTest() {
        double[] expected = new double[]{
                1, 2, 2, 0
        };
        double[][] input = new double[][]{
                {0, 1, 0},
                {0, 1, 2},
                {0, 1, 2},
                {1, 0, 0},
        };
        int[] actual = Tools.argmax(input);
        for (int i = 0; i < expected.length; i++) {
            Assertions.assertEquals(expected[i], actual[i]);
        }
    }

    @Test
    public void argMaxSingleTest() {
        double expected = 3;
        double[] input = new double[]{3, 6, 5, 9, 3, 2};
        int actual = Tools.argmax(input);
        Assertions.assertEquals(expected, actual);
    }

    @ParameterizedTest
    @MethodSource("accuracyTestInput")
    public void accuracyTest(double[] actual, double[] expected, double result) {
        double accuracy = Tools.accuracy(actual, expected);
        Assertions.assertEquals(result, accuracy);
    }

    private static Stream<Arguments> accuracyTestInput() {
        return Stream.of(
                Arguments.of(
                        new double[]{0, 0},
                        new double[]{0, 1},
                        0.5
                ),
                Arguments.of(
                        new double[]{0, 0},
                        new double[]{0, 0},
                        1.0
                ),
                Arguments.of(
                        new double[]{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                        new double[]{0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
                        0.9
                ),
                Arguments.of(
                        new double[]{1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                        new double[]{0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
                        0.1
                )
        );
    }


    @Test
    public void dvalueProduct() {
        double[][] expected = new double[][]{
                {0.5},
                {20.1},
                {10.9},
                {4.1}};

        double[][] input = new double[][]{
                {1, 2, 3, 2.5},
                {2., 5., -1., 2},
                {-1.5, 2.7, 3.3, -0.8}};

        double[][] dvalues = new double[][]{
                {1, 1, 1},
                {2, 2, 2},
                {3, 3, 3}};

        double[][] dot = Matrix.dot(Matrix.transpose(input), dvalues);
        for (int i = 0; i < dot.length; i++) {
            Assertions.assertEquals(expected[i][0], Math.round(dot[i][0] * 1000) / 1000.0);
        }
    }

    @Test
    public void dvalueSumAxis0() {
        double[][] expected = new double[][]{{6, 6, 6}};
        double[][] dvalues = new double[][]{
                {1, 1, 1},
                {2, 2, 2},
                {3, 3, 3}};

        double[][] sum = Matrix.sumAxis0(dvalues);
        for (int i = 0; i < sum.length; i++) {
            Assertions.assertEquals(expected[0][i], Math.round(sum[0][i] * 1000) / 1000.0);
        }
    }


    @Test
    public void ddvalueSumAxis1() {
        double[][] z = new double[][]{
                {1, 2, -3, -4},
                {2, -7, -1, 3},
                {-1, 2, 5, -1}};

        double[][] dvalues = new double[][]{
                {1, 2, 3, 4},
                {5, 6, 7, 8},
                {9, 10, 11, 12}};

        double[][] drelu = dvalues.clone();
        for (int i = 0; i < drelu.length; i++) {
            for (int j = 0; j < drelu[0].length; j++) {
                drelu[i][j] = z[i][j] <= 0 ? 0 : drelu[i][j];
            }
        }
    }

    @Test
    public void convertToOneHot() {
        double[][] expected = new double[][]{
                {1.0, 0.0, 0.0},
                {0.0, 1.0, 0.0},
                {0.0, 1.0, 0.0}};
        double[] d = {0, 1, 1};
        TestTools.assertMatrix(expected, Tools.convertToOneHot(d, 3));
    }

    @Test
    public void backwardPass() {
        double[][] expteced = new double[][]{
                {-0.1, 0.03333333, 0.06666667},
                {0.03333333, -0.16666667, 0.13333333},
                {0.00666667, -0.03333333, 0.02666667}};
        double[][] softmaxOutput = new double[][]{
                {0.7, 0.1, 0.2},
                {0.1, 0.5, 0.4},
                {0.02, 0.9, 0.08}};

        double[][] target = {
                {1, 0, 0},
                {0, 1, 0},
                {0, 1, 0}};


        ActivationSoftmaxLossCategoricalCrossentropy softmaxLoss = new ActivationSoftmaxLossCategoricalCrossentropy();
        softmaxLoss.backward(softmaxOutput, target);
        double[][] dvalues1 = softmaxLoss.dInputs;

        ActivationSoftmax activationSoftmax = new ActivationSoftmax();
        activationSoftmax.output = softmaxOutput;
        CategoricalCrossEntropyLoss entropyLoss = new CategoricalCrossEntropyLoss();
        entropyLoss.backward(softmaxOutput, target);
        activationSoftmax.backward(entropyLoss.dInputs);
        double[][] dvalues2 = activationSoftmax.dInputs;


        TestTools.assertMatrix(expteced, dvalues1);
        TestTools.assertMatrix(expteced, dvalues2);

    }
}
