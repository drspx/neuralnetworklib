package test;

import net.neuralnetwork.CategoricalCrossEntropyLoss;
import net.neuralnetwork.Loss;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static net.neuralnetwork.Matrix.applyFunction;

public class CategoricalCrossEntropyLossTest {

    @Test
    public void oneHotEncoding() {
        CategoricalCrossEntropyLoss l = new CategoricalCrossEntropyLoss();
        double[][] output = {
                {0.7, 0.1, 0.2},
                {0.1, 0.5, 0.4},
                {0.02, 0.9, 0.08}};
        double[][] targets = new double[][]{
                {1, 0, 0},
                {0, 1, 0},
                {0, 1, 0}};
        double[] expected = new double[]{0.35667494, 0.69314718, 0.10536052};

        double average = Loss.calculate(l, output, targets);
        double[] actual = l.output;

        for (int i = 0; i < targets.length; i++) {
            Assertions.assertEquals(Math.round(expected[i] * 1000) / 1000, Math.round(actual[i] * 1000) / 1000);
        }
        Assertions.assertEquals(0.385060880052168, average);
    }


    @Test
    public void oneDimensional() {
        CategoricalCrossEntropyLoss l = new CategoricalCrossEntropyLoss();
        double[][] output = {
                {0.7, 0.1, 0.2},
                {0.1, 0.5, 0.4},
                {0.02, 0.9, 0.08}};
        double[] target = {0, 1, 1};
        double[] expected = new double[]{0.35667494, 0.69314718, 0.10536052};

        double average = Loss.calculate(l, output, target);
        double[] actual = l.output;

        for (int i = 0; i < target.length; i++) {
            Assertions.assertEquals(Math.round(expected[i] * 1000) / 1000, Math.round(actual[i] * 1000) / 1000);
        }
        Assertions.assertEquals(0.385060880052168, average);
    }


    @Test
    public void logOfNot() {
        CategoricalCrossEntropyLoss loss = new CategoricalCrossEntropyLoss();
        double smallNumber = 1e-7;
        Assertions.assertEquals(-Math.log(smallNumber), -loss.findApproximateLog(0));
        Assertions.assertEquals(-Math.log(smallNumber), -loss.findApproximateLog(Double.MIN_VALUE));

        Assertions.assertEquals(-Math.log(1 - smallNumber), -loss.findApproximateLog(1));
        Assertions.assertEquals(-Math.log(1 - smallNumber), -loss.findApproximateLog((double) Integer.MAX_VALUE));
    }

    @Test
    public void sparseEncoding() {

    }

    @Test
    public void backwardTest() {
        double[][] dValue = new double[][]{
                {1, 1, 1},
                {2, 2, 2},
                {3, 3, 3}};
        double[][] yTrue = new double[][]{
                {1, 2, 3},
                {2., 5., -1.},
                {-1.5, 2.7, 3.3}};

        double samples = Arrays.stream(dValue).count();
        double[][] doubles = applyFunction((n1, n2) -> (-n1 / n2) / samples, yTrue, dValue);
        new CategoricalCrossEntropyLoss().backward(dValue, yTrue);

        for (int i = 0; i < yTrue.length; i++) {
            for (int j = 0; j < yTrue[0].length; j++) {
                Assertions.assertEquals((-yTrue[i][j] / dValue[i][j]) / samples, doubles[i][j]);
            }
        }

    }

}
