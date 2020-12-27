package test;

import net.neuralnetwork.ActivationSoftmax;
import net.neuralnetwork.Matrix;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.Arrays;
import java.util.stream.Stream;


public class ActivationSoftmaxTest {

    @Test
    public void normalValuesTest() {
        ActivationSoftmax softmax = new ActivationSoftmax();

        double[][] inputs = {
                {4.8, 1.21, 2.385},
                {8.9, -1.81, 0.2},
                {1.41, 1.051, 0.026}};

        Arrays.stream(Matrix.suml(softmax.getNormValues(inputs)))
                .forEach(a -> Assertions.assertEquals(Math.round(a * 100) / 100, 1));
    }

    @ParameterizedTest
    @MethodSource
    public void forwardTest(double[][] inputs, double[][] expected) {

        double[][] forward = new ActivationSoftmax().forward(inputs);

        for (int i = 0; i < forward.length; i++) {
            for (int j = 0; j < forward[0].length; j++) {
                Assertions.assertEquals(Math.round(expected[i][j] * 1000) / 1000, Math.round(forward[i][j] * 1000) / 1000);
            }
        }
    }

    private static Stream<Arguments> forwardTest() {
        return Stream.of(
                Arguments.of(
                        new double[][]{{1, 2, 3}},
                        new double[][]{{0.09003057, 0.24472847, 0.66524096}}
                ),
                Arguments.of(
                        new double[][]{{-2, -1, 0}},
                        new double[][]{{0.09003057, 0.24472847, 0.66524096}}
                ),
                Arguments.of(
                        new double[][]{{0.5, 1, 1.5}},
                        new double[][]{{0.18632372, 0.30719589, 0.50648039}}
                )
        );
    }

    @Test
    public void backwardTest() {


    }
}
