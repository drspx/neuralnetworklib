package test;

import org.junit.jupiter.api.Assertions;

public class TestTools {

    public static void assertMatrix(double[][] expected, double[][] actual) {
        double roundWith = 1000000.0;
        for (int i = 0; i < actual.length; i++) {
            for (int j = 0; j < actual[0].length; j++) {
                Assertions.assertEquals(Math.round(expected[i][j] * roundWith) / roundWith, Math.round(actual[i][j] * roundWith) / roundWith);
            }
        }
    }

    public static void assertMatrix(double[] expected, double[] actual) {
        double roundWith = 100000000.0;
        for (int i = 0; i < actual.length; i++) {
            Assertions.assertEquals(Math.round(expected[i] * roundWith) / roundWith, Math.round(actual[i] * roundWith) / roundWith);
        }
    }

}
