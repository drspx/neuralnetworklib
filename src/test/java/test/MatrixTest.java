package test;


import net.neuralnetwork.Matrix;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import static test.TestTools.assertMatrix;


public class MatrixTest {

    @Test
    public void testDotDifferentSizes() {
        double[][] m1 = new double[][]{
                {1, 2, 3, 2.5},
                {2., 5., -1., 2},
                {-1.5, 2.7, 3.3, -0.8}};

        double[][] m2 = new double[][]{
                {0.2, 0.5, -0.26},
                {0.8, -0.91, -0.27},
                {-0.5, 0.26, 0.17},
                {1.0, -0.5, 0.87}};

        double[][] expected = new double[][]{
                {2.8000, -1.7900, 1.8850},
                {6.9000, -4.8100, -0.3000},
                {-0.5900, -1.9490, -0.4740}};
        assertMatrix(expected, Matrix.dot(m1, m2));
    }


    @Test
    public void testDotEqualSize() {
        double[][] m1 = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 8}};
        double[][] expected = new double[][]{{30, 36, 39}, {66, 81, 90}, {95, 118, 133}};

        assertMatrix(expected, Matrix.dot(m1, m1));
    }

    @Test
    public void testAddForEachTwoMatrices() {
        double[][] m1 = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 8}};
        double[][] m2 = new double[][]{{1, 1, 1}, {1, 1, 1}, {1, 1, 1}};
        double[][] expected = new double[][]{{2, 3, 4}, {5, 6, 7}, {8, 9, 9}};

        assertMatrix(expected, Matrix.addForEach(m1, m2));
    }

    @Test
    public void testTranspose() {
        double[][] m1 = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 8}};
        double[][] expected = new double[][]{{1, 4, 7}, {2, 5, 8}, {3, 6, 8}};
        assertMatrix(expected, Matrix.transpose(m1));
    }

    @Test
    public void testnpZeros() {
        double[][] expected = new double[][]{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
        double[][] actual = Matrix.zeros(3, 3);
        assertMatrix(expected, actual);
    }

    @Test
    public void testnpRandom() {
        double[][] expected = new double[][]{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
        double[][] actual = Matrix.random(3, 3);

        for (int i = 0; i < expected.length; i++) {
            for (int j = 0; j < expected[0].length; j++) {
                Assertions.assertTrue(expected[i][j] < actual[i][j] + 1);
            }
        }
    }

    @Test
    public void dotTwoDimensionWithOneDimension(){
        double[][] m = new double[][]{
                {2.8000, -1.7900, 1.8850},
                {6.9000, -4.8100, -0.3000},
                {-0.5900, -1.9490, -0.4740}};
        double[] v = new double[]{2.0, 3.0, 0.5};
        double[] expected = new double[]{1.1725, -0.78  , -7.264};
        double[] dot = Matrix.dot(m, v);
        assertMatrix(expected, Matrix.dot(m,v));

    }

    @Test
    public void testPlus() {
        double[][] before = new double[][]{
                {2.8000, -1.7900, 1.8850},
                {6.9000, -4.8100, -0.3000},
                {-0.5900, -1.9490, -0.4740}};
        double[] addThis = new double[]{2.0, 3.0, 0.5};
        double[][] expected = new double[][]{
                {4.8, 1.21, 2.385,},
                {8.9000, -1.8100, 0.2000},
                {1.4100, 1.0510, 0.0260}};
        assertMatrix(expected, Matrix.addForEach(before, addThis));
    }

    @Test
    public void reshape(){
        double[] d = new double[]{1, 2, 3, 4, 5, 6};
        double[][] reshape = Matrix.reshape(d, 3, 2);
        Matrix.print(reshape);
    }

    @Test
    public void diagFlatTest(){
        double[][] d = new double[][]{{0.7}, {0.1}, {0.2}};
        Matrix.print(Matrix.diagFlat(d));
    }

}
