package com.test;

import com.net.CategoricalCrossEntropyLoss;
import com.net.Loss;
import com.net.Tools;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;


public class LossTest {


    @Test
    public void crossEntropyOneHotTest() {
        CategoricalCrossEntropyLoss cceLoss = new CategoricalCrossEntropyLoss();
//        double[] expected = new double[]{0.35667494, 0.69314718, 0.10536052};
        double[][] d = new double[][]{
                {0.7, 0.1, 0.2},
                {0.1, 0.5, 0.4},
                {0.02, 0.9, 0.08}};
        double[] t = {0, 1, 1};
        double average = Loss.calculate(cceLoss, d, Tools.convertToOneHot(t,3));
        Assertions.assertEquals(0.385060880052168, average);
    }
    @Test
    public void crossEntropyLineTest() {
        CategoricalCrossEntropyLoss cceLoss = new CategoricalCrossEntropyLoss();
//        double[] expected = new double[]{0.35667494, 0.69314718, 0.10536052};
        double[][] d = new double[][]{
                {0.7, 0.1, 0.2},
                {0.1, 0.5, 0.4},
                {0.02, 0.9, 0.08}};
        double[] t = {0, 1, 1};
        double average = Loss.calculate(cceLoss, d, t);
        Assertions.assertEquals(0.385060880052168, average);
    }

}
