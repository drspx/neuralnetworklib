package test;


import net.neuralnetwork.*;
import org.junit.jupiter.api.Test;

public class OptimizerSGDWithMomentumTest {

    @Test
    public void momentumTest() {
        DenseLayer dense1 = new DenseLayer(2, 64);
        ActivationReLU activation1 = new ActivationReLU();
        DenseLayer dense2 = new DenseLayer(64, 3);
        ActivationSoftmaxLossCategoricalCrossentropy lossActivation = new ActivationSoftmaxLossCategoricalCrossentropy();
        OptimizerSGDWithMomentum.decay = 8e-8;
        OptimizerSGDWithMomentum.momentum = 0.9;

        for (int i = 0; i < 1; i++) {
            dense1.forward(Tools.spiralDataX());
            activation1.forward(dense1.output);
            dense2.forward(activation1.output);
            double loss = lossActivation.forward(dense2.output, Tools.spiralDatay());
            double accuracy = Tools.accuracy(lossActivation.output, Tools.spiralDatay());


            lossActivation.backward(lossActivation.output, Tools.spiralDatay());
            dense2.backward(lossActivation.dInputs);
            activation1.backward(dense2.dInputs);
            dense1.backward(activation1.dInputs);

            OptimizerSGDWithMomentum.updateParamsWithMomentum(dense1);
            OptimizerSGDWithMomentum.updateParamsWithMomentum(dense2);
        }
    }
}
