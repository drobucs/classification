package com.drobucs.dl.nn;

import Jama.Matrix;
import com.drobucs.dl.data.TypedLearnData;
import com.drobucs.dl.functions.Functions;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.function.Function;

import static com.drobucs.dl.functions.Functions.*;

@SuppressWarnings("unused")
public class MultiLayerNet<T, D extends TypedLearnData<T>> {
    protected final Layer[] layers;
    protected final double learningSpeed;
    protected final Function<Double, Double> activation;
    protected final Function<Double, Double> activationDerived;

    public MultiLayerNet(double learningSpeed, Function<Double, Double> activation,
                         Function<Double, Double> activationDerived, int... layersSizes) {
        assert layersSizes.length > 1 : "The minimum number of layers is 2, but actual " + layersSizes.length;
        layers = new Layer[layersSizes.length - 1];
        for (int i = 0; i < layers.length; ++i) {
            layers[i] = new Layer(layersSizes[i], layersSizes[i + 1]);
        }
        this.learningSpeed = learningSpeed;
        this.activation = activation;
        this.activationDerived = activationDerived;
    }

    protected void run(D learnData) {
        layers[0].setInput(learnData.getValuesVector());
        layers[0].calculate();
        for (int i = 1; i < layers.length; ++i) {
            layers[i].setInput(
                    layers[i - 1].activate(activation)
            );
            layers[i].calculate();
        }
    }

    protected Matrix activateLast() {
        return last().activate(activation);
    }

    protected Layer last() {
        return layers[layers.length - 1];
    }

    protected void backpropagation(D learnData) {
        Matrix correctProbabilities = learnData.getProbabilityVector();
        Matrix dEdH = Functions.softmax(activateLast()).minus(correctProbabilities);
        for (int i = layers.length - 1; i >= 0; --i) {
            Matrix dEdT = dEdH.arrayTimes(apply(layers[i].output.copy(), activationDerived));
            Matrix dEdB = dEdT.copy();
            Matrix dEdW = layers[i].input.times(dEdT.transpose());
            dEdH = layers[i].edgesWeights.times(dEdT);
            layers[i].bayes = layers[i].bayes.minus(dEdB.times(learningSpeed));
            layers[i].edgesWeights = layers[i].edgesWeights.minus(dEdW.times(learningSpeed));
        }
    }

    public void learn(D learnData) {
        run(learnData);
        backpropagation(learnData);
    }

    public void learn(List<D> learnData, int epochs, int shufflePeriod) {
        for (int i = 0; i < epochs; ++i) {
            if (i % shufflePeriod == 0) {
                Collections.shuffle(learnData);
            }
            for (D data : learnData) {
                learn(data);
            }
        }
    }

    public void learn(D[] learnData, int epochs, int shufflePeriod) {
        learn(Arrays.asList(learnData), epochs, shufflePeriod);
    }

    public void learn(D[] learnData) {
        learn(learnData, 1, 1);
    }

    public Matrix getLastLayer() {
        return activateLast();
    }

    public double getError(D learnData, Matrix correctVector) {
        run(learnData);
        return Functions.crossEntropy(softmax(getLastLayer()), correctVector);
    }

    public T getType(D typedLearnData) {
        run(typedLearnData);
        int i = max(getLastLayer());
        return typedLearnData.getTypes()[i];
    }
}
