package com.drobucs.dl.functions;

import Jama.Matrix;

import java.util.Arrays;
import java.util.OptionalDouble;
import java.util.function.Function;

@SuppressWarnings("unused")
public class Functions {
    public static synchronized Matrix softmax(Matrix vector) {
        double[] result = new double[vector.getRowDimension()];
        for (int i = 0; i < vector.getRowDimension(); ++i) {
            result[i] = vector.get(i, 0);
        }
        OptionalDouble max = Arrays.stream(result).max();
        if (max.isEmpty()) {
            throw new IllegalArgumentException("Cannot find max element: vector is empty.");
        }
        result = Arrays.stream(result).map(a -> a - max.getAsDouble()).toArray();
        double sum = Arrays.stream(result).map(Math::exp).sum();
        result = Arrays.stream(result).map(Math::exp).map(a -> a / sum).toArray();
        Matrix mResult = new Matrix(vector.getRowDimension(), 1);
        for (int i = 0; i < vector.getRowDimension(); ++i) {
            mResult.set(i, 0, result[i]);
        }
        return mResult;
    }

    public static synchronized double relu(double x) {
        return Math.max(x, 0);
    }

    public static synchronized double reluDerived(double x) {
        return  Double.compare(x, 0) >= 0 ? 1 : 0;
    }


    public static synchronized double th(double x) {
        return Math.tanh(x);
    }

    public static synchronized double thDerived(double x) {
        return 1.0 - Math.pow(th(x), 2);
    }

    public static synchronized double leakyRelu(double x) {
        return x < 0 ? 0.01 * x : x;
    }

    public static synchronized double leakyReluDerived(double x) {
        return x < 0 ? 0.01 : 1;
    }

    public static synchronized double softPlus(double x) {
        return Math.log(1 + Math.exp(x));
    }

    public static synchronized double softPlusDerived(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    public static synchronized int max(Matrix vector) {
        assert vector.getRowDimension() > 0;
        assert vector.getColumnDimension() == 1;
        double max = Double.MIN_VALUE;
        int index = 0;
        for (int i = 0; i < vector.getRowDimension(); ++i) {
            if (Double.compare(vector.get(i, 0), max) > 0) {
                index = i;
                max = vector.get(i, 0);
            }
        }
        return index;
    }

    public static synchronized Matrix apply(Matrix matrix, Function<Double, Double> function) {
        for (int i = 0; i < matrix.getRowDimension(); ++i) {
            for (int j = 0; j < matrix.getColumnDimension(); ++j) {
                matrix.set(i, j, function.apply(matrix.get(i, j)));
            }
        }
        return matrix;
    }

    public static synchronized double sigma(double x) {
        return 1.0 / (1.0 - Math.exp(-x));
    }

    public static synchronized double sigmaDerived(double x) {
        double value = sigma(x);
        return value * (1.0 - value);
    }

    public static synchronized double crossEntropy(Matrix vector, Matrix correctVector) {
        double sum = 0;
        for (int i = 0; i < vector.getRowDimension(); ++i) {
            sum += correctVector.get(i, 0) * Math.log(vector.get(i, 0));
        }
        return -sum;
    }
}
