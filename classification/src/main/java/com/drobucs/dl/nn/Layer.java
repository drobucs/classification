package com.drobucs.dl.nn;

import Jama.Matrix;

import java.util.function.Function;

import static com.drobucs.dl.functions.Functions.apply;

class Layer {
    protected Matrix input;
    protected Matrix output;
    protected Matrix edgesWeights;
    protected Matrix bayes;
    protected final int rows;
    protected final int columns;
    public Layer(int rows, int columns) {
        input = Matrix.random(rows, 1);
        output = Matrix.random(columns, 1);
        edgesWeights = Matrix.random(rows, columns);
        bayes = Matrix.random(columns, 1);
        this.rows = rows;
        this.columns = columns;
    }

    public void setInput(Matrix vector) {
        assert vector.getColumnDimension() == input.getColumnDimension();
        assert vector.getRowDimension() == input.getRowDimension();
        input = vector.copy();
    }

    public void calculate() {
        output = edgesWeights.transpose().times(input).plus(bayes);
        assert output.getColumnDimension() == 1;
        assert edgesWeights.getColumnDimension() == output.getRowDimension();
        assert edgesWeights.getRowDimension() == rows;
        assert edgesWeights.getColumnDimension() == columns;
    }

    public Matrix activate(Function<Double, Double> activation) {
        return apply(output.copy(), activation);
    }
}
