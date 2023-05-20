package com.drobucs.dl.data;


import Jama.Matrix;

public interface LearnData {
    Matrix getValuesVector();
    Matrix getProbabilityVector();
}
