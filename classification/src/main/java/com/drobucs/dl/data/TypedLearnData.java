package com.drobucs.dl.data;

public interface TypedLearnData<R> extends LearnData {
    R[] getTypes();
}
