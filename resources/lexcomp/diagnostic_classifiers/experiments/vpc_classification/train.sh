#!/bin/bash

declare -a word_embeddings=(word2vec glove fasttext)
declare -a contextual_embeddings=(elmo openai bert)
declare -a layers=(all top)
declare -a all_encodings=(noenc bilm)

for encoding in "${all_encodings[@]}"
do
    for embeddings in "${word_embeddings[@]}"
    do
        allennlp train diagnostic_classifiers/experiments/vpc_classification/vpc_classification_${embeddings}_${encoding}.json \
                -s output/diagnostic_classifiers/vpc_classification/${embeddings}/${encoding} \
                --include-package diagnostic_classifiers &
    done
    wait

    for layer in "${layers[@]}"
    do
        for embeddings in "${contextual_embeddings[@]}"
        do
            allennlp train diagnostic_classifiers/experiments/vpc_classification/vpc_classification_${embeddings}_${layer}_${encoding}.json \
                    -s output/diagnostic_classifiers/vpc_classification/${embeddings}_${layer}/${encoding} \
                    --include-package diagnostic_classifiers &
        done
        wait
    done
    wait
done
