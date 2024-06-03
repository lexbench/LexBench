#!/bin/bash

declare -a word_embeddings=(word2vec glove fasttext)
declare -a contextual_embeddings=(elmo openai bert)
declare -a layers=(all top)
declare -a all_encodings=(noenc bilm att)

for encoding in "${all_encodings[@]}"
do
    for embeddings in "${word_embeddings[@]}"
    do
        allennlp predict output/diagnostic_classifiers/lvc_classification/${embeddings}/${encoding}/model.tar.gz \
        diagnostic_classifiers/data/lvc_classification/test.jsonl \
        --output-file output/diagnostic_classifiers/lvc_classification/${embeddings}/${encoding}/test.predictions \
        --predictor sentence_span_classification \
        --include-package diagnostic_classifiers &
    done
    wait

    for embeddings in "${contextual_embeddings[@]}"
    do
        for layer in "${layers[@]}"
        do
            allennlp predict output/diagnostic_classifiers/lvc_classification/${embeddings}_${layer}/${encoding}/model.tar.gz \
            diagnostic_classifiers/data/lvc_classification/test.jsonl \
            --output-file output/diagnostic_classifiers/lvc_classification/${embeddings}_${layer}/${encoding}/test.predictions \
            --predictor sentence_span_classification \
            --include-package diagnostic_classifiers &
        done
        wait
    done
    wait
done
