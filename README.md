# LexBench

The official repository of the research project ``Revisiting a Pain in the Neck: Semantic Phrase Processing Benchmark for Language Models''.

## Preparing Environments
```bash
conda env create -f environment.yml

cd lexbench
pip install -r requirements.txt
```

## Preparing Data
```bash
unzip resources/dataset.zip -d lexbench/
```

## Running Evaluation on Specific Task 
For example, the command for running idiom interpretation with `Claude-3-opus` is shown below.

```bash
python main.py \
  --task idiom-paraphrase \
  --api_key <Your API key> \
  --model claude-3-opus-20240229 \
  --prompt_path prompts/idiom_paraphrase_zeroshot.txt \
  --example_path dataset/idiom_paraphrase/prepared/examples.tsv \
  --input_path dataset/idiom_paraphrase/prepared/idiom_paraphrase_prepared.tsv \
  --output_path results/idiom-paraphrase_0-shot_claude-3-opus-20240229.json \
  --evaluate \
  --shot_num 0 \
  --max_query 1000 \
  --max_tokens 128 \
  --temperature 0 \
  --presence_penalty 0 \
  --frequency_penalty 0
```

## Benchmarking Scaling-category Semantic Categorization

```bash
./run_lcc_scaling.sh
```
