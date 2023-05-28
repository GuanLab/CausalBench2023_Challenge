# CausalBench ICLR-23 Challenge 

This repository includes our winning solution on the [2023 CausalBench Challenge](https://www.gsk.ai/causalbench-challenge/). The method was developed by Kaiwen Deng ([dengkw@umich.edu](mailto:dengkw@umich.edu)) and Yuanfang Guan ([gyuanfan@umich.edu](mailto:gyuanfan@umich.edu)). Please contact us if you have any questions or suggestions.

[CausalBench](https://arxiv.org/abs/2210.17283) is a comprehensive benchmark suite for evaluating network inference methods on perturbational single-cell gene expression data. 
CausalBench introduces several biologically meaningful performance metrics and operates on two large, curated and openly available benchmark data sets for evaluating methods on the inference of gene regulatory networks from single-cell data generated under perturbations.

## Install

```bash
pip install -r requirements.txt
```

## Use

### Setup

- Create a data directory. This will hold any preprocessed and downloaded datasets for faster future invocation.
  - `$ mkdir /path/to/data/`
  - _Replace the above with your desired cache directory location._
- Create an output directory. This will hold all program outputs and results.
  - `$ mkdir /path/to/output/`
  - _Replace the above with your desired output directory location._
- Create a plot directory. This will hold all plots and final metrics of your experiments.
  - `$ mkdir /path/to/plots`
  - _Replace the above with your desired plots directory location._


### Run the full benchmark suite?

Before running the pipeline, you may need to modify or notice:
- `DATASET_NAME="weissmann_rpe1"`. There're two available dataset: rpe1 and k562
- Change `OUTPUT_DIRECTORY`, `DATA_DIRECTORY` and `PLOT_DIRECTORY` to your pre-defined paths

```bash
bash run_pipeline.sh
```

### Reference
[https://github.com/causalbench/causalbench-starter](https://github.com/causalbench/causalbench-starter)