# MPASL
MPASL: Multi-perspective Learning Knowledge Graph Attention Network for Synthetic Lethality Prediction


## Introduction
we propose MPASL, a multi-perspective learning knowledge graph attention network to enhance synthetic lethality prediction.

## Files in the folder

- `data/`: datasets
  - `Leave_out/`
  - `SynlethDB/`
- `src/model/`: implementation of MPASL.
- `output/`: storing log files
- `misc/`: storing genes being evaluating, and sharing embeddings.

## Environment Requirement
The code has been tested running under Python 3.6.5. The required packages are as follows:
* tensorflow == 1.15.0
* numpy == 1.16.0
* scipy == 1.1.0
* sklearn == 0.20.0

## Example to Run the Codes

* MPASL
```
$ cd src/bash/
$ bash main_run.sh "MPASL" $dataset $gpu

```
  
* `dataset`
  * It specifies the dataset.
  * Here we provide two options, including  * `Leave_out`, or `SynlethDB`.
 

* `gpu`
  * It specifies the gpu, e.g. * `0`, `1`, and `2`.

