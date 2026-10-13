# MGAF-DDI

**MGAF-DDI**: a drug combination risk-level prediction method based on multi-relational graph attention fusion.

## Layout

| Directory | Task |
|---|---|
| `multi_relation/` | Risk-level prediction on the DDInter dataset |
| `multi_label/` | Multi-label composite DDI type identification on DDInter |

## Requirements

Python 3.x with TensorFlow 2.x. See `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Usage

```bash
cd multi_relation/Model
python MGAF_main.py

cd ../..
cd multi_label/Model
python MGAF_main.py
```

Hyperparameters can be changed via the command line (see `utility/parser.py`).

## Data

Datasets are **not included** in this repository. Please download DDInter from https://ddinter.scbdd.com/ and place the processed data under each task's `Data/` directory (expected filenames are documented in `utility/loader_mgaf.py`).

## Acknowledgements

The code is adapted from [RANEDDI](https://github.com/DongWenMin/RANEDDI) (Yu et al., *RANEDDI: Relation-aware network embedding for drug-drug interaction prediction*, Information Sciences, 2022) with substantial modifications.

<!-- TODO: add the MGAF-DDI paper citation here upon publication. -->
