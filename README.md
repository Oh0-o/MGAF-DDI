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
