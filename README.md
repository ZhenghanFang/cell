# Cell Tracking
Train a U-Net for detecting all the cells of a specific type in a 3D microscopy image of mouse brain.

## Install Environment
```bash
conda env create -f environment.yml
conda activate cell
```

## Train
```bash
python train.py
```
The trained model is saved as `checkpoint.pth`.

## Test
Run `test.ipynb`. 

Numerical results are printed at the end of "Numerical Results" section. Visual result is saved as `result.{gif,mov}`.
