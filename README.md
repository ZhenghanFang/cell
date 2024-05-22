# Cell Tracking
Train a U-Net for detecting all the cells of a specific type in a 3D microscopy image of mouse brain.

## Install Environment
```
conda env create -f environment.yml
conda activate cell
```

## Train
```bash
python train.py
```
The trained model will be saved as `checkpoint.pth`. Training will take approx. 18 hours. A trained checkpoint is provided on [Google drive](https://drive.google.com/file/d/1CibEB6hrJ4l-9xpe06hh3YNfZrWHy3HB/view?usp=sharing). 

## Test
Run `test.ipynb`. 

Numerical results are printed at the end of "Numerical Results" section. Visual result is saved as `result.{gif,mov}`.
