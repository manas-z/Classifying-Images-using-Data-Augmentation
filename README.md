# Image Classification with Data Augmentation

This project trains a small convolutional neural network to distinguish between **flower** and **landscape** patches that are generated from scikit-learn's sample images. Two training regimes are compared:

- **Baseline** – images are normalized only.
- **Augmented** – random flips, rotations, affine transforms and colour jitter are applied in addition to normalization.

Both runs share the exact same architecture (three convolutional blocks followed by two fully connected layers with dropout) and are trained for the same number of epochs. Training artefacts (metrics and plots) are written to the `outputs/` directory, which is ignored by git so that fresh runs can regenerate them locally.

## Getting started

1. (Optional) Create and activate a virtual environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Training via script

The `train.py` entry point regenerates the dataset, trains both the baseline and augmented models, and stores their metrics and plots.

```bash
python train.py --epochs 10 --batch-size 32 --learning-rate 1e-3 --output-dir outputs
```

Key command-line options:

- `--epochs`: number of training epochs (default `10`).
- `--batch-size`: batch size for the data loaders (default `32`).
- `--learning-rate`: Adam learning rate (default `1e-3`).
- `--output-dir`: directory for JSON metrics and PNG plots (default `outputs/`).
- `--seed`: random seed used for patch sampling and weight initialisation (default `42`).

Running the script will print per-epoch training/validation metrics for both runs and produce the following artefacts per configuration in the specified `output-dir`:

- `{run}_metrics.json` – per-epoch loss and accuracy.
- `{run}_learning_curves.png` – side-by-side training/validation loss and accuracy plots.
- `{run}_confusion_matrix.png` – validation confusion matrix at the best validation accuracy epoch.

## Training via notebook

The `Image_Classification_Augmentation.ipynb` notebook mirrors the behaviour of `train.py` for interactive exploration. Executing the notebook end-to-end will:

1. Regenerate the flower vs. landscape patch datasets.
2. Train the baseline and augmented models for the configured number of epochs.
3. Save the same metrics and plots to `outputs/`.
4. Display summary tables of the recorded metrics inside the notebook for quick inspection.

## Dataset details

- Source images: [`sklearn.datasets.load_sample_images`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_sample_images.html).
- Patch size: `64×64`.
- Train patches per class: `80`.
- Validation patches per class: `200`.

Each invocation of the script/notebook resamples the patches with the configured random seed so the training data is freshly generated for every run.

## Reproducing reported results

Using the default hyperparameters (10 epochs, batch size 32, learning rate 1e-3, seed 42) the expected best validation accuracies are approximately:

- **Baseline:** ~58.5%
- **Augmented:** ~68.3%

Exact numbers can vary between runs due to the stochastic nature of patch sampling and optimisation.
