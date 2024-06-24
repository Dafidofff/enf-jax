# Experiments

We provide commands per experiment below. With create_latent_dataset_from_checkpoint.py, a dataset of latent point-clouds is created for a given dataset and a trained model. The latent dataset is saved in the checkpoint folder. With downstream_classification_image_ponita.py, a classifier is trained on the latent point cloud.

Please execute these commands in the repo root.

**Fitting images and downstream classification**
```bash
export PYTHONPATH=. && python experiments/fitting/fit_image_meta.py
export PYTHONPATH=. && python experiments/downstream/create_latent_dataset_from_checkpoint.py
export PYTHONPATH=. && python experiments/downstream/downstream_classification_image_ponita.py 
```

**Fitting shapes and downstream classification**
```bash
export PYTHONPATH=. && python experiments/fitting/fit_shape.py
export PYTHONPATH=. && python experiments/downstream/create_latent_dataset_from_checkpoint_shape.py
export PYTHONPATH=. && python experiments/downstream/downstream_classification_image_ponita.py 
```

