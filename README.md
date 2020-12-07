## Exemplar-VAE
Code for reproducing results in [Exemplar VAE](https://arxiv.org/abs/2004.04795) paper; Accepted to NeurIPS 2020

## Requirements
```
pip3 install -r requirements.txt
```
## Exemplar VAE Samples
<img src="images/full_generation.png" width="600"/>


## Exemplar Based Generation
```
python3 analysis.py --dir pretrained_model  --generate
```
<img src="images/celebA_exemplar_generation.png" width="600"/>
<img src="images/exemplar_generation.png" width="600"/>


## Density Estimation 
```
python3 density_estimation.py --prior exemplar_prior --dataset {dynamic_mnist|fashion_mnist|omniglot} --model_name {vae|hvae_2level|convhvae_2level} --number_components {25000|11500} --approximate_prior {True|False} 
```
<img src="images/density_estimation.png" width="500"/>


## Data Augmentation
```
python3 analysis.py --dir pretrained_model  --classify
```
<img src="images/data_augmentation.png" width="400"/>


## Cyclic Generation
```
python3 analysis.py --dir pretrained_model  --cyclic_generation
```
<img src="images/cyclic_generation.png" width="600"/>

