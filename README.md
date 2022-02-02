
# Learning cortical representations through perturbed and adversarial dreaming

This repository contains the code to reproduce the results of the eLife submission "Learning cortical representations through perturbed and adversarial dreaming" (also available on [ArXiv](https://arxiv.org/abs/2109.04261)).

## Requirements 

To install requirements:
 ```
 pip install -r requirements.txt
 
```
## Training & Evaluation 

In order to train the model with for example the CIFAR-10 dataset, for 50 epochs, with all phases (Wake, NREM REM), execute: 
```
python main_PAD.py --dataset 'cifar10' --niter 50 --batchSize 64  --outf 'model_wnr' --nz 256  --is_continue 1 --W 1.0  --N 1.0  --R 1.0 
```

Setting one of the phase parameters (```W, N, R```) to zero will remove the phase from training. At each epoch, the endoder and generator networks, as well as the training losses, are saved into the file ```trained.pth```. 

### Figure 3
Once the previous command has been executed for different conditions, in order to display samples for Figure 3, for early and late training, execute:
```
python fig3_generate_samples.py --dataset 'cifar10' --nz 256 --outf 'model_wnr' 
```
In order to compute FID score, use the full PAD model trained networks obtained above for 4 runs (ex: model_wnr, model_wnr1, etc.) and execute: 
```
python fig3_compute_FID.py --dataset 'cifar10' --outf 'model_wnr' n_samples 500 --split 20
```
Once computed, call the following file in order to display the FID bar graph: 
```
python fig3_plot_FID.py
```



### Figure 4
In order to compute linear classification accuracy, execute: 
```
python linear_classif.py --dataset 'cifar10' --niterC 20  --outf 'model_wnr' --nz 256
```

Classification accuracy can be computed at each epoch by running the following command in a batch script:
```
dset='cifar10'
folder='model_wnr'
for i in {1..50}
do
        python main_PAD.py --dataset $dset --niter $i --batchSize 64  --outf $folder --nz 256  --is_continue 1 --W 1.0  --N 1.0  --R 1.0  --epsilon 0.0 
        python linear_classif.py --dataset $dset --niterC 20  --outf $folder --nz 256
done
```
Several runs can be saved using the following file names: model_wnr, model_wnr1, model_wnr2, model_wnr3, etc. 
Following this nomenclature, Figure 4 results can be displayed by executing:
```
python fig4_plot_accuracies.py --dataset 'cifar10' 
```

### Figure 5
In order to compute accuracies with different level of occlusions, execute in a batch script:
```
dset='cifar10'
folder='model_wnr'
for proba in {0..100..10}
do
       python linear_classif_occ.py --dataset $dset --niterC 20  --outf $folder --acc_file 'accuracies_levels.pth' --tile_size 4  --nz 256 --drop $proba
done

```
To display Figure 5, execute: 
```
python fig5_plot_accuracies.py 
```

### Figure 6 
PCA results are displayed by loading trained networks while executing the following: 
```
python fig6_plot_PCA.py --dataset 'cifar10' --outf 'model_wnr' --nz 256 --n_samples 50 --n_samples 1000 --n_samples_per_class 50 
```
Intra/Inter class and clean/occluded ratios by loading trained networks for each condition (4 runs) and executing: 
```
python fig6_compute_distances.py --dataset 'cifar10' --n_samples 1000 
```
These metrics are saved in the file ```cifar10_clustering_metrics.pth```.
Once computed for each dataset, in order to plot Fig.6e and 6f, load the metrics file executing:  
```python fig6_plot_distances.py --
```
by setting the argument ```occlusions``` to ```False``` for Fig.6e and to ```True``` for Fig.6f. 














