# Code for Selection, Ensemble, and Adaptation: Advancing Multi-Source-Free Domain Adaptation via Architecture Zoo

To train a source model with a specific architecture, run

```bash
python train_source.py --dset office-home --lr 1e-3 --batch_size 128 --net resnet101 --optimizer sgd
```

Or you can simply run the following to train all source models

```bash
./source_training.sh
```

Then, run the following to generate configs for Zoo_MSFDA

```bash
python extract_benchmark.py
```

Run the following for transferability estimation and adaptation using our SUTE method and SEA framework. You can also replace our SUTE by existing transferability estimation methods (such as ANE, NMI, and MDE) by adjusting "trans_method".

```bash
python adaptation.py --trans_method SUTE --lr 1e-2 --max_epoch 15
```
