# Code for Selection, Ensemble, and Adaptation: Advancing Multi-Source-Free Domain Adaptation via Architecture Zoo

To train a source model with a specific architecture, run

```bash
python train_source.py --dset office-home --lr 1e-3 --batch_size 128 --net resnet101 --optimizer sgd
```

Or you can simply run to train all source models

```bash
./source_training.sh
```

For adaptation, run

```bash
python adaptation.py --trans_method SUTE --lr 1e-2 --max_epoch 15
```
