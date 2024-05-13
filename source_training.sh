
######## For Zoo-MSFDA main

python train_source2.py --dset office-home --lr 1e-2 --batch_size 128 --net resnet50  --optimizer sgd
python train_source.py --dset office-home --lr 1e-3 --batch_size 128 --net resnet101 --optimizer sgd
python train_source.py --dset office-home --lr 1e-3 --batch_size 128 --net efficientnet_v2_l --optimizer sgd
python train_source.py --dset office-home --lr 1e-3 --batch_size 128 --net efficientnet_v2_m --optimizer sgd
python train_source.py --dset office-home --lr 1e-3 --batch_size 128 --net efficientnet_v2_s --optimizer sgd
python train_source.py --dset office-home --lr 1e-3 --batch_size 128 --net efficientnet_v2_m --optimizer sgd
python train_source.py --dset office-home --lr 1e-3 --batch_size 128 --net swin_b --optimizer sgd
python train_source.py --dset office-home --lr 1e-3 --batch_size 128 --net swin_l --optimizer sgd
python train_source.py --dset office-home --lr 1e-3 --batch_size 128 --net swin_s --optimizer sgd
python train_source.py --dset office-home --lr 1e-3 --batch_size 128 --net swin_t --optimizer sgd
python train_source.py --dset office-home --lr 1e-3 --batch_size 128 --net vit_b_16 --optimizer sgd
python train_source.py --dset office-home --lr 1e-3 --batch_size 128 --net vit_b_32 --optimizer sgd
python train_source.py --dset office-home --lr 1e-3 --batch_size 128 --net vit_l_16 --optimizer sgd
python train_source.py --dset office-home --lr 1e-3 --batch_size 128 --net vit_l_32 --optimizer sgd
python train_source.py --dset office-home --lr 1e-3 --batch_size 128 --net vit_h_14 --optimizer sgd --fix true
python train_source.py --dset office-home --lr 1e-3 --batch_size 128 --net swin_v2_t --optimizer sgd
python train_source.py --dset office-home --lr 1e-3 --batch_size 128 --net swin_v2_s --optimizer sgd
python train_source.py --dset office-home --lr 1e-3 --batch_size 128 --net swin_v2_b --optimizer sgd