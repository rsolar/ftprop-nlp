# ftprop-nlp

WIP

https://github.com/afriesen/ftprop

https://github.com/pmsosa/CS291K

```bash
# sentiment140
python train.py --ds=sentiment140 --arch=cnn --nonlin=relu --loss=crossent --epochs 30 --opt=adam --lr=0.0005 --wtdecay=5e-4 --lr-decay=0.1 --lr-decay-epochs 20 25
python train.py --ds=sentiment140 --arch=cnn --nonlin=sign11 --loss=crossent --tp-rule=SoftHinge --epochs 30 --opt=adam --lr=0.0005 --wtdecay=5e-4 --lr-decay=0.1 --lr-decay-epochs 20 25
python train.py --ds=sentiment140 --arch=cnn --nonlin=qrelu --loss=crossent --tp-rule=SoftHinge --epochs 30 --opt=adam --lr=0.0005 --wtdecay=5e-4 --lr-decay=0.1 --lr-decay-epochs 20 25

# semeval
python train.py --ds=semeval --no-val --arch=bilstm --nonlin=relu --loss=crossent --epochs 50 --opt=adam --lr=0.001 --momentum=0.9 --wtdecay=2e-3 --lr-decay=0.1 --lr-decay-epochs 30 40
python train.py --ds=semeval --no-val --arch=bilstm --nonlin=sign11 --loss=crossent --tp-rule=SoftHinge --epochs 50 --opt=adam --lr=0.001 --momentum=0.9 --wtdecay=2e-3 --lr-decay=0.1 --lr-decay-epochs 30 40
python train.py --ds=semeval --no-val --arch=bilstm --nonlin=qrelu --loss=crossent --tp-rule=SoftHinge --epochs 50 --opt=adam --lr=0.001 --momentum=0.9 --wtdecay=2e-3 --lr-decay=0.1 --lr-decay-epochs 30 40

# semeval (testing)
python train.py --ds=semeval --no-val --arch=bilstm --nonlin=relu --loss=crossent --epochs 50 --opt=adam --lr=0.001 --momentum=0.9 --wtdecay=2e-3 --lr-decay=0.1 --lr-decay-epochs 30 40 --test-model ".\experiments\logs\semeval\2018.05.27_08.45.19.bilstm.crossent.adam_lr0.001_mu0.9_wd0.002noval.relu\model_checkpoint_epoch50.pth.tar"
python train.py --ds=semeval --no-val --arch=bilstm --nonlin=sign11 --loss=crossent --tp-rule=SoftHinge --epochs 50 --opt=adam --lr=0.001 --momentum=0.9 --wtdecay=2e-3 --lr-decay=0.1 --lr-decay-epochs 30 40 --test-model ".\experiments\logs\semeval\2018.05.27_09.05.52.bilstm.crossent.adam_lr0.001_mu0.9_wd0.002noval.sign11-SoftHinge\model_checkpoint_epoch50.pth.tar"
python train.py --ds=semeval --no-val --arch=bilstm --nonlin=qrelu --loss=crossent --tp-rule=SoftHinge --epochs 50 --opt=adam --lr=0.001 --momentum=0.9 --wtdecay=2e-3 --lr-decay=0.1 --lr-decay-epochs 30 40 --test-model ".\experiments\logs\semeval\2018.05.27_09.10.12.bilstm.crossent.adam_lr0.001_mu0.9_wd0.002noval.qrelu-SoftHinge\model_checkpoint_epoch50.pth.tar"
```
