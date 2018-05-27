# ftprop-nlp

WIP

https://github.com/afriesen/ftprop

https://github.com/pmsosa/CS291K



```bash
nohup python3 train.py --ds=sentiment140 --arch=cnn --nonlin=relu --loss=crossent --epochs 30 --opt=adam --lr=0.0005 --wtdecay=5e-4 --lr-decay=0.1 --lr-decay-epochs 20 25 </dev/null >log1.txt 2>&1 &
nohup python3 train.py --ds=sentiment140 --arch=cnn --nonlin=sign11 --loss=crossent --tp-rule=SoftHinge --epochs 30 --opt=adam --lr=0.0005 --wtdecay=5e-4 --lr-decay=0.1 --lr-decay-epochs 20 25 </dev/null >log2.txt 2>&1 &
nohup python3 train.py --ds=sentiment140 --arch=cnn --nonlin=qrelu --loss=crossent --tp-rule=SoftHinge --epochs 30 --opt=adam --lr=0.0005 --wtdecay=5e-4 --lr-decay=0.1 --lr-decay-epochs 20 25 </dev/null >log3.txt 2>&1 &

nohup python3 train.py --ds=semeval --no-val --arch=bilstm --nonlin=relu --loss=crossent --epochs 50 --opt=adam --lr=0.001 --momentum=0.9 --wtdecay=2e-3 --lr-decay=0.1 --lr-decay-epochs 30 40 </dev/null >log4.txt 2>&1 &
nohup python3 train.py --ds=semeval --no-val --arch=bilstm --nonlin=sign11 --loss=crossent --tp-rule=SoftHinge --epochs 50 --opt=adam --lr=0.001 --momentum=0.9 --wtdecay=2e-3 --lr-decay=0.1 --lr-decay-epochs 30 40 </dev/null >log5.txt 2>&1 &
nohup python3 train.py --ds=semeval --no-val --arch=bilstm --nonlin=qrelu --loss=crossent --tp-rule=SoftHinge --epochs 50 --opt=adam --lr=0.001 --momentum=0.9 --wtdecay=2e-3 --lr-decay=0.1 --lr-decay-epochs 30 40 </dev/null >log6.txt 2>&1 &
```

