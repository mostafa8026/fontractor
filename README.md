Install nvtop:
```
apt update
apt install nvtop
```

Install these dependency:

```
pip install tqdm fonttools pytorch-lightning pygit2 tensorboard gradiod
```

Train:

```
python train.py -i -e 500 -a v3 -d 0 1 -n roi-500-v3
```

Tensorboard:
```
tensorboard --logdir=tensorboard --host=0.0.0.0 --port=6006
```

Demo:
```
python train.py -d 0 1 -c -p 8880 -a 0.0.0.0
```