# VOC Example


## Training


```bash
./torchconvs/datasets/download_flair1.sh

 python -m example.train -g 0 --model deeplab --batch-size 24 --data-path $PATH$

./view_log logs/XXX/log.csv
```


## Evaluating

```bash
 python -m example.evaluate -g 0 /example/logs/deeplab/model_best.pth.tar

./learning_curve example/logs/deeplab/log.csv
```

