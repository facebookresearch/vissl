Run the tests against the PyPI backage.


1. Make sure you have cuda 10.1 working as
that is assumed by the testing phase. E.g. on the FAIR cluster

```
module purge
module load cuda/10.2
module load NCCL/2.7.8-1-cuda.11.0
```

2. Run `bash run.sh` in this directory.
