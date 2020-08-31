## Benchmark various VISSL components

We provide several scripts to benchmark VISSL components like dataloader, transforms etc. This can help see the performance of these components and optimize them.

### Benchmarking dataloader with `benchmark_data.py`

To benchmark any dataset, simply run the `benchmark_data.py` on any config of your choice. For example:

```bash
buck run @mode/opt deeplearning/projects/ssl_framework/tools/perf_measurement:benchmark_data -- \
    config=test/integration_test/quick_simclr \
    config.DATA.TRAIN.DATA_SOURCES=[disk_folder] \
    config.DATA.TRAIN.DATASET_NAMES=[imagenet1k_folder] \
    config.DATA.TRAIN.DATA_LIMIT=-1 \
    config.DATA.NUM_DATALOADER_WORKERS=10 \
    config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=64
```

Changing these lines to test fairstore:

```
    config.DATA.TRAIN.DATA_SOURCES=[fairstore] \
    config.DATA.TRAIN.DATASET_NAMES=[fairstore_testing] \
    +config.DATA.FAIRSTORE_SHARED_BUFFER_FILL_WORKERS=0 \
```

Setting the following is useful in debugging the dataset in a single process mode (e.g. with pdb):

```
config.DATA.NUM_DATALOADER_WORKERS=0
```

In the example above, ensure that you have imagenet data already installed in `/data/local`.

This will output the images/sec, sec/image.

### Benchmarking transformations

```bash
buck run @mode/opt deeplearning/projects/ssl_framework/tools/perf_measurement:benchmark_transforms
```
