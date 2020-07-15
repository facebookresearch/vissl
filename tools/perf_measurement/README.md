## Benchmark various VISSL components

We provide several scripts to benchmark VISSL components like dataloader, transforms etc. This can help see the performance of these components and optimize them.

### Benchmarking dataloader with `benchmark_data.py`

To benchmark any dataset, simply run the `benchmark_data.py` on any config of your choice. For example:

```bash
cd $HOME/vissl
python tools/perf_measurement/benchmark_data.py config=pretrain/swav/swav_node_resnet config.DATA.TRAIN.DATA_SOURCES=[disk_
folder] config.DATA.TRAIN.DATASET_NAMES=[imagenet1k_folder] config.DATA.TRAIN.DATA_LIMIT=-1 config.DATA.NUM_DATALOADER_WORKERS=10 config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=64
```

This will output the images/sec, sec/image.
