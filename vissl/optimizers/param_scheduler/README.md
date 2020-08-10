## How to setup different Learning rate schedule

Below we provide some examples of how to setup various types of Learning rate schedules. Note that these are merely some examples and you should set your desired parameter values.

1. Cosine

```yaml
name: cosine
start_value: 0.15   # LR for batch size 256
end_value: 0.0000
```

2. Multi-Step

```yaml
name: multistep
values: [0.01, 0.001]
milestones: [1]
update_interval: epoch
```

3. Linear Warmup + Cosine

```yaml
name: composite
schedulers:
  - name: linear
    start_value: 0.6
    end_value: 4.8
  - name: cosine
    start_value: 4.8
    end_value: 0.0048
interval_scaling: [rescaled, fixed]
update_interval: step
lengths: [0.1, 0.9]                 # 100ep
```

4. Cosine with restarts

```yaml
name: cosine_warm_restart
start_value: 0.15   # LR for batch size 256
end_value: 0.00015
# restart_interval_length: 0.334
# wave_type: full
restart_interval_length: 0.5
wave_type: half
```

5. Linear warmup + cosine with restarts

```yaml
name: composite
schedulers:
    - name: linear
        start_value: 0.6
        end_value: 4.8
    - name: cosine_warm_restart
        start_value: 4.8
        end_value: 0.0048
        # wave_type: half
        # restart_interval_length: 0.5
        wave_type: full
        restart_interval_length: 0.334
interval_scaling: [rescaled, rescaled]
update_interval: step
lengths: [0.1, 0.9]                 # 100ep
```

6. Multiple linear warmups and cosine
```yaml
schedulers:
    - name: linear
        start_value: 0.6
        end_value: 4.8
    - name: cosine
        start_value: 4.8
        end_value: 0.0048
    - name: linear
        start_value: 0.0048
        end_value: 2.114
    - name: cosine
        start_value: 2.114
        end_value: 0.0048
update_interval: step
interval_scaling: [rescaled, rescaled, rescaled, rescaled]
lengths: [0.0256, 0.48722, 0.0256, 0.46166]         # 1ep IG-500M
```
