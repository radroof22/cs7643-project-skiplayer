# cs7643-project-skiplayer
Improvement on SkipLayer with early exiting.

## Setup
Follow instructions [here](https://piazza.com/class/m5k29i4gzsf4ab/post/484)

## Experiments Guide
Edit [here](https://docs.google.com/spreadsheets/d/1On8nT8upmKvkyMd5u0Jgk9Sxy3gStIQqFxJTxCfmsn8/edit?gid=0#gid=0)

## Modifications

When running benchmark.py, you can add in the dynamic_early_exit_mode flag. Currently, you can set the flag value to 'consistent_tokens', which sets the draft model to exit once it has seen the same token four times in a row. An example of a command is:

```
torchrun benchmark.py --model facebook/layerskip-llama2-7B     --dataset cnn_dm_summarization     --num_samples 100     --generation_strategy self_speculative     --exit_layer 8     --num_speculations 6     --output_dir ./logs --dynamic_early_exit_mode consistent_tokens
```

You can either omit the dynamic_early_exit_mode flag or set it to 'none' to use the original benchmarking code.