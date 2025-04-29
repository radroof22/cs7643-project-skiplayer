# cs7643-project-skiplayer
Our project is an extension on the LayerSkip repository. We have implemented a few different variations of dynamic early exiting along with random layer skipping and tested our strategies on a variety of different datasets along with pretraining the model for a couple of the datasets and testing them further.

ALthough our results were not positive, we believe that our approach may work better on larger models compared to the LlaMa 7B that we tested it on (due to compute constraints).

## Setup for the Dataset and Environment
Follow instructions [here](https://piazza.com/class/m5k29i4gzsf4ab/post/484)

## Experiments Guide
Edit [here](https://docs.google.com/spreadsheets/d/1On8nT8upmKvkyMd5u0Jgk9Sxy3gStIQqFxJTxCfmsn8/edit?gid=0#gid=0)

## Modifications

When running benchmark.py, you can add in the dynamic_early_exit_mode flag. Currently, you can set the flag value to 'consistent_tokens', which sets the draft model to exit once it has seen the same token four times in a row. An example of a command is:

```
torchrun benchmark.py --model facebook/layerskip-llama2-7B     --dataset cnn_dm_summarization     --num_samples 100     --generation_strategy self_speculative     --exit_layer 8     --num_speculations 6     --output_dir ./logs --dynamic_early_exit_mode consistent_tokens --layer_skip_proportion 0.25
```

You can either omit the dynamic_early_exit_mode flag or set it to 'none' to use the original benchmarking code. Other options:
random - randomly exit at a layer between the first and fourth quartile
logits_current - use logits at each layer to exit when we reach a desired confidence threshold
logits_future - use logits at each layer to modify the exit layer for future generations

Note for logits_future: please switch to the future_logits branch in order to run this.

You can set the layer_skip_proportion flag to a number from 0 to 1 (defaulted to 0). For example, if set to 0.25, every layer used to create draft tokens has a 25% chance of being skipped.

