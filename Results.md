## Base Testing without LayerSkip

This runs the benchmark using the basic implementation of auto-regressive generation.

```json
{
  "model": "facebook/layerskip-llama2-7B",
  "model_args": null,
  "seed": 42,
  "output_dir": "./logs",
  "model_arg": {}
}
```

```json
{
  "dataset": "cnn_dm_summarization",
  "data_path": null,
  "random_shuffle": true,
  "num_samples": 100,
  "n_shot": 0,
  "template": null
}
```

```json
{
  "max_steps": 512,
  "exit_layer": -1,
  "num_speculations": -1,
  "generation_strategy": "autoregressive",
  "sample": true,
  "temperature": 0.6,
  "top_k": 0,
  "top_p": 0.9,
  "no_repeat_ngram_size": null,
  "stop_words": null,
  "stop_token_ids": []
}
```

```json
{
  "predicted_text": {
    "rouge-l": 0.12595337629318237,
    "rouge-1": 0.18180321156978607,
    "rouge-2": 0.08389847725629807,
    "rouge-3": 0.0493178553879261,
    "bleu_score": 0,
    "exact_match": 1737.1099853515625
  },
  "acceptance_rate": {
    "mean": 0
  },
  "total_time": {
    "mean": 8.73208086013794
  },
  "time_per_token": {
    "mean": 0.017876401264220475
  },
  "tokens_per_second": {
    "mean": 55.95230880737304
  }
}
```

## Base Testing with LayerSkip

This runs a benchmark using the current implementation of **LayerSkip**.

```json
{
  "model": "facebook/layerskip-llama2-7B",
  "model_args": null,
  "seed": 42,
  "output_dir": "./logs",
  "model_arg": {}
}
```

```json
{
  "dataset": "cnn_dm_summarization",
  "data_path": null,
  "random_shuffle": true,
  "num_samples": 100,
  "n_shot": 0,
  "template": null
}
```

```json
{
  "max_steps": 512,
  "exit_layer": 8,
  "num_speculations": 6,
  "generation_strategy": "self_speculative",
  "sample": true,
  "temperature": 0.6,
  "top_k": 0,
  "top_p": 0.9,
  "no_repeat_ngram_size": null,
  "stop_words": null,
  "stop_token_ids": []
}
```

```json
{
  "predicted_text": {
    "rouge-l": 0.12508590519428253,
    "rouge-1": 0.17781290411949158,
    "rouge-2": 0.08037687093019485,
    "rouge-3": 0.04680638015270233,
    "bleu_score": 0,
    "exact_match": 1730.699951171875
  },
  "acceptance_rate": {
    "mean": 0.7252408319711685
  },
  "total_time": {
    "mean": 5.051447117328644
  },
  "time_per_token": {
    "mean": 0.010429023425094783
  },
  "tokens_per_second": {
    "mean": 103.09081768035888
  }
}
```

## Base Testing with LayerSkip - Using Fixed Exit Layer of 2

This runs a benchmark using the current implementation of **LayerSkip** with a lower than optimal exit layer. This is expected to have a low acceptance rate and lower tokens per second.

```json
{
  "model": "facebook/layerskip-llama2-7B",
  "model_args": null,
  "seed": 42,
  "output_dir": "./logs",
  "model_arg": {}
}
```

```json
{
  "dataset": "cnn_dm_summarization",
  "data_path": null,
  "random_shuffle": true,
  "num_samples": 100,
  "n_shot": 0,
  "template": null
}
```

```json
{
  "max_steps": 512,
  "exit_layer": 2,
  "num_speculations": 6,
  "generation_strategy": "self_speculative",
  "sample": true,
  "temperature": 0.6,
  "top_k": 0,
  "top_p": 0.9,
  "no_repeat_ngram_size": null,
  "stop_words": null,
  "stop_token_ids": []
}
```

```json
{
  "predicted_text": {
    "rouge-l": 0.126164048910141,
    "rouge-1": 0.1815703809261322,
    "rouge-2": 0.08243454247713089,
    "rouge-3": 0.047392792999744415,
    "bleu_score": 0,
    "exact_match": 1745.27001953125
  },
  "acceptance_rate": {
    "mean": 0.05320687802508473
  },
  "total_time": {
    "mean": 11.75547928094864
  },
  "time_per_token": {
    "mean": 0.023881013263016938
  },
  "tokens_per_second": {
    "mean": 42.13248233795166
  }
}
```

## Dynamic Early Exit - Exit on repeated token production

This runs a benchmark using a modified implementation of **LayerSkip** with the draft model exiting once it sees the same token twice.

```json
{
  "model": "facebook/layerskip-llama2-7B",
  "model_args": null,
  "seed": 42,
  "output_dir": "./logs",
  "model_arg": {}
}
```

```json
{
  "dataset": "cnn_dm_summarization",
  "data_path": null,
  "random_shuffle": true,
  "num_samples": 100,
  "n_shot": 0,
  "template": null
}
```

```json
{
  "max_steps": 512,
  "exit_layer": 8,
  "num_speculations": 6,
  "generation_strategy": "self_speculative",
  "sample": true,
  "temperature": 0.6,
  "top_k": 0,
  "top_p": 0.9,
  "no_repeat_ngram_size": null,
  "stop_words": null,
  "stop_token_ids": []
}
```

```json
{
  "predicted_text": {
    "rouge-l": 0.044874850660562515,
    "rouge-1": 0.060123004019260406,
    "rouge-2": 0.009091850370168686,
    "rouge-3": 0.0026279911398887634,
    "bleu_score": 0,
    "exact_match": 1585.1099853515625
  },
  "acceptance_rate": {
    "mean": 0.12566105492413043
  },
  "total_time": {
    "mean": 12.026322107315064
  },
  "time_per_token": {
    "mean": 0.024134453032165767
  },
  "tokens_per_second": {
    "mean": 43.77809970855713
  }
}
```
