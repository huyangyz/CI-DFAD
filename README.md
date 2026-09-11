<img width="1503" height="659" alt="image" src="https://github.com/user-attachments/assets/26e668c1-4b7a-4305-beed-be23e2e3cc60" /># CI-DFAD

Reference implementation of **Cross-Scale Interactive Dynamic Feature
Learning and Adversarial Detection for Spatiotemporal Traffic Anomalies**.

## Environment

- torch==1.13.1+cu116
- torchaudio==0.13.1+cu116
- torchvision==0.14.1+cu116
- numpy==1.24.4
- pandas==2.0.3
- matplotlib==3.7.5
- certifi==2024.8.30
- charset-normalizer==3.4.0
- contourpy==1.1.1
- cycler==0.12.1
- fonttools==4.54.1
- kiwisolver==1.4.7
- idna==3.10
- packaging==24.1
- pillow==10.4.0
- pyparsing==3.1.4
- python-dateutil==2.9.0.post0
- pytz==2024.2
- requests==2.32.3
- six==1.16.0
- typing_extensions==4.12.2
- urllib3==2.2.3
- zipp==3.20.2
- tzdata==2024.2
- seaborn==0.13.2
- importlib_resources==6.4.5

Install the dependencies with:

```bash
python -m pip install -r requirements.txt
```

## Project layout

```text
CI-DFAD/
  model.py               
  dataset.py             
  trainer.py
  evaluator.py
  drp/                   
  data/<dataset>-data/   
      data.npy
      node_dist.txt
      time_features.txt
  data/<dataset>/
    processed/
  outputs/<dataset>/
    checkpoints/
    results/
  main.py
```

The three large source datasets (NYC, Chicago, and PeMS) and the pretrained models can be downloaded from the following link: https://data.mendeley.com/preview/trr94mt9p5?a=0d676f5d-7c54-4576-9d7d-4813d74407c0

Generate the NYC DRP tensors from the project root with:

```bash
python -m drp.build --dataset nyc
```

The builder implements `W(t) = alpha * D + (1 - alpha) * F(t)`, excludes the
center node, and selects its eight strongest dynamically related neighbors.
It writes `drp_adjacency.npy`, `drp_node_indices.npy`, and `drp_config.json`
to `data/<dataset>/processed`. The default `alpha` is 0.5. To avoid target
leakage, prediction at time `t` uses the graph constructed at `t-1`.

## Train and evaluate

From this directory, run:

```bash
python main.py --dataset nyc
```

Its source data must be stored in `data/nyc`. The configuration uses 2 thirty-minute intervals per hour, 62 training days, a two-hour recent window, num_feature=4, input_dim=4, and 39 time features.


For Chicago, first build DRP and then train/evaluate:

```bash
python -m drp.build --dataset chicago
python main.py --dataset chicago
```

PeMS uses the same two-step workflow:

```bash
python -m drp.build --dataset pems
python main.py --dataset pems
```

The default generator-loss weights are `lambda_G=500`, `lambda_U=10`, and
`lambda_A=lambda_E=0.05`. The KL divergence is normalized by the number of training
samples, and the external-encoder regularization is averaged across its layers.

The script saves state-dict checkpoints as
`generator_epoch_NNN.pt` and `discriminator_epoch_NNN.pt`. Evaluation creates:

- `outputs/nyc/results/component_scores.npy`: raw components in the order
  `[reconstruction, D(real), D(generated), uncertainty]`.

For anomaly detection, we use beta=0.4 and gamma=0.3, and compute the area under the curve, recall, and hit rate for the top k items on a daily basis under different thresholds.

For more detailed hyperparameter configurations, please refer to Section 5.1.4, "Implementation Details," of the paper. To ensure the credibility of the experimental results, we conducted repeated experiments using random seeds 20, 21, and 22.
