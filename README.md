# Weight & Biases demo with PyTorch & GAN

Generating MNIST handwritten digit images by GANs built on PyTorch and visualize/track it with [Weights & Biases](https://wandb.com).

Demo project can be found [here](https://app.wandb.ai/ikr7/wandb-pytorch-gan-mnist-demo?workspace=user-ikr7).

![](https://i.imgur.com/uQ3YkJ6.png)

## Getting Started

Clone this repo and install required packages. Use of [venv](https://docs.python.org/3/library/venv.html) is recommended for isolating the environment.

```shell
$ git clone https://github.com/ikr7/wandb-pytorch-gan-mnist-demo.git
$ cd wandb-pytorch-gan-mnist-demo
$ pip install -r requirements.txt
```

Create an account on [Weights & Biases](https://app.wandb.ai/login?signup=true) and login from your command line. API keys can be found on [User Settings](https://app.wandb.ai/settings).

```
$ wandb login <your API key>
```

Create a config file from `params.yaml.template` and edit it whatever you want.

```
$ cp params.yaml.template params.yaml
(edit params.yaml)
```

Start your training and presto!

```
$ python src/train.py --params params.yaml
```
