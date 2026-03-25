# wrld : simple toy "world" model  
I wanted to see if i could simulate a spring mass damper using a simple world model, end to end, and as simple as possible just to understand what was going on.

## to run
1. you will need `uv`, found [here](https://docs.astral.sh/uv/getting-started/installation/)
2. this repo setup assumes you have cuda 12.6 installed, if not it will fallback to cpu.  you can always modify the settings in `pyproject.toml` to your current cuda version.
3. `uv sync`
4. `uv run scripts/train.py`

it'll generate all the worst case loss and reconstruction curves, along with the loss curves.

## plots
### architecture
so heres a quick overview of the system architecture, i know its a little bit of a mess. courtesy of claude, i am no graphic designer.
![](./figs/architecture.png)

it boils down to an encoder -> latent dynamics model -> deccoder, but if theoretically your rollouts can live in the latent space as you add your action vector directly to the latent vector, thus removing the lossy encoder + decoder step.

### loss
![](./figs/loss_curves.png)
not much to be said, but interestingly if you remove the velocity from the observed state it still converges to a simiarly low loss.
definitions:
reconstruction loss -> covers the encoder / decoder effectiveness at reconstructing the original state, no transformation applied.
dynamics loss -> covers how good (or bad) the latent dynamics model is 
KL loss -> tries to keep the latent representation dense and meaningful, rather than overfitting.

### worst case reconstructions
![](./figs/reconstruction.png)
not much to see here, the reconstruction was nailing it

### worst case rollouts
![](./figs/rollout.png)
performance seems to degrade as you hit timestep 30 in worst cases, and model seems to do poorly for transient loading / huge changes in acceleration, as seen with seq 46 and 153. might just be issues of out of domain data, but we shall see


if you have any reccomendations or notice any big errors, please feel free to open a PR :)