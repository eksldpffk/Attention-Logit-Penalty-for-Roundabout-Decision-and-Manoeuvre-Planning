# Attention-Logit-Penalty-for-Roundabout-Decision-and-Manoeuvre-Planning
A lightweight attention modification for roundabout scene understanding that suppresses less relevant interactions using learned pairwise penalties.

## Why this project?

At a roundabout, the ego vehicle must understand several things at once:
- whether to enter or wait
- when and how to enter
- which nearby agents actually matter
- how confident the model should be

Standard self-attention learns relationships mainly from feature similarity. The problem is that two agents can look similar in feature space while having very different importance for the current manoeuvre.

## Main idea

Instead of changing the whole Transformer, we add a small learned penalty directly to the attention scores.

For each token pair, a lightweight MLP looks at interaction features such as:
- relative position and motion
- time-to-contact (TTC)
- gap information
- scene-level manoeuvre constraints

The MLP outputs a non-negative penalty. Pairs that are less useful for a safe manoeuvre receive a larger penalty and therefore less attention.

## How it works

<p>
<img src="assets/DP_arch.png"  align="left" width="430">

The model uses an encoder-only Transformer to processes all tokens at once with:
  <ul>
    <li> ego token </li>
    <li> neighbour tokens </li>
    <li> map/context tokens </li>
    <li> CLS token for scene-level prediction </li>
  </ul>
The standard attention score is $S_{ij} = \frac{Q_i K_j^T}{\sqrt{d}}$ <br>
We modify it as **S'<sub>ij</sub> = S<sub>ij</sub> - $\lambda$ C<sub>ij</sub>**
where:
  <ul>
    <li> $C_{ij}$ is the learned penalty for token pair $(i,j)$ </li>
    <li> $\lambda$ controls how strongly the penalty affects attention </li>
  </ul>
</p>
    
A small penalty keeps the original attention almost unchanged. <br>
A large penalty suppresses that interaction before softmax.


## Outputs

The final scene representation is used for three tasks:
- **Decision**: Enter or Wait
- **Manoeuvre planning**: entry time, entry speed, heading
- **Uncertainty**: confidence / abstention signal

## Why use a penalty?

The goal is not to force the model to follow hand-written rules. The penalty network is still learned from data.
We only give the attention mechanism extra information about the **relationship between two agents and the current scene**, so it can reduce attention to interactions that are less relevant for the manoeuvre.

_This keeps the main Transformer architecture unchanged and adds only a small interaction-aware module._
