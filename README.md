# PyTorch Implementation of [Transformer Interpretability Beyond Attention Visualization](https://arxiv.org/abs/2012.09838) for ESM [CVPR 2021]

## ESM explainability notebook:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1orZJw51GuGVb95-_18PvVumJg1__seXz#scrollTo=6_yQDJg3r0XY)
---

## Methodology
The adaptation of the official implementation of 
[Transformer Interpretability Beyond Attention Visualization](https://arxiv.org/abs/2012.09838) to ESM consists of 3 
phases as also visualized below.

1. Calculating relevance for each attention matrix using the novel formulation of LRP.

2. Backpropagation of gradients for each attention matrix w.r.t. the visualized class. Gradients are used to average 
   attention heads.

3. Layer aggregation with rollout.

<p align="center">
  <img width="600" height="200" src="https://github.com/hila-chefer/Transformer-Explainability/blob/main/method-page-001.jpg">
</p>

In order to make it work for ESM, we had to change some BERT specifics to ESM configurations.

## Credits

BERT implementation is taken from the [huggingface Transformers library](https://huggingface.co/transformers/) and adapted to work with ESM.

Text visualizations in supplementary were created using [TAHV heatmap generator for text](https://github.com/jiesutd/Text-Attention-Heatmap-Visualization.

And of course the [underlying repo of this work](https://github.com/hila-chefer/Transformer-Explainability) putting everything together.
