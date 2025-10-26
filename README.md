# effisegnet_torch

Basic adaptation of the EffiSegNet for binary classification.
## Paper
[![arXiv](https://img.shields.io/badge/arXiv-2407.16298-b31b1b.svg)](https://arxiv.org/abs/2407.16298)

In order to adapt the model for binary classification a prediction head has been added at the end of the model combined with a sigmoid activation.


# Usage

!git clone ##repository

from model.effisegnet import EffiSegNetBN

instantiate the model and hf!