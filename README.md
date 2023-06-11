# Code for "Like hiking? You probably enjoy nature: Persona-grounded Dialog with Commonsense Expansions"

[Like hiking? You probably enjoy nature: Persona-grounded Dialog with Commonsense Expansions](https://www.aclweb.org/anthology/2020.emnlp-main.739.pdf)

Bodhisattwa Prasad Majumder, Harsh Jhamtani, Taylor Berg-Kirkpatrick, Julian McAuley

Published at EMNLP 2020.

![compac](https://github.com/majumderb/compac/blob/master/image/compac.png?raw=true)

# Data

Huggingface cleaned dataset

`wget https://s3.amazonaws.com/datasets.huggingface.co/personachat/personachat_self_original.json`

(Optional) Personachat Download:
http://parl.ai/downloads/personachat/personachat.tgz

## COMET-expanded PersonaChat

You can download PersonaChat dataset with added COMET expansions from [here](https://drive.google.com/file/d/1tJih0IecAmP3IlP6TYvDjy3kOpIbMUIH/view?usp=sharing). 

# Training

The training script is [here](https://github.com/majumderb/compac/blob/master/models/reinforce_model/train.py). Please download the data (mainly COMET expansions) before training. The training may require 4 2080Tis or higher.



# FiJhGa-compac-ChatGPT
