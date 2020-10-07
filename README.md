# GIKT

The code is related to the paper `GIKT: A Graph-based Interaction Model for Knowledge Tracing` published at ECML-PKDD 2020[Paper in arXiv](https://arxiv.org/abs/2009.05991).


## Datasets
You can download the preprocessed dataset: [download](https://www.dropbox.com/sh/i317xy6ys6csnbw/AAAwGTvrb-mXtymuS6JgOoWpa?dl=0)

If you want to process the datasets by yourself, you can download them by the following links:

ASSIST09:[download](https://drive.google.com/file/d/1NNXHFRxcArrU0ZJSb9BIL56vmUt5FhlE/view)

ASSIST2012: [download](https://drive.google.com/file/d/0BxCxNjHXlkkHczVDT2kyaTQyZUk/edit?usp=sharing)

EdNet: [download](https://drive.google.com/file/d/1AmGcOs5U31wIIqvthn9ARqJMrMTFTcaw/view)


## Environment Requirement

python == 3.6.5

tensorflow == 1.12.0

numpy == 1.15.2

## Examples to run the model

### ASSIST09 dataset
* Command
```
python main.py --dataset assist09_3 --n_hop 3 --skill_neighbor_num 4 --question_neighbor_num 4 --hist_neighbor_num 3 --next_neighbor_num 4 --model hsei --lr 0.001 --att_bound 0.7 --sim_emb question_emb --dropout_keep_probs [0.8,0.8,1]
```
