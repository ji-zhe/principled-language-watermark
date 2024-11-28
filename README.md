# A Principled Approach to Natural Language Watermarking #
- Code for the paper: [A Principled Approach to Natural Language Watermarking](https://dl.acm.org/doi/10.1145/3664647.3681544) 

Codes in this repo are partly adapted from the repo [AWT](https://github.com/S-Abdelnabi/awt), which is one of our baselines.


## Enviroment ##
- Main requirements:
	- Python 3.7.6
	- PyTorch 1.2.0
- To set it up: 
```sh
conda env create --name YOUR_ENV_NAME --file=environment.yml
```
- - -

## Requirements ##

- Model checkpt of InferSent:
	- get the model infersent2.pkl from: [InferSent](https://github.com/facebookresearch/InferSent), place it in 'sent_encoder' directory, or change the argument 'infersent_path' in 'main_train.py' accordingly
  
	- Download GloVe following the instructions in: [inferSent](https://github.com/facebookresearch/InferSent), place it in 'sent_encoder/GloVe' directory, or change the argument 'glove_path' in 'main_train.py' accordingly
  
- Model checkpt of SBERT:
	- Follow instructions from: [sentence-transformer](https://github.com/UKPLab/sentence-transformers)
- - -

## Dataset ##

- You will need the WikiText-2 (WT2) dataset. Follow the instructions in: [AWD-LSTM](https://github.com/salesforce/awd-lstm-lm) to download it

- - -

## Training ##
```javascript
python main_train_without_sideinfo.py --msg_len 4 --data data/wikitext-2 --batch_size 80  --epochs 200 --save CKPT_NAME --optimizer adam --bptt 80 --msg_weight 3 --gen_weight 2 --reconst_weight 2 
```


## Evaluating Robustness ##


### Adaptive Attacker ###

#### Training ####
The checkpoint of watermark encoder is needed.
```sh
python train_adaptive_atk_cgan.py 
```

#### Evaluate ####

```sh
python evaluate_denoise_autoenc_attack_greedy_revised.py --data data/wikitext-2 --bptt 80 --msg_len 4 --gen_path CKPT --disc_path CKPT --autoenc_attack_path atk_CKPT --use_lm_loss 1 
```

- - -
## Citation ##

- If you find this code helpful, please cite our paper:
```tex
@inproceedings{10.1145/3664647.3681544,
author = {Ji, Zhe and Hu, Qiansiqi and Zheng, Yicheng and Xiang, Liyao and Wang, Xinbing},
title = {A Principled Approach to Natural Language Watermarking},
year = {2024},
isbn = {9798400706868},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3664647.3681544},
doi = {10.1145/3664647.3681544},
booktitle = {Proceedings of the 32nd ACM International Conference on Multimedia},
pages = {2908–2916},
numpages = {9},
keywords = {natural language processing, watermarking},
location = {Melbourne VIC, Australia},
series = {MM '24}
}
```
- - -

## Acknowledgement ##

- We thank the authors of [InferSent](https://github.com/facebookresearch/InferSent), [sentence-transformer](https://github.com/UKPLab/sentence-transformers), and [AWT](https://github.com/S-Abdelnabi/awt) for their repositories and pre-trained models which we use in our training and experiments. 

- - -



