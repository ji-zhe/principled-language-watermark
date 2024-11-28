# cals-part #
文件夹下的该部分代码文件需要配合ACL那篇论文对CALS的实现运行，需要覆盖其中的两个python文件。
https://github.com/bangawayoo/nlp-watermarking

- 运行origen.py
```javascript
python origen.py
```

- 运行context-ls-pro.py生成水印文本
```javascript
python -u context-ls-pro.py -embed T --num_sample 5000 --exp_name wt2raw0803 --spacy_model en_core_web_sm --dtype wikitext -metric_only F
```

- 运行context-ls-pro.py得到水印
```javascript
python context-ls-pro.py --num_sample 1000 --spacy_model en_core_web_sm --dtype wikitext2 --exp_name wt2 -extract T -extract_corrupted T --corrupted_file_dir <corrupted_file_dir>
```

# awt-part #
文件夹下的该部分代码文件需要配合AWT原版代码运行，只需要awt的环境中使用Pip安装transformers即可。
https://github.com/S-Abdelnabi/awt

- 运行data_bert:(以WT2数据集为例)
```javascript
python data_bert.py
```
- 运行main_train_dae_bertemb.py(以WT2raw数据集为例)
```javascript
nohup python main_train_dae_bertemb.py --bptt 80 --pos_drop 0.1 --optimizer adam --save WT2raw --emsize 768 --batch_size 64 --epochs 2000 --dropoute 0.05 --sub_prob 0.1 --log-interval 200 --saved_corpus tokenizer_WT2raw.data
```

- 运行cals_dae_zyc_pro.py
```javascript
python cals_dae_zyc_pro.py --model WT2raw_dae_bertemb.pt --watermarked <watermarked txt>  --origin <origin txt> --clean <only corrputed txt> --composite <corrupted txt and parameters>
```

-运行cals_compare.py 获得metric
```javascript
python cals_compare.py --ori <origin txt> --cor <only corrupted txt>
```