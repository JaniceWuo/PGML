# PGML
code for paper

## Requirements
1. tensorflow-gpu == 1.15
2. Python3.6 cuda 10.1
3. stanfordcorenlp-3.9.1.1

## Dataset
1. You should download the PERSPECTRUM dataset proposed in [paper](https://www.aclweb.org/anthology/N19-1053), and execute the command to generate train,valid and test set under the `data` folder:
```cmd
python preprocessing.py
```
2. [Optional] Download the Wikitext-103 from [here](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/)<br/>
3. Download the glove word embedding from [here](https://nlp.stanford.edu/projects/glove/)

## Train
If you don't pretrain the model with wikitext:
```cmd
nohup python -u main.py --mode=train --model=embmin --use_pretrain=False > nohup.out 2>&1 &
``` 
else:
```cmd
nohup python -u main.py --mode=lm_train --model=lm --use_pretrain=True > nohup_pre.out 2>&1 &
```