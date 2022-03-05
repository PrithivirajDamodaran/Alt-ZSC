# Alt-ZSC
Alternate Implementation for Zero Shot Text Classification: 


* Intentionally super simple yet useful.
* Why Alt-ZSC can be attractive for some users?

ZSC vs Alt-ZSC

Size

English  - facebook/bart-large-mnli - 1.5G
Mulitlingual - joeddav/xlm-roberta-large-xnli - 2G

English - Custom base size BERT-ish transformer with some ideas from GPT2 (as per the paper) - <= 500M
Mulitlingual - sentence-transformers/clip-ViT-B-32-multilingual-v1 (Internally uses distilbert-base-multilingual-cased) - 500M


### Installation
```python 
!pip install git+https://github.com/PrithivirajDamodaran/Alt-ZSC.git
```

### Usage

