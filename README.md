# Alt-ZSC - An Alternate Implementation for Zero Shot Text Classification

### What? 
* Intentionally super simple yet useful if you are $ concious.
* Instead of reframing NLI/XNLI, Alt-ZSC reframes the text backbone of CLIP models to do ZSC. Perks can be, lightweight + supports more languages without trading-off accuracy, especially if you are using ZSC in a Low code or Auto* libraries or simply looking to do weak labelling.
* Standing on the shoulder of gaints - OpenAI , Sentence-Transformers, HuggingFace, 


### Why Alt-ZSC can be attractive?

<img src="./images/ZSC vs Alt-ZSC.png" width="900">


### Installation
```python 
!pip install git+https://github.com/PrithivirajDamodaran/Alt-ZSC.git
```

### Usage

#### English

```python
zstc = ZeroShotTextClassification()

preds = zstc(text="Do dogs really make better pets than cats or hamsters?",
            candidate_labels=["kittens", "hamsters", "cats", "dogs"], 
            )
            
print(preds)

'''
prints the following
Loading OpenAI CLIP model ViT-B/32 ...
Label language en ...

{'text': 'Do dogs really make better pets than cats or hamsters?', 
'scores': (0.988218, 0.011007968, 0.0007573191, 1.6704575e-05), 
'labels': ('dogs', 'cats', 'hamsters', 'kittens')}
'''
```


