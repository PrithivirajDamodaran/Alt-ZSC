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

#### Spanish

```python
zstc = ZeroShotTextClassification(lang="multi")

preds = zstc(text="¿Los perros realmente son mejores mascotas que los gatos o los hámsters?",
            candidate_labels=["gatita", "perros", "gatas","leonas"],
            )
            
print(preds)


'''
prints the following
Loading sentence transformer model sentence-transformers/clip-ViT-B-32-multilingual-v1 ...
Label language multi ...

{'text': '¿Los perros realmente son mejores mascotas que los gatos o los hámsters?', 
'scores': (0.99424934, 0.002653174, 0.002569642, 0.0005278819),
'labels': ('perros', 'gatas', 'gatita', 'leonas')}

```

### FAQ

- Multi-lingual option covers english why use OpenAI CLIP for "en" ?
OpenCLIP uses a Custom base BERT-ish transformer with some ideas from GPT2 (as per the paper, see blow), but ```sentence-transformers/clip-ViT-B-32-multilingual-v1``` uses ```distilbert-base-multilingual-cased```. It means in practice OpenCLIP text backbone gives a better score than ```distilbert```. For instance, trying the same english example shows the difference in scores (check the scores and label order below). While nothing stops you from using english text/labels under ```multi```, Alt-ZSC uses OpenAI CLIP as default for the better scores it can give for english text.


>The text encoder is a Transformer (Vaswani et al., 2017)
with the architecture modifications described in Radford
et al. (2019). As a base size we use a 63M-parameter 12-
layer 512-wide model with 8 attention heads. The transformer operates on a lower-cased byte pair encoding (BPE)
representation of the text with a 49,152 vocab size (Sennrich et al., 2015). For computational efficiency, the max
sequence length was capped at 76. The text sequence is
bracketed with [SOS] and [EOS] tokens and the activations of the highest layer of the transformer at the [EOS]
token are treated as the feature representation of the text
which is layer normalized and then linearly projected into
the multi-modal embedding space. Masked self-attention
was used in the text encoder to preserve the ability to initialize with a pre-trained language model or add language
modeling as an auxiliary objective, though exploration of
this is left as future work.

```python
zstc = ZeroShotTextClassification(lang="multi")

preds = zstc(text="Do dogs really make better pets than cats or hamsters?",
            candidate_labels=["kittens", "hamsters", "cats", "dogs"], 
            )
            
print(preds)


'''
prints the following
Loading sentence transformer model sentence-transformers/clip-ViT-B-32-multilingual-v1 ...
Label language multi ...

{'text': 'Do dogs really make better pets than cats or hamsters?', 
'scores': (0.93635553, 0.06061751, 0.0016885924, 0.0013383164), 
'labels': ('dogs', 'cats', 'kittens', 'hamsters')}
'''
```


