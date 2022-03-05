# An Alternate Implementation for Zero Shot Text Classification

### What is Alt-ZSC? 
* Intentionally super simple yet useful if you are $ concious.
* Instead of reframing NLI/XNLI, Alt-ZSC reframes the text backbone of CLIP models to do ZSC. Perks can be, lightweight + supports more languages without trading-off accuracy, especially if you are using ZSC in a Low code or Auto* libraries or simply looking to do weak labelling.
* In some cases CLIP based Alt-ZSC models gives better accuracy than NLI/XNLI based ZSC. (Not a benchmark but from casual tests)
* Standing on the shoulder of gaints - OpenAI CLIP, Sentence-Transformers, HuggingFace Transformers, 


### Why Alt-ZSC can be attractive?

<img src="./images/ZSC vs Alt-ZSC.png" width="900">


### Installation
```python 
!pip install git+https://github.com/PrithivirajDamodaran/Alt-ZSC.git
```

### Usage

#### English

```python
from AltZSC import ZeroShotTextClassification

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
'''

```

### You can use it with over 50 languages
```python
#View Supported lang codes
zstc.available_languages()

#Prints the following

{'ar',
 'bg',
 'ca',
 'cs',
 'da',
 'de',
 'el',
 'en',
 'es',
 'et',
 'fa',
 'fi',
 'fr',
 'fr-ca',
 'gl',
 'gu',
 'he',
 'hi',
 'hr',
 'hu',
 'hy',
 'id',
 'it',
 'ja',
 'ka',
 'ko',
 'ku',
 'lt',
 'lv',
 'mk',
 'mn',
 'mr',
 'ms',
 'my',
 'nb',
 'nl',
 'pl',
 'pt',
 'pt-br',
 'ro',
 'ru',
 'sk',
 'sl',
 'sq',
 'sr',
 'sv',
 'th',
 'tr',
 'uk',
 'ur',
 'vi',
 'zh-cn',
 'zh-tw'}
 ```

### FAQ

- Multi-lingual option covers english why use OpenAI CLIP for "en" ?
- [Ans] OpenCLIP uses a Custom base BERT-ish transformer with some ideas from GPT2 (as per the paper, see blow), but ```sentence-transformers/clip-ViT-B-32-multilingual-v1``` uses ```distilbert-base-multilingual-cased```. It means in practice OpenCLIP text backbone gives a better score than ```distilbert```. For instance, trying the same english example shows the difference in scores (check the scores and label order below). While nothing stops you from using english text/labels under ```multi```, Alt-ZSC uses OpenAI CLIP as default for the better scores it can give for english text.


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

- If Alt-ZSC is not based on NLI/XNLI (unlike ZSC) why does it support hypothesis templates?
- [Ans] Longer fluent sentences with context offer a nice prefix to the labels, it gives developers an option to squeeze better scores if needed (as per the CLIP paper, see below). So feel free to use when you see fit. 

```python
preds = zstc(text="Do dogs really make better pets than cats or hamsters?",
            candidate_labels=["kittens", "hamsters", "cats", "dogs"], 
            hypothesis_template = "This is {}"
            )
```            
            
>Another issue we encountered is that it’s relatively rare in
our pre-training dataset for the text paired with the image
to be just a single word. Usually the text is a full sentence
describing the image in some way. To help bridge this
distribution gap, we found that using the prompt template
“A photo of a {label}.” to be a good default that
helps specify the text is about the content of the image. This
often improves performance over the baseline of using only
the label text. For instance, just using this prompt improves
accuracy on ImageNet by 1.3%.            



