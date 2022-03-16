from typing import List, Union
import torch
import clip
import PIL
from PIL import Image
import requests
import numpy as np
import os
from sentence_transformers import SentenceTransformer, util


class ZeroShotTextClassification():


  def __init__(self, 
               *args, 
               **kwargs):
    
         """
          Load either Open AI CLIP Text backbone or Sentence Transformer CLIP Text backbone 
          based on language param.
          
          Args:
              Lang (`str`, *optional*, defaults to `en`):
              en - for English
              multi - for Other Multi-lingual
         """
    
         if "lang" in kwargs:
            self.lang = kwargs["lang"]
         else:
            self.lang = "en"

         lang_codes = {"en", "multi"}

         if self.lang not in lang_codes:
            raise Exception('Language code {} not valid, supported codes are {} '.format(self.lang, lang_codes))
            return 

         device = "cuda:0" if torch.cuda.is_available() else "cpu" 

         if self.lang == "en":
            model_tag = "ViT-B/32"
            print("Loading OpenAI CLIP model {} ...".format(model_tag))    
            self.model, self.preprocess = clip.load(model_tag, device=device) 
            print("Label language {} ...".format(self.lang))
         else:          
            model_tag = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
            print("Loading sentence transformer model {} ...".format(model_tag))
            self.model = SentenceTransformer(model_tag, device=device)
            print("Label language {} ...".format(self.lang))

  def available_languages(self):
      """Returns the codes of available languages"""
      codes = """ar, bg, ca, cs, da, de, en, el, es, et, fa, fi, fr, fr-ca, gl, gu, he, hi, hr, hu, 
      hy, id, it, ja, ka, ko, ku, lt, lv, mk, mn, mr, ms, my, nb, nl, pl, pt, pt, pt-br, 
      ro, ru, sk, sl, sq, sr, sv, th, tr, uk, ur, vi, zh-cn, zh-tw"""
      return set([code.strip() for code in codes.split(",")])


  def __call__(
        self, 
        text: Union[str, List[str]],
        candidate_labels: Union[str, List[str]],
        *args,
        **kwargs,
    ):

        """
        Classify the text using the candidate labels given

        Args:
            text (`str` or `List[str]`):
                Text to be classified, Can be a string or list of strings
            candidate_labels (`str` or `List[str]`):
                The set of possible class labels to classify each sequence into. Can be a single label, a string of
                comma-separated labels, or a list of labels.
            hypothesis_template (`str`, *optional*):
                The template used to turn each label into a string. This template must include a {} or
                similar syntax for the candidate label to be inserted into the template. 

        Return:
            A `dict` or a list of `dict`: Each result comes as a dictionary with the following keys:
            - **text** (`str`) -- The text for which this is the output.
            - **labels** (`List[str]`) -- The labels sorted by order of likelihood.
            - **scores** (`List[float]`) -- The probabilities for each of the labels.
        """

        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        
        if "hypothesis_template" in kwargs:
            hypothesis_template = kwargs["hypothesis_template"] 
        else:
            hypothesis_template = "{}"

        if isinstance(candidate_labels, str):
          labels = [hypothesis_template.format(candidate_label) for candidate_label in candidate_labels.split(",")]
        else:    
          labels = [hypothesis_template.format(candidate_label) for candidate_label in candidate_labels]


        if str(type(self.model)) == "<class 'clip.model.CLIP'>":
            text_tokens = clip.tokenize(text).to(device)
            label_tokens = clip.tokenize(labels).to(device)
            text_features = self.model.encode_text(text_tokens)
            labels_features = self.model.encode_text(label_tokens)
        else:    
            text_features = torch.tensor(self.model.encode(text))
            labels_features = torch.tensor(self.model.encode(labels))
        
        sim_scores = util.cos_sim(text_features, labels_features)
        preds = []

        if isinstance(text, str):
            text = [text]

        for textlet, sim_score in zip(text, sim_scores):
            out = []
            pred = {}
            for raw_score in sim_score:
                out.append(raw_score.item() * 100)
            probs = torch.tensor([out])
            probs = probs.softmax(dim=-1).cpu().numpy()
            scores = list(probs.flatten())

            sorted_sl = sorted(zip(scores, candidate_labels), key=lambda t:t[0], reverse=True)  

            pred["text"] = textlet
            pred["scores"], pred["labels"] = zip(*sorted_sl)
            preds.append(pred)

        if len(preds) == 1:
          return preds[0]
        else:
          return preds
