# factcheck.py

import torch
from typing import List
import numpy as np
import spacy
import gc
import re
import json
# import nltk
# nltk.download('punkt')
# nltk.download('stopwords')

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.metrics.scores import recall
STOP_WORDS = set(stopwords.words('english'))


def get_word_set(text):
    no_punc = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(no_punc)
    filtered = [word for word in tokens if word.lower() not in STOP_WORDS]
    fact_set = set(filtered)
    return fact_set


class FactExample:
    """
    :param fact: A string representing the fact to make a prediction on
    :param passages: List[dict], where each dict has keys "title" and "text". "title" denotes the title of the
    Wikipedia page it was taken from; you generally don't need to use this. "text" is a chunk of text, which may or
    may not align with sensible paragraph or sentence boundaries
    :param label: S, NS, or IR for Supported, Not Supported, or Irrelevant. Note that we will ignore the Irrelevant
    label for prediction, so your model should just predict S or NS, but we leave it here so you can look at the
    raw data.
    """
    def __init__(self, fact: str, passages: List[dict], label: str):
        self.fact = fact
        self.passages = passages
        self.label = label

    def __repr__(self):
        return repr("fact=" + repr(self.fact) + "; label=" + repr(self.label) + "; passages=" + repr(self.passages))


class EntailmentModel:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.labels = ["S", "NS", "NS"]

    def check_entailment(self, premise: str, hypothesis: str):
        with torch.no_grad():
            # Tokenize the premise and hypothesis
            inputs = self.tokenizer(premise, hypothesis, return_tensors='pt', truncation=True, padding=True)
            # Get the model's prediction
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Note that the labels are ["entailment", "neutral", "contradiction"]. There are a number of ways to map
        # these logits or probabilities to classification decisions; you'll have to decide how you want to do this.

        result = self.labels[torch.argmax(logits, dim=1).int()]

        # To prevent out-of-memory (OOM) issues during autograding, we explicitly delete
        # objects inputs, outputs, logits, and any results that are no longer needed after the computation.
        del inputs, outputs, logits
        gc.collect()

        return result


class FactChecker(object):
    """
    Fact checker base type
    """

    def predict(self, fact: str, passages: List[dict]) -> str:
        """
        Makes a prediction on the given sentence
        :param fact: same as FactExample
        :param passages: same as FactExample
        :return: "S" (supported) or "NS" (not supported)
        """
        raise Exception("Don't call me, call my subclasses")


class RandomGuessFactChecker(object):
    def predict(self, fact: str, passages: List[dict]) -> str:
        prediction = np.random.choice(["S", "NS"])
        return prediction


class AlwaysEntailedFactChecker(object):
    def predict(self, fact: str, passages: List[dict]) -> str:
        return "S"


class WordRecallThresholdFactChecker(object):
    def predict(self, fact: str, passages: List[dict]) -> str:
        fact = re.sub(r'[^\w\s]', '', fact)
        fact_tokens = word_tokenize(fact)
        fact_tokens_filtered = [word for word in fact_tokens if word.lower() not in STOP_WORDS]
        full_text = ''
        for passage in passages:
            title = passage['title']
            text = passage['text'].replace("<s>"+title+" ", "")
            full_text += text
        full_text = full_text.replace("<s>", " ")
        full_text = full_text.replace("</s>", " ")
        full_text = ' '.join(full_text.split())
        full_text = re.sub(r'[^\w\s]', '', full_text)
        passage_tokens = word_tokenize(full_text)
        passage_tokens_filtered = [word for word in passage_tokens if word.lower() not in STOP_WORDS]
        score = recall(set(fact_tokens_filtered), set(passage_tokens_filtered))
        if score >= 0.65:
            return "S"
        else:
            return "NS"


class EntailmentFactChecker(object):
    def __init__(self, ent_model):
        self.ent_model = ent_model
        with open('false_neg.json', 'r') as file:
            self.false_neg = json.load(file)
        with open('false_pos.json', 'r') as file:
            self.false_pos = json.load(file)

    def predict(self, fact: str, passages: List[dict]) -> str:
        if fact in self.false_pos:
            pass
        if fact in self.false_neg:
            pass
        fact_set = get_word_set(fact)
        full_text = ''
        for passage in passages:
            title = passage['title']
            text = passage['text'].replace("<s>" + title + " ", "")
            full_text += text
        full_text = full_text.replace("<s>", " ")
        full_text = full_text.replace("</s>", " ")
        full_text = ' '.join(full_text.split())
        passage_sentences = sent_tokenize(full_text)

        for sent in passage_sentences:
            sent_set = get_word_set(sent)
            overlap_score = recall(set(fact_set), set(sent_set))
            if overlap_score <= 0.10:
                continue

            result = self.ent_model.check_entailment(premise=sent, hypothesis=fact)
            if result == "NS":
                continue

            return "S"
        return "NS"



# OPTIONAL
class DependencyRecallThresholdFactChecker(object):
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

    def predict(self, fact: str, passages: List[dict]) -> str:
        raise Exception("Implement me")

    def get_dependencies(self, sent: str):
        """
        Returns a set of relevant dependencies from sent
        :param sent: The sentence to extract dependencies from
        :param nlp: The spaCy model to run
        :return: A set of dependency relations as tuples (head, label, child) where the head and child are lemmatized
        if they are verbs. This is filtered from the entire set of dependencies to reflect ones that are most
        semantically meaningful for this kind of fact-checking
        """
        # Runs the spaCy tagger
        processed_sent = self.nlp(sent)
        relations = set()
        for token in processed_sent:
            ignore_dep = ['punct', 'ROOT', 'root', 'det', 'case', 'aux', 'auxpass', 'dep', 'cop', 'mark']
            if token.is_punct or token.dep_ in ignore_dep:
                continue
            # Simplify the relation to its basic form (root verb form for verbs)
            head = token.head.lemma_ if token.head.pos_ == 'VERB' else token.head.text
            dependent = token.lemma_ if token.pos_ == 'VERB' else token.text
            relation = (head, token.dep_, dependent)
            relations.add(relation)
        return relations

