import re
import string
import numpy as np
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob.wordnet import NOUN, VERB, ADJ, ADV

_WORD_PAT = r"\w[\w']{3,}"
_SENT_PAT = r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s"
_NW_LINE = r"\\n+"
_WHITE_SPACE = r"\\t|\\r|\\v|\\f"
_SUBBED = {
    r"i'm": "i am",
    r"he's": "he is",
    r"she's": "she is",
    r"that's": "that is",
    r"what's": "what is",
    r"where's": "where is",
    r"\'ll": " will",
    r"\'ve": " have",
    r"\'re": " are",
    r"won't": " will not",
    r"can't": "cannot",
    r"don't": " do not",
    _NW_LINE: ". ",
    r"(?<=[0-9])\,(?=[0-9])": "",
    r"\$": " dollar",
    r"\%": "percent",
    r"\&": "and",
    _WHITE_SPACE: " "
}


class Normalizer:

    def __init__(self, text):
        self.stop_words = set(stopwords.words("english"))
        self.punctuations = set(string.punctuation)
        self.pos_tags = {
            NOUN: ['NN', 'NNS', 'NNP', 'NNPS', 'PRP', 'PRP$', 'WP', 'WP$'],
            VERB: ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
            ADJ: ['JJ', 'JJR', 'JJS'],
            ADV: ['RB', 'RBR', 'RBS', 'WRB']
        }

        self.text = text

    def _remove_stop_words(self, words):
        return [w for w in words if w not in self.stop_words]

    def _remove_regex(self):
        self.text = self.text.lower()
        for pat in _SUBBED:
            self.text = re.sub(pat, _SUBBED[pat], self.text)

        patterns = re.finditer("#[\w]*", self.text)
        for pattern in patterns:
            self.text = re.sub(pattern.group().strip(), "", self.text)

        self.text = "".join(ch for ch in self.text if ch not in self.punctuations)

    def _tokenize(self):
        return re.findall(_WORD_PAT, self.text)

    def _process_content_for_pos(self, words):
        tagged_words = pos_tag(words)
        pos_words = []
        for word in tagged_words:
            flag = False
            for key, value in self.pos_tags.items():
                if word[1] in value:
                    pos_words.append((word[0], key))
                    flag = True
                    break
            if not flag:
                pos_words.append((word[0], NOUN))
        return pos_words

    def _remove_noise(self):
        self._remove_regex()
        words = self._tokenize()
        noise_free_words = self._remove_stop_words(words)
        return noise_free_words

    def _normalize_text(self, words):
        lem = WordNetLemmatizer()
        pos_words = self._process_content_for_pos(words)
        normalized_words = [lem.lemmatize(w, pos=p) for w, p in pos_words]
        return normalized_words

    def sent_tokenize(self):
        return re.split(_SENT_PAT, self.text)

    def clean_up(self):
        cleaned_words = self._remove_noise()
        cleaned_words = self._normalize_text(cleaned_words)
        return cleaned_words


def _get_embeddings():
    word_embeddings = dict()
    with open("glove.6B.100d", encoding='utf-8') as file:
        for line in file:
            data = line.split()
            word = data[0]
            embeddings = np.array(data[1], dtype='float32')
            word_embeddings[word] = embeddings
    return word_embeddings


def _to_text(it):
    return " ".join(it)
