import logging
import re
import sys
from hashtagger_config import config
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem import PorterStemmer
from nltk import pos_tag, ne_chunk
from nltk.tree import Tree
from nltk import data as nltk_data
from sner import Ner
from stemming.porter2 import stem
from string import digits, punctuation
import unicodedata

_logger = logging.getLogger(__name__)

if config.NLTK_DATA_PATH is not None:
    nltk_data.path.append(config.NLTK_DATA_PATH)
_stanford_ner_tagger = Ner(host=config.STANFORD_NER_TAGGER_SERVER['host'], port=config.STANFORD_NER_TAGGER_SERVER['port'])

_set_of_stopwords = set(nltk_stopwords.words("english"))
# the following is very useful, but shouldn't be done for the purpose of comparing to SOTA with a standard preprocessing
# for sw in ["BBC", "TheJournal", "ie", "Al", "Jazeera", "News"]:
#     set_of_stopwords.add(sw)


def ner_tokenize(text):
    stemmer = PorterStemmer()

    # unicode categories can be found here http://www.unicode.org/reports/tr44/#General_Category_Values
    chunked = ne_chunk(pos_tag(word_tokenize(
        "".join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    )))
    nes = []
    nouns = []
    all_tokens = []

    for t in chunked:
        if type(t) == Tree:
            ne = " ".join([token for token, pos in t.leaves()])
            nes.append(ne)
            all_tokens.append(ne)
        elif t[1][:2] == "NN" and t[0] not in _set_of_stopwords:
            """ if necessary add non-ascii character removal, like single quotes """
            nn = stemmer.stem(t[0].lower().strip().translate({ord(k): None for k in digits + punctuation}))
            nouns.append(nn)
            all_tokens.append(nn)
        else:
            continue

    unique_tokens = list(set(all_tokens))
    processed_text = " ".join(all_tokens)

    return nes, nouns, all_tokens, unique_tokens, processed_text


def get_ne_continuous(text):
    """
    based on https://stackoverflow.com/a/31838373/2262424
    make sure that the text is not lowercased
    :param text:
    :return:
    """
    chunked = ne_chunk(pos_tag(word_tokenize(text)))
    # prev = None
    continuous_chunk = []
    current_chunk = []

    for i in chunked:
        if type(i) == Tree:
            current_chunk.append(" ".join([token for token, pos in i.leaves()]))
        elif current_chunk:
            named_entity = " ".join(current_chunk)
            if named_entity not in continuous_chunk:
                continuous_chunk.append(named_entity)
                current_chunk = []
        else:
            continue

    if continuous_chunk:
        named_entity = " ".join(current_chunk)
        if named_entity not in continuous_chunk:
            continuous_chunk.append(named_entity)

    continuous_chunk = [ne for ne in continuous_chunk if len(ne) > 1]

    return continuous_chunk


def stanford_ner_tag(text):
    """ it is much faster with a local NER server as described here https://stackoverflow.com/a/43165776/2262424 """
    # the sentences shorter than 6 characters are ignored
    all_tags = (
        (continuous_tokens.lower(), label) for sentence in sent_tokenize(text)
        for (continuous_tokens, label) in
        get_continuous_ne_chunks_from_stanford_tags(_stanford_ner_tagger.get_entities(sentence))
        if len(sentence) > 5
    )
    return list(all_tags)


def get_continuous_ne_chunks_from_stanford_tags(tags):
    """
    based on https://stackoverflow.com/a/31838373/2262424
    make sure that the text is not lowercased
    expects as an input the an output of StanfordNERTagger().tag(StanfordTokenizer().tokenize(text))
    """
    continuous_chunks = []
    current_chunk = []

    for token, label in tags:
        if label != "O":
            # when it's a proper noun
            current_chunk.append((token, label))
        elif current_chunk:
            # when the proper noun ended
            named_entity = " ".join([t for (t, l) in current_chunk])
            if named_entity not in continuous_chunks:
                continuous_chunks.append((named_entity, current_chunk[0][1]))
                current_chunk = []
        else:
            # when there was no proper noun in the current chunk
            continue

    if current_chunk:
        # when the proper noun is in the very end of the tags
        named_entity = " ".join([t for (t, l) in current_chunk])
        if named_entity not in continuous_chunks:
            continuous_chunks.append((named_entity, current_chunk[0][1]))

    continuous_chunks = [ne for ne in continuous_chunks if len(ne) > 1]

    return continuous_chunks


def is_ascii(s):
    """
    Returns True if a string is ASCII, False otherwise.

    based on https://stackoverflow.com/a/196391/2262424
    """
    try:
        s.encode('ascii')
        return True
    except UnicodeEncodeError:
        # _logger.debug("it was not a ascii-encoded unicode string")
        return False
    else:
        # _logger.debug("It may have been an ascii-encoded unicode string")
        return False


def deaccent_and_replace_quotes(text):
    # unicode categories can be found here http://www.unicode.org/reports/tr44/#General_Category_Values
    # an alternative implementation in https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/utils.py
    normalized_text = unicodedata.normalize(
        'NFD', re.sub(u'[\u201c\u201d]', '"', re.sub(u'[\u2018\u2019]', "'", text))
    ).encode('ascii', 'ignore').decode(sys.stdout.encoding).strip()
    return normalized_text


def remove_stopwords_non_alpha_and_lemmatize(text, lemmatizer, stopwords=None):
    if stopwords is None:
        stopwords = _set_of_stopwords
    processed_text = " ".join(
        [stem(lemmatizer.lemmatize(w)) for w in
         word_tokenize(re.sub("[:;>?<=*+()./,\-#!&\"$%\{˜|\}'\[ˆ_\\@\]1234567890’‘]", " ", text.lower()))
         if w not in stopwords and len(w) >= 2]
    ).strip()
    return processed_text


def process_tweet_content(text, stopwords=None, min_word_length=2):
    """Removes urls, mentions, hashtags, non-alphanumeric characters and numbers
    from ASCII-normalized casefolded text. Removes stopwords and words with length < min_word_length characters"""
    if stopwords is None:
        stopwords = _set_of_stopwords
    clean_text = " ".join(w for w in word_tokenize(re.sub(
        '((www\.[\s]+)|(https?://[^\s]+)|(@[^\s]+)|(#[^\s]+)|([:;>?<=*+()./,\-#!&"$%\{˜|\}\'\[ˆ_\\@\]’‘])|([\d]))',
        '',
        unicodedata.normalize('NFD', text.casefold()).encode('ascii', 'ignore').decode(sys.stdout.encoding).strip()
    ))
                          if w not in stopwords and len(w) >= min_word_length
                          )
    return clean_text


########################################################################################################################
#  -------------------------------------------------- OLD BAD CODE --------------------------------------------------- #
########################################################################################################################
# _sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
# _stanford_ner_tagger = nltk.tag.StanfordNERTagger('english.all.3class.distsim.crf.ser.gz')
# _tokenizer = StanfordTokenizer()
# def stanford_ner_tag_old(text):
#     """ THIS FUNCTION IS VERY EXPENSIVE """
#     sentences = [s for s in _sentence_tokenizer.tokenize(text) if len(s) > 5]
#
#     all_tags = []
#     for sentence in sentences:
#         tags = _stanford_ner_tagger.tag(_tokenizer.tokenize(sentence.replace("‘", "")))
#         continuous_tags = get_continuous_ne_chunks_from_stanford_tags(tags)
#         all_tags.extend(continuous_tags)
#     return [(token.lower(), label) for (token, label) in all_tags]
#
#
# def stanford_ner_tag_faster_than_old(text):
#     """ THIS FUNCTION IS STILL VERY EXPENSIVE
#     a speedup is expected with tag_sent() as described here https://stackoverflow.com/a/33749127/2262424 """
#     tokenized_sentences = [word_tokenize(sent) for sent in sent_tokenize(text) if len(sent) > 5]
#     # the sentences shorter than 6 characters are ignored
#     # _logger.debug("tokenized text into %d sentences" % len(tokenized_sentences))
#
#     tagged_sentences = _stanford_ner_tagger.tag_sents(tokenized_sentences)
#     # _logger.debug("tagged all %d sentences" % len(tokenized_sentences))
#     all_tags = (
#         (continuous_tokens.lower(), label) for tags in tagged_sentences
#         for (continuous_tokens, label) in get_continuous_ne_chunks_from_stanford_tags(tags)
#     )
#     return list(all_tags)
# ------------------------------------------------ END OLD BAD CODE -------------------------------------------------- #
