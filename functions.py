import logging
import csv
import numpy as np
import nltk
import itertools
import re
import json
from datetime import timedelta, datetime, timezone
from nltk.tokenize import sent_tokenize, word_tokenize
from stemming.porter2 import stem
from collections import Counter, defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from itertools import chain, zip_longest

from hashtagger_config import config
from common.text_functions import is_ascii, stanford_ner_tag, ner_tokenize, remove_stopwords_non_alpha_and_lemmatize
from common.functions import ordered_sample_without_replacement, dict_cosine_similarity, json_serial, \
    find_curve_elbow_idx_based_on_max_dist
from es_article import search_es as search_es_articles
from es_article import get_by_id as get_articles_by_id
from es_article import update_esindex as update_article_es_index
from es_article import if_not_exist as if_not_exist_article_index
from es_tweet import search_es as search_es_tweets
from es_tweet import get_by_id as get_tweets_by_id

_logger = logging.getLogger(__name__)
_file_logger = logging.getLogger("features")
_stopwords = set(nltk.corpus.stopwords.words('english') + config.ADDITIONAL_STOPWORDS)
_noun_phrase_regex_parser = nltk.RegexpParser("NP: {<NNP><NNP>(<NNP>*)|<NNP><CC><NNP>}")
_lemmatizer = nltk.WordNetLemmatizer()
_tfidf_vectorizer = TfidfVectorizer(
    max_df=config.TFIDF_VECTORIZER_MAX_DF, min_df=config.TFIDF_VECTORIZER_MIN_DF, stop_words='english', use_idf=True
)
_coldstart_knn_classifier = KNeighborsClassifier(n_neighbors=config.COLDSTART_KNN_N_TRAINING_NEIGHBOURS)


########################################################################################################################
# ------------------------------------------------- THE MAIN FUNCTION ------------------------------------------------ #
########################################################################################################################
def load_articles_from_json_lines(articles_json_path, article_epoch_accessor, article_constructor_function,
                                  id_field_name="id", id_list=None, overwrite_existing_articles=False,
                                  tag_articles_flag=True, export_file_name=None, export_es_instance=None,
                                  local_tweet_window_before=24 * 3600, local_tweet_window_after=24 * 3600,
                                  global_tweet_window_before=24 * 3600, global_tweet_window_after=24 * 3600,
                                  global_article_window_before=60 * 24 * 3600, global_article_window_after=24 * 3600,
                                  current_global_stats_time_window_margin=3600):
    start_time = datetime.utcnow()
    if export_file_name is not None:
        # create an empty file
        open(export_file_name, "w").close()
        export_file = open(export_file_name, "a")
        _logger.info(
            "the articles will be %swritten to '%s'" % ("tagged and " if tag_articles_flag else "", export_file_name)
        )
    if export_es_instance is not None:
        # create the article index if it doesn't exist yet
        if_not_exist_article_index(es_instance=export_es_instance)
        _logger.info(
            "the articles will be %sindexed in '%s'" % ("tagged and " if tag_articles_flag else "", export_es_instance)
        )

    article_es_docs = []
    with open(articles_json_path) as articles_json_file:
        imported_articles_counter = 0
        # there are duplicate articles in WaPost corpus that have different IDs and slightly different urls
        # the duplicates may have e.g. 'feed/' or '87654567/' appended to the base url
        # there are also minor difference in the content xml tags, but usually not in the content text itself
        dict_of_unique_timestamp_title_pairs = dict()
        duplicate_article_counter = 0
        for a_raw in articles_json_file.readlines():  # ADD SMTH THAT WILL ORDER THE ARTICLES IN TIME FOR THE GLOBAL W!
            a = json.loads(a_raw)

            if id_list:
                if a[id_field_name] not in id_list:
                    continue
            try:
                if not article_epoch_accessor(a):
                    _logger.debug("skipping %s because its timestamp field is malformed" % a[id_field_name])
                    continue
            except Exception as e:
                _logger.debug("skipping %s because it is malformed" % a[id_field_name])
                continue

            update_global_window_stats_and_data_flag = False
            if imported_articles_counter != 0 and \
                    abs(article_epoch_accessor(a) - global_window_reference_t) > current_global_stats_time_window_margin:
                # check if the stats window is within current_global_stats_time_window_margin
                # from the optimal window for the article
                # Doing the check by the article timestamps is cleaner,
                # but checking the window's offset would have been more logical and easier to adapt changes to.
                update_global_window_stats_and_data_flag = True

            if update_global_window_stats_and_data_flag or imported_articles_counter == 0:  # also if it's the first doc
                # the offset from the optimal window is too big, update the window and re-estimate the stats -----------
                _logger.debug("updating the global time window stats and data")
                n_articles_in_global_window, global_window_word_df_dict = get_global_window_article_stats(
                    doc_unix_timestamp=article_epoch_accessor(a),  # converting milliseconds to seconds
                    global_article_window_before=global_article_window_before,
                    global_article_window_after=global_article_window_after
                )
                if tag_articles_flag:  # tweet stats are used only in the tagging process
                    hashtag_global_freq_table, max_global_freq, global_window_hashtag_tweets_dict, \
                        coldstart_article_tweet_ids = get_global_window_tweet_stats(
                            doc_unix_timestamp=article_epoch_accessor(a),  # converting milliseconds to seconds
                            global_tweet_window_before=global_tweet_window_before,
                            global_tweet_window_after=global_tweet_window_after,
                            tweet_sample_size=config.GLOBAL_TWEET_SAMPLE_SIZE,
                            coldstart_flag=config.COLDSTART_FLAG,
                            coldstart_article_window_before=config.COLDSTART_ARTICLE_WINDOW_BEFORE,
                            coldstart_article_window_after=config.COLDSTART_ARTICLE_WINDOW_AFTER
                        )
                global_window_reference_t = article_epoch_accessor(a)  # converting milliseconds to seconds

            # check if the article is already in the ES index otherwise create it --------------------------------------
            _logger.info("-" * 40 + "\n")
            _logger.debug("loading '%s'" % a[id_field_name])
            if overwrite_existing_articles:  # don't even check if an article exists or no
                matching_articles_in_es_count = 0
            else:
                matching_articles_in_es, matching_articles_in_es_count = get_articles_by_id([a[id_field_name]])
            if matching_articles_in_es_count == 1:
                article_es_doc = matching_articles_in_es[0]['_source']
                _logger.debug("article %s is already indexed, retrieved it from ES" % article_es_doc['id'])
                # convert the fields to the expected types
                article_es_doc['datetime'] = datetime.fromisoformat(article_es_doc['datetime'])  # python 3.7 only !
                # article_es_doc['datetime'] = datetime.strptime(article_es_doc['datetime'], "%Y-%m-%dT%H:%M:%S%z")
            else:
                if overwrite_existing_articles:
                    _logger.debug("creating article %s ... will overwrite if it's already indexed" % a[id_field_name])
                else:
                    _logger.debug("article %s is not found in the index, creating..." % a[id_field_name])
                # obtain a dictionary for the article
                article_es_doc = article_constructor_function(a=a)
            # check if a duplicate of this article has been already written or not
            if (article_es_doc['datetime'], article_es_doc['headline']) in dict_of_unique_timestamp_title_pairs:
                _logger.warning(
                    "article %s has the same title and timestamp as %s... skipping this article" %
                    (
                        article_es_doc['id'],
                        dict_of_unique_timestamp_title_pairs[(article_es_doc['datetime'], article_es_doc['headline'])]
                    )
                )
                dict_of_unique_timestamp_title_pairs[(article_es_doc['datetime'], article_es_doc['headline'])].append(
                    article_es_doc['id']
                )
                duplicate_article_counter += 1
                continue
            else:
                dict_of_unique_timestamp_title_pairs[(article_es_doc['datetime'], article_es_doc['headline'])] = [
                    article_es_doc['id']
                ]
            # create article profile -----------------------------------------------------------------------------------
            profile_keywords = [word for word in article_es_doc['stemming_title'].strip().split(" ") if word != " "]
            if len(profile_keywords) > 0 and len(article_es_doc['stemming_content']) > 0:
                article_es_doc['profile'] = get_article_profile(
                    keywords=profile_keywords,
                    article_content=article_es_doc['stemming_content'],
                    n_articles_in_global_window=n_articles_in_global_window,
                    global_window_word_df_dict=global_window_word_df_dict
                )
            else:
                article_es_doc['profile'] = {}
            _logger.debug("created the article profile")
            # recompute the 'stream_keywords' based on the profile -----------------------------------------------------
            # TODO: recompute the 'stream_keywords' based on the profile
            """ THIS WAS NOT DONE IN THE LEGACY CODE AND INSTEAD THERE WAS A BUG WITH A WRONG ARGUMENT BEING PASSED 
            THIS WILL BE ADDED IN THE FUTURE RELEASES OF THE TOOL"""

            imported_articles_counter += 1

            if tag_articles_flag:
                # assign tweets to the article -------------------------------------------------------------------------
                article_es_doc, all_article_tweets_set = assign_tweets_to_article(
                    article_dict=article_es_doc, coldstart_article_tweet_ids=coldstart_article_tweet_ids,
                    tweet_window_before=local_tweet_window_before, tweet_window_after=local_tweet_window_after,
                    n_keywords_for_tweet_match=config.N_KEYWORDS_FOR_TWEET_MATCH
                )

                # tag the article --------------------------------------------------------------------------------------
                if len(all_article_tweets_set) > 0 or article_es_doc['numbertweets'] > 0:
                    article_hashtags = recommend_hashtags_for_an_article(
                        article=article_es_doc,
                        article_tweet_hits=all_article_tweets_set,
                        hashtag_global_freq_table=hashtag_global_freq_table,
                        max_global_freq=max_global_freq,
                        global_window_hashtag_tweets_dict=global_window_hashtag_tweets_dict,
                        article_tweets_sample_size=config.LOCAL_TWEET_SAMPLE_SIZE,
                        article_tweets_sample_type=config.LOCAL_TWEET_SAMPLE_TYPE,
                        recommendation_conf_thres=config.RECOMMENDATION_CONF_THRES, hashtag_limit=config.HASHTAG_LIMIT
                    )

                    logger_message = "hashtags for %s '%s' are: %s" % (
                        article_es_doc['id'],
                        article_es_doc['headline'],
                        str(["%s: %.2f" % (toa['hashtag'], toa['score']) for toa in article_hashtags])
                        if article_hashtags else " - "
                    )
                    if export_es_instance is None and export_file_name is None:
                        _logger.info(logger_message)
                    else:
                        _logger.debug(logger_message)
                    del logger_message
                else:
                    article_hashtags = []
                    _logger.debug(
                        "the article %s '%s' didn't have any hashtagged tweets associated with it" %
                        (article_es_doc['id'], article_es_doc['headline'])
                    )

                # create the post-tagging fields for the article_es_doc ------------------------------------------------
                if article_hashtags:
                    hashtag_recommendations = dict((toa['hashtag'], toa['score']) for toa in article_hashtags)
                    hashtag_profile = create_hashtag_profile(
                        doc_hashtags=article_hashtags, hashtag_blacklist=config.HASHTAG_BLACKLIST,
                        min_score=config.HASHTAG_PROFILE_MIN_SCORE
                    )
                else:
                    hashtag_recommendations = {}
                    hashtag_profile = {}
                article_es_doc['recommendations'] = json.dumps(hashtag_recommendations)
                article_es_doc['hashtag_profile'] = json.dumps(hashtag_profile)
                article_es_doc['good_hashtags'] = list(hashtag_profile.keys()) if hashtag_profile else []
                article_es_doc['n_good_hashtags'] = len(article_es_doc['good_hashtags'])
                # article_es_doc['all_hashtags'] = " ".join(all_hashtags_of_the_article)
                article_es_doc['all_hashtags'] = [toa['hashtag'] for toa in article_hashtags] if article_hashtags else []
                article_es_doc['n_hashtags'] = len(article_es_doc['all_hashtags'])
                _logger.info(
                    "the hashtag profile of %s '%s' is %s" %
                    (article_es_doc['id'], article_es_doc['headline'], article_es_doc['hashtag_profile'])
                )

            # write the article to the export file ---------------------------------------------------------------------
            if export_file_name is not None:
                json.dump(article_es_doc, export_file, default=json_serial)
                export_file.write("\n")

            # save the article for indexing in Elasticsearch -----------------------------------------------------------
            article_es_docs.append(article_es_doc)
            # index the saved articles in Elasticsearch ----------------------------------------------------------------
            if len(article_es_docs) >= config.ARTICLE_BATCH_SIZE:
                if export_es_instance is not None:  # send the articles for indexing in Elasticsearch
                    _logger.info(
                        "passing %d articles (%d since the start) to write to the elasticsearch index" %
                        (len(article_es_docs), imported_articles_counter)
                    )
                    update_article_es_index(article_es_docs)
                article_es_docs = []
        if article_es_docs:
            if export_es_instance is not None:  # send the articles for indexing in Elasticsearch
                _logger.info(
                    "passing the last %d articles (%d since the start) to write to the elasticsearch index" %
                    (len(article_es_docs), imported_articles_counter)
                )
                update_article_es_index(article_es_docs)
        with open("duplicate_import_docs_ids_%s.txt" % datetime.utcnow().strftime('%H_%M_%S'), "a") as f:
            for (key, list_of_dup_ids) in dict_of_unique_timestamp_title_pairs.items():
                if len(list_of_dup_ids) > 1:
                    f.write(str(list_of_dup_ids) + "\n")
    if export_file_name is not None:
        export_file.close()
    _logger.info(
        "finished importing %d articles from %s in %d seconds" %
        (imported_articles_counter, articles_json_path, (datetime.utcnow() - start_time).total_seconds())
    )
# ---------------------------------------------- END THE MAIN FUNCTION ----------------------------------------------- #


########################################################################################################################
# ------------------------------------------- GLOBAL DATA & STATS FUNCTIONS ------------------------------------------ #
########################################################################################################################
def get_global_window_article_stats(doc_unix_timestamp,
                                    global_article_window_before=24 * 3600, global_article_window_after=3600):
    time_start = datetime.fromtimestamp(doc_unix_timestamp - global_article_window_before, tz=timezone.utc)
    time_end = datetime.fromtimestamp(doc_unix_timestamp + global_article_window_after, tz=timezone.utc)
    _logger.debug(
        "getting global word frequencies and article stats between %s and %s" %
        (datetime.strftime(time_start, "%Y-%m-%d %H:%M"), datetime.strftime(time_end, "%Y-%m-%d %H:%M"))
    )

    # retrieve all articles in the global window
    es_hits_of_articles_in_global_time_window, n_articles_in_global_window = search_es_articles(
        time_start=time_start,
        time_end=time_end,
        return_generator=True
    )
    # list_of_pseudoarticle_words_in_global_time_window = [
    #     word for a in es_hits_of_articles_in_global_time_window for word in a['_source']['stemming_title']
    # ]
    # global_window_article_word_count_dict = Counter(list_of_pseudoarticle_words_in_global_time_window)

    # create a word document frequency table
    keyword_article_count_dict = defaultdict(int)
    max_article_count = 0
    for a in es_hits_of_articles_in_global_time_window:
        # note that <a['_source']['stemming_title']> is not just the title, but the stemmed pseudoarticle !!!!!!!!
        # the confusing name is left for consistency with the legacy code and data
        for word in set(a['_source']['stemming_title'].strip().split(" ")):
            keyword_article_count_dict[word] += 1
            if keyword_article_count_dict[word] > max_article_count:
                max_article_count = keyword_article_count_dict[word]
    global_window_word_df_dict = dict(
        (word, article_count / max_article_count) for (word, article_count) in keyword_article_count_dict.items()
    )
    return n_articles_in_global_window, global_window_word_df_dict


def get_global_window_tweet_stats(doc_unix_timestamp,
                                  global_tweet_window_before=24 * 3600, global_tweet_window_after=3600,
                                  tweet_sample_size=10000, coldstart_flag=True,
                                  coldstart_article_window_before=60 * 24 * 3600,
                                  coldstart_article_window_after=24 * 3600):
    # get tweet-related features and stats
    hashtag_global_freq_table, max_global_freq, hashtag_global_tweets_dict = get_global_all_hashtag_freq_from_tweets(
        time_start=datetime.fromtimestamp(doc_unix_timestamp - global_tweet_window_before, tz=timezone.utc),
        time_end=datetime.fromtimestamp(doc_unix_timestamp + global_tweet_window_after, tz=timezone.utc),
        sample_size=tweet_sample_size
    )

    # bootstrap tweets from similar articles (cold start in real-time hashtagger) --------------------------------------
    coldstart_article_tweet_ids = []
    if coldstart_flag:
        es_hits_of_coldstart_articles, n_coldstart_articles = search_es_articles(
            time_start=datetime.fromtimestamp(doc_unix_timestamp - coldstart_article_window_before, tz=timezone.utc),
            time_end=datetime.fromtimestamp(doc_unix_timestamp + coldstart_article_window_after, tz=timezone.utc),
            only_tagged=True, return_generator=True
        )
        _logger.info("retrieved %d articles to train a kNN classifier for coldstart" % n_coldstart_articles)
        # train a kNN classifier on the old articles -------------------------------------------------------------------
        if n_coldstart_articles > 0:
            coldstart_pseudoarticles = []
            for a in es_hits_of_coldstart_articles:
                coldstart_pseudoarticles.append(a['_source']['processed_pseudoarticle'])
                coldstart_article_tweet_ids.append(a['_source']['tweets'])
            training_content = _tfidf_vectorizer.fit_transform(coldstart_pseudoarticles)
            training_labels = ["Empty"] * n_coldstart_articles
            # The following "classifier" will only learn a representation based on TFIDF of the pseudoarticles.
            #     THERE IS NO CLASSIFICATION GOING HERE PER SE!!!
            _coldstart_knn_classifier.fit(training_content, training_labels)
            _logger.debug("finished training a kNN classifier for coldstart")
    return hashtag_global_freq_table, max_global_freq, hashtag_global_tweets_dict, coldstart_article_tweet_ids


def get_global_all_hashtag_freq_from_tweets(time_start, time_end, sample_size=10000):
    # DateTime = datetime.utcnow().replace(tzinfo=utc)
    # th = DateTime - timedelta(hours=6)
    _logger.debug(
        "getting global hashtag frequencies between %s and %s" %
        (datetime.strftime(time_start, "%Y-%m-%d %H:%M"), datetime.strftime(time_end, "%Y-%m-%d %H:%M"))
    )

    # sample tweets for speed
    if sample_size > 10000 and config.GLOBAL_TWEET_SAMPLE_RANDOM_FLAG:
        # Elasticsearch doesn't support random scoring for sizes bigger than 10000
        # list_of_tweets, tweets_true_count = search_es_tweets(
        #     time_start=time_start, time_end=time_end, min_hashtags=1,
        #     return_random_sample_flag=False, size=None, return_generator=False
        # )
        # # manually sample from the full tweet collection
        # if tweets_true_count // sample_size > 1:
        #     list_of_tweets = list(ordered_sample_without_replacement(
        #         list_of_tweets, sample_size
        #     ))
        # according to the discussion here https://discuss.elastic.co/t/random-scan-results/15908
        # it is better to issue multiple randomly ordered <=10000 document requests than a scan
        # TODO: add a check for the real count and sample only if tweets_true_count // sample_size > 1:
        list_of_tweets = []
        random_sample_subset_size = 1000
        for i in range(int(np.ceil(sample_size / random_sample_subset_size))):
            iteration_list_of_tweets, iteration_tweets_true_count = search_es_tweets(
                time_start=time_start, time_end=time_end, min_hashtags=1,
                return_random_sample_flag=True, size=random_sample_subset_size, return_generator=True
            )
            list_of_tweets.extend(iteration_list_of_tweets)
        list_of_tweets = list({t['_id']: t for t in list_of_tweets}.values())
        tweets_true_count = len(list_of_tweets)
    else:
        list_of_tweets, tweets_true_count = search_es_tweets(
            time_start=time_start, time_end=time_end, min_hashtags=1,
            return_random_sample_flag=config.GLOBAL_TWEET_SAMPLE_RANDOM_FLAG, size=sample_size, return_generator=False
        )
    n_tweets = len(list_of_tweets)
    _logger.debug(
        "retrieved a random sample of %d tweets (from counted %d) for 'global' hashtag count estimation" %
        (n_tweets, tweets_true_count)
    )
    if n_tweets > 0:
        sample_ratio = int(tweets_true_count / n_tweets)
    else:
        sample_ratio = 0

    # get the hashtag global count dictionary
    hashtag_global_freq_table, max_global_freq, hashtag_global_tweets_dict = get_hashtag_count_and_tweets_dict(
        t["_source"] for t in list_of_tweets
    )
    _logger.debug("got the hashtag global count dictionary")

    # compensate for sampling
    if sample_ratio > 1:
        for h, h_count in hashtag_global_freq_table.items():
            hashtag_global_freq_table[h] = h_count * sample_ratio
        max_global_freq *= sample_ratio

    return hashtag_global_freq_table, max_global_freq, hashtag_global_tweets_dict


def get_hashtag_count_and_tweets_dict(list_of_tweets):
    """
        because this query is frequent, it makes sense to store the 'hashtag' field separately
        by adding {..., "store": "yes"} in the mapping for the 'hashtag' field
        https://www.elastic.co/guide/en/elasticsearch/reference/current/mapping-store.html
        https://discuss.elastic.co/t/extracting-all-values-for-a-term/4254/4
        """
    # list_of_hashtags = [
    #     h for h_list in [t["hashtags_list"] for t in list_of_tweets] for h in h_list if h is not None
    # ]

    # create a tweet_id lookup table for hashtags,
    # otherwise in every iteration of the next loop it will require an iteration through all the tweets
    hashtag_tweets_dict = defaultdict(list)
    list_of_hashtags = []
    for t in list_of_tweets:
        for h in t['hashtags_list']:
            hashtag_tweets_dict[h].append(t)
            list_of_hashtags.append(h)
    # hashtag_tweets_dict = reverse_dict_of_lists(dict((t["id"], t["hashtags_list"]) for t in article_tweets))

    hashtag_freq_table = Counter(list_of_hashtags)
    if len(hashtag_freq_table) == 0:
        return hashtag_freq_table, 0, hashtag_tweets_dict

    max_freq = hashtag_freq_table.most_common(1)[0][1]
    return hashtag_freq_table, max_freq, hashtag_tweets_dict
# ---------------------------------------- END GLOBAL DATA & STATS FUNCTIONS ----------------------------------------- #


########################################################################################################################
# ------------------------------------------------- ARTICLE FUNCTIONS ------------------------------------------------ #
########################################################################################################################
def extract_keywords(text, max_keywords=5, unigram_mode="no_verbs"):
    # stopwords = set(nltk.corpus.stopwords.words('english').extend(config.ADDITIONAL_STOPWORDS))
    pos_tokens = nltk.pos_tag(word_tokenize(text))
    entities = _noun_phrase_regex_parser.parse(pos_tokens)

    pos_tagged_nouns = []
    noun_phrases = []
    set_of_noun_phrase_tokens = set([])
    selected_nouns = []
    selected_verbs = []
    for e in entities:
        if isinstance(e, nltk.tree.Tree):  # in other cases, the element is a tuple, not a tree of tuples !!!!!!!!!!!
            if e.label() == "NP":
                noun_phrase_tokens = []
                for (pos_token, pos_type) in e:
                    if pos_type in ['NN', 'NNP', 'NNS']:
                        noun = re.sub("[:;>?<=*+().,!\"$%\{˜|\}\[ˆ_\\]’‘]", "", pos_token.lower())
                        if noun not in _stopwords:
                            noun_phrase_tokens.append(noun)
                set_of_noun_phrase_tokens.update(noun_phrase_tokens)
                noun_phrase = " ".join(noun_phrase_tokens).replace(" & ", "&").replace(" @ ", " @").strip()
                pos_tagged_nouns.append((noun_phrase, "NP"))
                if noun_phrase not in noun_phrases:
                    noun_phrases.append(noun_phrase)
        elif e[1] in ['NN', 'NNP', 'NNS']:
            noun = e[0].lower()
            if noun not in _stopwords:
                pos_tagged_nouns.append((noun, e[1]))
                selected_nouns.append(noun)
        elif e[1] in ['VB']:
            verb = e[0].lower()
            if verb not in _stopwords:
                selected_verbs.append(verb)
    # nouns_freq = nltk.FreqDist(pos_tagged_nouns)

    # remove the unigrams that appear in noun phrases
    unigrams_not_in_phrases = [w for w in selected_nouns if w not in set_of_noun_phrase_tokens and len(w) > 1]
    # prioritize the noun phrases and randomly select some nouns and verbs
    if len(noun_phrases) >= max_keywords:
        return noun_phrases[:max_keywords]
    if unigram_mode == "nouns_first":
        keywords = noun_phrases + unigrams_not_in_phrases + selected_verbs
    elif unigram_mode == "verbs_first":
        keywords = noun_phrases + selected_verbs + unigrams_not_in_phrases
    elif unigram_mode == "no_verbs":
        keywords = noun_phrases + unigrams_not_in_phrases
    elif unigram_mode == "no_nouns":
        keywords = noun_phrases + selected_verbs
    elif unigram_mode == "random":
        unigram_nouns_and_verbs = unigrams_not_in_phrases + selected_verbs
        keywords = noun_phrases + list(
            ordered_sample_without_replacement(
                unigram_nouns_and_verbs,
                sample_size=min(
                    max(max_keywords - len(noun_phrases), 0),
                    len(unigram_nouns_and_verbs)
                )
            )
        )
    elif unigram_mode == "equal":
        keywords = noun_phrases + [
            u for u in chain.from_iterable(zip_longest(unigrams_not_in_phrases, selected_verbs)) if u is not None
        ]
    else:
        raise Exception(
            "the 'unigram_mode' can take only one the following values: "
            "['random', 'equal', 'nouns_first', 'verbs_first', 'no_nouns', 'no_verbs']"
        )
    return keywords[:max_keywords]


def get_keyword_ngrams(keywords, entities, profile, ngram_length=2, entity_boost=3, location_boost=2):
    ngrams = []
    unigrams_to_pair = []
    for keyword in set(keywords + [e for (e, e_type) in entities]):
        if " " in keyword.strip():
            # create all possible n-grams (bigrams) from a noun phrase or entity with alphabetically ordered terms
            for ngram in itertools.combinations(sorted(keyword.strip().split(" "), key=str.lower), ngram_length):
                ngrams.append(" ".join(ngram))
        else:
            unigrams_to_pair.append(keyword.strip())

    unigrams_to_pair = sorted(list(set(unigrams_to_pair)), key=str.lower)
    if len(unigrams_to_pair) > 1:
        for ngram in itertools.combinations(unigrams_to_pair, ngram_length):
            ngrams.append(" ".join(ngram))
    ngrams = list(set(ngrams))

    # rank the ngrams
    if profile:
        min_profile_score = min(profile.values())
    else:
        min_profile_score = 1
    entities_dict = dict(entities)
    scored_ngrams = []
    for ngram in ngrams:
        ngram_tokens = ngram.split(" ")
        ngram_score = min_profile_score
        for ngram_token in ngram_tokens:
            ngram_token_stemmed = stem(_lemmatizer.lemmatize(ngram_token))  # because this is what we've in the profile
            if profile:
                if ngram_token_stemmed in profile:
                    ngram_score *= profile[ngram_token_stemmed]
            if ngram_token in entities_dict:
                if entities_dict[ngram_token] in ['PERSON', 'ORGANIZATION']:
                    ngram_score *= entity_boost
                elif entities_dict[ngram_token] == 'LOCATION':
                    ngram_score *= location_boost
        scored_ngrams.append((ngram, ngram_score))
    ranked_ngrams = [ngram for (ngram, s) in sorted(scored_ngrams, key=lambda ngram_s: ngram_s[1], reverse=True)]
    return ",".join(ranked_ngrams)


def create_article_dict_without_profile(a_id, title, subtitle, body, unix_timestamp, url, source, a_type):
    # first_sentence = body.split(".")[0]  # for consistency with the legacy code (but this is not the best way)
    if body:
        first_sentence = sent_tokenize(body)[0]
    else:
        first_sentence = ""
    stemmed_title = remove_stopwords_non_alpha_and_lemmatize(
        title + " " + subtitle + " " + first_sentence, lemmatizer=_lemmatizer, stopwords=_stopwords
    )
    # the <stemmed_title> is in fact the stemmed pseudoarticle
    # and the confusing name is left for consistency with the legacy code and data
    stemmed_content = remove_stopwords_non_alpha_and_lemmatize(body, lemmatizer=_lemmatizer, stopwords=_stopwords)

    keywords_list = extract_keywords(
        text=title + ", " + subtitle + ", " + first_sentence,
        max_keywords=config.MAX_N_ARTICLE_KEYWORDS, unigram_mode='no_verbs'
    )
    keywords = ", ".join(keywords_list)
    _logger.debug("keywords: %s" % keywords)
    # the following function was the most expensive part of the pre-tagging pipeline
    entities = stanford_ner_tag(title + " " + subtitle + " " + first_sentence)
    _logger.debug("entities: %s" % entities)
    paired_keyword_str = get_keyword_ngrams(keywords_list, entities, None)  # HERE IS A BUG with <tfidf> !!!!!!!!!!!!
    _logger.debug("paired keywords with ner (assigned as 'stream_keywords'): %s" % paired_keyword_str)

    a_dict = {
        'id': a_id, 'headline': title, 'subheadline': subtitle, 'content': body,
        'url': url, 'datetime': datetime.fromtimestamp(unix_timestamp, tz=timezone.utc),
        'keywords': keywords, 'stream_keywords': paired_keyword_str,
        'stemming_title': stemmed_title, 'stemming_content': stemmed_content, 'profile': {},
        'type': a_type, 'source': source, 'reference': None, 'numbertweets': 0,
        'havehashtag': False
    }

    # add/modify fields beyond the the legacy RDBMS Article schema
    if a_dict['source'] == "The Washington Post":
        a_dict['unique_id'] = a_dict['id']
    else:
        try:
            a_dict['unique_id'] = a_dict['source'].replace(" ", "") + "_" + re.findall(
                "\w+/rss2/|\d+\-\w+\d+|\d+\.\d+|\d+|id[\w|\d]+", a_dict['url']
            )[-1].replace('/rss2/', '')
        except IndexError as e:
            # when re.findall() returns an empty list
            a_dict['unique_id'] = a_dict['source'].replace(" ", "") + "_" + str(a_dict['epoch'])
    # a_dict['profile'] = str(a_dict['profile'])  # !!!!!!!!!!!!! MAY DIFFER FROM THE LEGACY DATA !!!!!!!!!!!!!!!!!!!!!!
    a_dict['all_hashtags'] = []  # this field must be updated later and must not override the current values

    a_dict['first_sentence'] = first_sentence
    a_dict['pseudoarticle'] = " ".join([a_dict['headline'], a_dict['subheadline'], a_dict['content']])
    a_dict['nes'], a_dict['noun_tokens'], a_dict['tokens'], a_dict['unique_tokens'], a_dict['processed_pseudoarticle'] = \
        ner_tokenize(a_dict['pseudoarticle'])
    a_dict['epoch'] = a_dict['datetime'].timestamp()
    return a_dict


def get_article_profile(keywords, article_content, n_articles_in_global_window, global_window_word_df_dict):
    # get the tfidf profile of the article's content
    # based on the words in the stemmed pseudoarticle (title + subheadline + first sentence)
    if n_articles_in_global_window == 0:
        return dict((w, round(1 / len(keywords), 4)) for w in keywords)
    content_keyword_count_dict = Counter([word for word in article_content.split(" ") if word in keywords])
    try:
        max_word_freq = content_keyword_count_dict.most_common(1)[0][1]
    except IndexError:
        _logger.warning(
            "the number of pseudoarticle keywords found in the content is 0, skipping TFIDF for profile computation"
        )
        # when none of the keywords in the pseudoarticle (title + subheadline + first sentence) are found in the title
        return dict((w, round(1 / len(keywords), 4)) for w in keywords)
    keywords_tfidf_tuples = []
    for word in keywords:
        try:
            global_window_word_df = global_window_word_df_dict[word]
        except KeyError:
            global_window_word_df = 1  # to avoid division by 0
        keywords_tfidf_tuples.append(
            (
                word,
                np.log(1 + content_keyword_count_dict[word] / max_word_freq) *  # TF
                np.log(n_articles_in_global_window / global_window_word_df)  # IDF
            )
        )
    # normalize the profile so that the sum of the frequency squares is 1 (because it's going to be treated as a vector)
    vec_length = np.sqrt(sum(freq ** 2 for (w, freq) in keywords_tfidf_tuples))
    if vec_length == 0:
        return dict((w, round(1 / len(keywords), 4)) for w in keywords)
    keywords_tfidf_scores_dict = dict((w, round(freq / vec_length, 4)) for (w, freq) in keywords_tfidf_tuples)
    # the rounding is probably for shorter number if converted to a string (came from the legacy code)
    return keywords_tfidf_scores_dict


def construct_tweet_es_query_from_ngrams(list_of_ngrams, field_name="tweetcontent_clean", mode="must"):
    if mode not in ['must', 'should', 'must_not', 'filter']:
        _logger.error("the mode can only be an ES 'bool' query type: ['must', 'should', 'must_not', 'filter']")
    tweet_query_should_clauses = []
    for term_ngram in list_of_ngrams:
        n_gram_must_clause = []
        for term in term_ngram.split(" "):
            n_gram_must_clause.append(
                {"match_phrase": {field_name: term}}
            )
        tweet_query_should_clauses.append({"bool": {mode: n_gram_must_clause}})
    return {"should": tweet_query_should_clauses}


def assign_tweets_to_article(article_dict, tweet_window_before=24*3600, tweet_window_after=24*3600,
                             coldstart_article_tweet_ids=None, n_keywords_for_tweet_match=5):
    # bootstrap tweets from similar articles ---------------------------------------------------------------------------
    coldstart_tweets = []
    if coldstart_article_tweet_ids:
        knn_neighbor_article_indices = _coldstart_knn_classifier.kneighbors(
            _tfidf_vectorizer.transform([article_dict['processed_pseudoarticle']]),
            n_neighbors=config.COLDSTART_N_NEIGHBOURS, return_distance=False
        )[0]
        coldstart_tweet_ids = []
        for i in knn_neighbor_article_indices:
            coldstart_tweet_ids.extend(
                coldstart_article_tweet_ids[i].split(",")[-config.COLDSTART_N_TWEETS_PER_NEIGHBOUR_ARTICLE:]
            )
        # this query is meant to substitute
        # iteration over the retrieved tweet set
        # and applying matchKeywords2(article_obj.Stream_Keywords, tweet.TweetContent_Clean.split(' ')))
        coldstart_tweet_should_query = construct_tweet_es_query_from_ngrams(
            article_dict["stream_keywords"].split(",")[:n_keywords_for_tweet_match],
            field_name=config.COLDSTART_TWEET_FIELD_NAME, mode=config.COLDSTART_TWEET_NGRAM_MATCH_MODE
        )
        coldstart_tweets, n_coldstart_tweets = get_tweets_by_id(
            [tweet_id for tweet_id in set(coldstart_tweet_ids) if tweet_id != ""], query=coldstart_tweet_should_query,
            min_hashtags=config.LOCAL_TWEET_MIN_N_HASHTAGS, max_hashtags=config.LOCAL_TWEET_MAX_N_HASHTAGS,
            size=len(coldstart_tweet_ids), return_generator=False
        )
        _logger.debug("retrieved %d from %d potential coldstart tweets found in %d nearest neighbour articles" % (
            n_coldstart_tweets, len(coldstart_tweet_ids), config.COLDSTART_N_NEIGHBOURS
        ))

    # assign tweets to the article querying in the local time window
    tweet_window_before_datetime = article_dict['datetime'] - timedelta(seconds=tweet_window_before)
    tweet_window_after_datetime = article_dict['datetime'] + timedelta(seconds=tweet_window_after)
    # the legacy code was written for a streaming scenario and the timestamps were taken w.r.t. datetime.now()

    # this query is meant to substitute
    # iteration over the retrieved tweet set
    # and applying matchKeywords2(article_obj.Stream_Keywords, tweet.TweetContent_Clean.split(' ')))
    article_tweet_should_query = construct_tweet_es_query_from_ngrams(
        article_dict["stream_keywords"].split(",")[:n_keywords_for_tweet_match],
        field_name=config.LOCAL_TWEET_FIELD_NAME, mode=config.LOCAL_TWEET_NGRAM_MATCH_MODE
    )
    _logger.debug("constructed the tweet query")
    # for the reasons of backward compatibility and considering that there is no agreed schema for tweets in ES,
    # 'query' given to the 'search_es_tweets' must be a dict and an argument to 'bool' and not textual
    all_tweets_with_hashtags, tweets_true_count = search_es_tweets(
        query=article_tweet_should_query,
        time_start=tweet_window_before_datetime, time_end=tweet_window_after_datetime,
        min_hashtags=config.LOCAL_TWEET_MIN_N_HASHTAGS, max_hashtags=config.LOCAL_TWEET_MAX_N_HASHTAGS,
        return_generator=False
    )
    # TODO: repeat with a bigger time interval and/or more keywords
    #  if there are less than <SOME_THRESHOLD> tweets assigned
    _logger.debug("%d tweet(s) retrieved between %s and %s matching at least one of these n-grams: %s" % (
        len(all_tweets_with_hashtags),
        datetime.strftime(tweet_window_before_datetime, "%Y-%m-%d %H:%M"),
        datetime.strftime(tweet_window_after_datetime, "%Y-%m-%d %H:%M"),
        article_dict["stream_keywords"].split(",")[:n_keywords_for_tweet_match]
    ))

    # modify the coldstart tweet scores for mixing with the other article tweets
    # the scores make sense only for a given retrieval and never across different retrievals
    if len(coldstart_tweets) > 0 and tweets_true_count > 0:
        max_tweet_score = all_tweets_with_hashtags[0]['_score']
        # by replacing the coldstart tweet scores with the highest scored keyword-retrieved tweet score
        # we allow to prioritize the coldstart tweets
        coldstart_tweets = [t_hit for t_hit in coldstart_tweets if not t_hit.update({'_score': max_tweet_score})]

    # when selecting the unique tweets with .values() either use OrderedDict or sort afterwards!
    all_article_tweets_set = sorted({
        t["_id"]: t for t in coldstart_tweets + all_tweets_with_hashtags
        if len(t["_source"]["tweetcontent_clean"].split(" ")) >= config.LOCAL_TWEET_MIN_N_TOKENS
    }.values(), key=lambda tt: tt["_score"], reverse=True)
    _logger.debug(
        "the article got assigned %d unique tweet(s) with >=%d words from %d queried and %d coldstart tweets" %
        (len(all_article_tweets_set), config.LOCAL_TWEET_MIN_N_TOKENS, tweets_true_count, len(coldstart_tweets))
    )
    # TODO: consider not throwing away all the existing old tweets if any
    article_dict['tweets'] = [t["_id"] for t in all_article_tweets_set]
    article_dict['numbertweets'] = len(article_dict['tweets'])
    return article_dict, all_article_tweets_set
# ---------------------------------------------- END ARTICLE FUNCTIONS ----------------------------------------------- #


########################################################################################################################
# ------------------------------------------------- FEATURE FUNCTIONS ------------------------------------------------ #
########################################################################################################################
def min_max_normalize_item(item, max_value, min_value, log_flag=False):
    if item != 0:
        if max_value != min_value:
            if log_flag:
                item = np.log(item)
                max_value = np.log(max_value)
                min_value = np.log(min_value)
            return (item - min_value) / (max_value - min_value)
        else:
            return 1
    else:
        return 0


def get_burstiness_features(article_tweets_containing_the_hashtag, article_timestamp,
                            hashtag_local_frequency, max_local_freq, interval=30*60):
    n_tweets_recent_interval = 0
    n_tweets_earlier_interval = 0
    for t in article_tweets_containing_the_hashtag:
        tweet_timestamp = datetime.strptime(t['datetime'][:-3] + "00", "%Y-%m-%dT%H:%M:%S%z")
        if article_timestamp - timedelta(seconds=2 * interval) < tweet_timestamp < article_timestamp:
            if tweet_timestamp > article_timestamp - timedelta(seconds=interval):
                n_tweets_recent_interval += 1
            else:
                n_tweets_earlier_interval += 1

    # Trending hashtag: Captures a significant increase in local hashtag frequency and
    # aims to identify article-wise trending hashtags.
    if n_tweets_earlier_interval != 0:
        tweet_count_growth_ratio = (n_tweets_recent_interval - n_tweets_earlier_interval) / n_tweets_earlier_interval
        if tweet_count_growth_ratio > 2:
            tweet_count_growth_ratio_capped = 2
        elif tweet_count_growth_ratio < -1:
            tweet_count_growth_ratio_capped = -1
        else:
            tweet_count_growth_ratio_capped = tweet_count_growth_ratio
        # this way is also efficient https://stackoverflow.com/a/22902954/2262424
    else:
        tweet_count_growth_ratio_capped = 2

    # Expected Gain: Captures the potential of h in the next time window,
    # and is expected to boost trending hashtags while punishing fading ones.
    expected_local_freq = hashtag_local_frequency * (1 + tweet_count_growth_ratio_capped)
    # normalize, so that it's comparable with other hashtags
    expected_local_freq_log_score = min_max_normalize_item(
        item=expected_local_freq,
        max_value=max(max_local_freq, expected_local_freq), min_value=1, log_flag=True
    )
    expected_local_freq_lin_score = min_max_normalize_item(
        item=expected_local_freq,
        max_value=max(max_local_freq, expected_local_freq), min_value=1, log_flag=False
    )
    return tweet_count_growth_ratio_capped, expected_local_freq_log_score, expected_local_freq_lin_score


def get_time_window_tweets_profile(contents_of_tweets, min_word_length=2, top_words_limit=20):
    list_of_words = []
    for tweet_content in contents_of_tweets:
        list_of_words.extend(tweet_content.split(" "))
    list_of_words = [word for word in list_of_words if is_ascii(word) and len(word) >= min_word_length]
    word_count_dict = Counter(list_of_words)
    # keep only 'top_words_limit' most frequent words
    word_count_dict_filtered = sorted(word_count_dict.items(), key=lambda x: x[1], reverse=True)[:top_words_limit]

    # log-normalize the word frequencies... the rounding is for shorter numbers when converted to a string
    highest_freq = word_count_dict_filtered[0][1]
    word_freq_list = [(w, np.log(freq / highest_freq + 1)) for (w, freq) in word_count_dict_filtered]
    # normalize the profile so that the sum of the frequency squares is 1 (because it's going to be treated as a vector)
    vec_length = np.sqrt(sum(freq ** 2 for (w, freq) in word_freq_list))
    time_window_tweet_profile = dict((w, round(freq / vec_length, 4)) for (w, freq) in word_freq_list)
    return time_window_tweet_profile


def get_hashtag_time_window_features(contents_of_tweets, hashtag_frequency, max_freq, reference_profile,
                                     sample_size=None):
    if len(contents_of_tweets) > 0:
        if sample_size is not None:
            if sample_size > len(contents_of_tweets):
                sample_size = len(contents_of_tweets)
            contents_of_tweets = ordered_sample_without_replacement(
                contents_of_tweets, sample_size=sample_size
            )
        hashtag_tweet_profile = get_time_window_tweets_profile(
            contents_of_tweets=contents_of_tweets,
            min_word_length=config.MIN_WORD_LENGTH, top_words_limit=config.TOP_WORDS_LIMIT
        )
    else:
        hashtag_tweet_profile = {}
    similarity_score = dict_cosine_similarity(reference_profile, hashtag_tweet_profile)
    freq_log_score = min_max_normalize_item(item=hashtag_frequency, max_value=max_freq, min_value=1, log_flag=True)
    freq_lin_score = min_max_normalize_item(item=hashtag_frequency, max_value=max_freq, min_value=1, log_flag=False)
    return freq_log_score, freq_lin_score, similarity_score, hashtag_tweet_profile


def get_local_hashtag_features(hashtag_local_frequency, max_local_freq, article_tweets_containing_the_hashtag,
                               article_profile, article_timestamp):
    # The following three constants are correspondingly the average/max/median number of followers estimated on a
    # "large" tweet collection at some point...
    # Currently though this is needed because the L2R classifier training data was scaled with these constants!
    # Technically, if the L2R classifier is a Random Forest (so if it doesn't require normalized features),
    # the training data can be scaled back and the scaling in the code can be removed.
    ptp_avg = 12.5805270718
    ptp_max = 16.8929446083
    ptp_median = 12.0549212135

    # get the article tweet features -----------------------------------------------------------------------------------
    # ids_of_article_tweets_containing_the_hashtag = []
    contents_of_article_tweets_containing_the_hashtag = []
    users_of_article_tweets_containing_the_hashtag = []
    n_followers_of_the_users_of_article_tweets_containing_the_hashtag = []
    for t in article_tweets_containing_the_hashtag:
        # ids_of_article_tweets_containing_the_hashtag.append(t['tweetid'])
        contents_of_article_tweets_containing_the_hashtag.append(t['tweetcontent_clean'])
        users_of_article_tweets_containing_the_hashtag.append(t['user'])
        n_followers_of_the_users_of_article_tweets_containing_the_hashtag.append(t['follower'])

    # get the local time window features for the hashtag ---------------------------------------------------------------
    local_freq_log_score, local_freq_lin_score, similarity_score, local_window_hashtag_tweet_profile = \
        get_hashtag_time_window_features(
            contents_of_tweets=contents_of_article_tweets_containing_the_hashtag,
            hashtag_frequency=hashtag_local_frequency, max_freq=max_local_freq, reference_profile=article_profile,
            sample_size=config.HASHTAG_WINDOW_TWEET_SAMPLE_SIZE
        )

    # get twitter user-related features --------------------------------------------------------------------------------
    unique_user_ratio = len(set(users_of_article_tweets_containing_the_hashtag)) \
                        / len(users_of_article_tweets_containing_the_hashtag)
    mean_n_foll = np.mean(n_followers_of_the_users_of_article_tweets_containing_the_hashtag)
    max_n_foll = max(n_followers_of_the_users_of_article_tweets_containing_the_hashtag)
    median_n_foll = np.median(n_followers_of_the_users_of_article_tweets_containing_the_hashtag)
    # NOTE! np.log(0) returns -inf whereas math.log(0) raises a ValueError
    avg_n_followers = min(np.log(mean_n_foll) / ptp_avg, 1) if mean_n_foll > 0 else 0
    max_n_followers = min(np.log(max_n_foll) / ptp_max, 1) if max_n_foll > 0 else 0
    median_n_followers = min(np.log(median_n_foll) / ptp_median, 1) if median_n_foll > 0 else 0

    # get anomaly-based features ---------------------------------------------------------------------------------------
    trending, expected_local_freq_log_score, expected_local_freq_lin_score = get_burstiness_features(
        article_tweets_containing_the_hashtag=article_tweets_containing_the_hashtag,
        article_timestamp=article_timestamp,
        hashtag_local_frequency=hashtag_local_frequency, max_local_freq=max_local_freq,
        interval=config.BURSTINESS_INTERVAL
    )
    return local_freq_log_score, local_freq_lin_score, similarity_score, local_window_hashtag_tweet_profile, \
           unique_user_ratio, avg_n_followers, max_n_followers, median_n_followers, \
           trending, expected_local_freq_log_score, expected_local_freq_lin_score
# ---------------------------------------------- END ARTICLE FUNCTIONS ----------------------------------------------- #


########################################################################################################################
# ----------------------------------------- HASHTAG RECOMMENDATION FUNCTIONS ----------------------------------------- #
########################################################################################################################
def recommend_hashtags_for_an_article(article, hashtag_global_freq_table, max_global_freq,
                                      global_window_hashtag_tweets_dict, article_tweet_hits=None,
                                      article_tweets_sample_size=5000, article_tweets_sample_type="elbow",
                                      recommendation_conf_thres=0.5, hashtag_limit=10):
    if article_tweet_hits is None:
        article_tweet_hits, retrieved_tweet_count = get_tweets_by_id(
            list_of_ids=article['tweets'], min_hashtags=1, return_generator=False
        )
        _logger.debug("retrieved %d article tweets, expected %d" % (retrieved_tweet_count, article['numbertweets']))
        if article_tweets_sample_size and article_tweets_sample_type != "random":
            _logger.warning(
                "retrieved tweets for '%s' based on previously assigned tweet ids, "
                "therefor have to apply 'random' sampling of tweets instead of '%s'"
                % (article['id'], article_tweets_sample_type)
            )
            article_tweets_sample_type = "random"

    # subsample if there are more than twice tweets than 'article_tweets_sample_size' ----------------------------------
    if article_tweets_sample_size and len(article_tweet_hits) // article_tweets_sample_size > 1:
        if article_tweets_sample_type == "random":
            article_tweet_hits = ordered_sample_without_replacement(
                article_tweet_hits, min(article_tweets_sample_size, len(article_tweet_hits))
            )
            _logger.debug(
                "sampled %d random tweets from %d tweets the article had been assigned" %
                (article_tweets_sample_size, len(article['tweets']))
            )
        elif article_tweets_sample_type == "top":
            # assuming the tweets have been ordered by their relevance scores (to the keyword query) in decreasing order
            article_tweet_hits = article_tweet_hits[:article_tweets_sample_size]
            _logger.debug(
                "took top %d of %d tweets the article had been assigned" %
                (article_tweets_sample_size, len(article['tweets']))
            )
        elif article_tweets_sample_type == "elbow":
            tweet_score_elbow_idx = find_curve_elbow_idx_based_on_max_dist(
                [t_hit['_score'] for t_hit in article_tweet_hits]
            )
            article_tweet_hits = article_tweet_hits[:tweet_score_elbow_idx]
            _logger.debug(
                "took top %d of %d tweets (elbow cutoff) the article had been assigned" %
                (tweet_score_elbow_idx, len(article['tweets']))
            )
        elif article_tweets_sample_type in ["coldstart+elbow", "coldstart+elbow+random"]:
            # take all the coldstart tweets and apply an elbow cutoff to the keyword-retrieved tweets
            # assuming that the tweets are ordered by relevance scores and
            # the coldstart tweets have an artificial score equal to the highest score from the retrieved articles
            # in addition to "coldstart+elbow", also add LOCAL_TWEET_SAMPLE_SIZE random tweets from remaining tweets
            max_tweet_score = article_tweet_hits[0]["_score"]
            for i, t_hit in enumerate(article_tweet_hits):
                if t_hit["_score"] != max_tweet_score:
                    coldstart_idx = i
                    break
            coldstart_tweet_hits = article_tweet_hits[:coldstart_idx]
            retrieved_tweet_hits = article_tweet_hits[coldstart_idx:]

            tweet_score_elbow_idx = find_curve_elbow_idx_based_on_max_dist(
                [t_hit["_score"] for t_hit in retrieved_tweet_hits]
            )
            article_tweet_hits = coldstart_tweet_hits + retrieved_tweet_hits[:tweet_score_elbow_idx]
            if article_tweets_sample_type == "coldstart+elbow+random":
                article_tweet_hits += ordered_sample_without_replacement(
                    seq=retrieved_tweet_hits[tweet_score_elbow_idx:],
                    sample_size=min(
                        article_tweets_sample_size, len(article_tweet_hits),
                        len(retrieved_tweet_hits) - tweet_score_elbow_idx
                    )
                )
            _logger.debug(
                "took all %d coldstart tweets and top %d of %d tweets (elbow cutoff) retrieved for the article %s" %
                (len(coldstart_tweet_hits), tweet_score_elbow_idx, len(retrieved_tweet_hits),
                 "+ up to %d random tweets" % article_tweets_sample_size
                 if article_tweets_sample_type == "coldstart+elbow+random" else "")
            )
        else:
            raise Exception(
                "LOCAL_TWEET_SAMPLE_TYPE can only take the following values: "
                "['elbow', 'top', 'random', 'coldstart+elbow', 'coldstart+elbow+random'], "
                "whereas '%s' was given" % article_tweets_sample_type
            )

    local_hashtag_freq_table, max_local_freq, local_hashtag_tweets_dict = get_hashtag_count_and_tweets_dict(
        [t["_source"] for t in article_tweet_hits]
    )
    """In the legacy code of real-time Hashtagger+ nothing like 'local_hashtag_tweets_dict' was returned and instead
    for each hashtag there was a loop over a (5000-tweet-sized subset of) the article tweets and filtering the tweets 
    which contained the hashtag, smth like
        ids = [t["_id"] for t in article_tweets if h in t["_source"]["hashtags_list"]]
    Whereas the new code doesn't subsample the article tweets and pre-selects the hashtag tweets in 
    get_hashtag_count_and_tweets_dict() readying for further access without new iterations over the article tweet set"""

    hashtag_features_dict = {}
    list_of_hashtags = []
    list_of_clf_features = []

    for h in local_hashtag_freq_table.keys():  # !!!!!!!!!!!!!! THE ORDER !!!!!!!!!!!!!
        # get the tweets containing the hashtag ------------------------------------------------------------------------
        article_tweets_containing_the_hashtag = local_hashtag_tweets_dict[h]
        n_article_tweets_containing_the_hashtag = len(article_tweets_containing_the_hashtag)

        # sanity check -------------------------------------------------------------------------------------------------
        if n_article_tweets_containing_the_hashtag != local_hashtag_freq_table[h]:
            _logger.error(
                "something is wrong with the counts of article tweets containing the hashtag %s" % h +
                "\n counted tweets with hashtag --> %d != %d <-- from the local frequency table" %
                (n_article_tweets_containing_the_hashtag, local_hashtag_freq_table[h])
            )

        # subsample if there are more than twice tweets than 'config.GLOBAL_HASHTAG_TWEET_SAMPLE_SIZE' -----------------
        global_window_hashtag_tweets = global_window_hashtag_tweets_dict[h]
        """ !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        This is a leftover from the legacy code and is kept for reproducibility purposes only... IT'S BAD, AVOID IT !
        The reason why the following sampling is wrong is that 'hashtag_global_freq_table' and 'max_global_freq' 
        are computed on a different corpus of tweets"
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! """
        if config.GLOBAL_HASHTAG_TWEET_SAMPLE_SIZE:
            if n_article_tweets_containing_the_hashtag // config.GLOBAL_HASHTAG_TWEET_SAMPLE_SIZE > 1:
                global_window_hashtag_tweets = ordered_sample_without_replacement(
                    global_window_hashtag_tweets_dict[h], config.GLOBAL_HASHTAG_TWEET_SAMPLE_SIZE
                )
                _logger.warning(
                    "sampled %d random tweets from %d tweets containing hashtag %s in the [-%d <--> %d] seconds window"
                    % (config.GLOBAL_HASHTAG_TWEET_SAMPLE_SIZE, n_article_tweets_containing_the_hashtag, h,
                       config.GLOBAL_TWEET_WINDOW_BEFORE, config.GLOBAL_TWEET_WINDOW_AFTER)
                )

        # if the hashtag appears in not that many tweets, don't consider it --------------------------------------------
        if n_article_tweets_containing_the_hashtag <= max(
                config.LOCAL_MIN_N_ARTICLE_TWEETS_PER_TAG,
                max_local_freq * config.LOCAL_MIN_N_ARTICLE_TWEETS_PER_TAG_RELATIVE_TO_MAX
        ):
            continue

        # get the local time window features for the hashtag -----------------------------------------------------------
        local_freq_log_score, local_freq_lin_score, hashtag_article_similarity_score, \
            local_window_hashtag_tweet_profile, \
            unique_user_ratio, avg_n_followers, max_n_followers, median_n_followers, \
            trending, expected_local_freq_log_score, expected_local_freq_lin_score = \
            get_local_hashtag_features(
                hashtag_local_frequency=local_hashtag_freq_table[h],
                max_local_freq=max_local_freq,
                article_tweets_containing_the_hashtag=article_tweets_containing_the_hashtag,
                article_profile=article['profile'],
                article_timestamp=article['datetime']
            )
        # get the global time window features for the hashtag ----------------------------------------------------------
        global_freq_log_score, global_freq_lin_score, local_global_similarity_score, \
            global_window_hashtag_tweet_profile = \
            get_hashtag_time_window_features(
                contents_of_tweets=[t['tweetcontent_clean'] for t in global_window_hashtag_tweets],
                hashtag_frequency=hashtag_global_freq_table[h], max_freq=max_global_freq,
                reference_profile=local_window_hashtag_tweet_profile, sample_size=None
            )
        # get article content-related features, check if the tag without "#" is in the pseudoarticle -------------------
        tag_in_headline = int(h.strip("#") in article['pseudoarticle'].replace(" ", ""))

        # features
        features = [
            local_freq_log_score, local_freq_lin_score, hashtag_article_similarity_score,
            global_freq_log_score, global_freq_lin_score, local_global_similarity_score,
            tag_in_headline, unique_user_ratio, avg_n_followers, max_n_followers, median_n_followers,
            trending, expected_local_freq_log_score, expected_local_freq_lin_score
        ]
        _file_logger.debug("the features for %s in %s are %s" % (h, article['id'], features))
        hashtag_features_dict[h] = features
        list_of_hashtags.append(h)
        list_of_clf_features.append(features)

    if not hashtag_features_dict:
        _logger.debug("the article %s didn't have any hashtagged tweets associated with it" % article['id'])
        return None

    prediction_scores = _l2r_classifier.predict_proba(list_of_clf_features)
    ranked_hashtags = sorted(zip(list_of_hashtags, prediction_scores[:, 1]), key=lambda x: x[1], reverse=True)

    hashtag_article_recommend = []
    # pick the best tags
    for (h, h_score) in ranked_hashtags[:hashtag_limit]:
        if h_score > recommendation_conf_thres:
            hashtag_article_recommend.append(
                {"hashtag": h, "profile": hashtag_features_dict[h], "score": h_score})
        else:
            break  # because the hashtags were ranked already

    # if hashtag_article_recommend:
    #     addArticleToHashtag(article, hashtag_article_recommend)
    #
    #     # this is important as these will be child objects for the article object
    #     createTagOfArticleObject(article, "Recommend_new", hashtag_article_recommend)
    #     article.HaveHashtag = True

    return hashtag_article_recommend


def train_rf_classifier(data_type, n_rf_estimators=100):
    if data_type == 'Train':
        file_path = config.CLASSIFIER_DATA_TRAIN_FILE
    else:
        file_path = config.CLASSIFIER_DATA_ALL_FILE
    data = []
    with open(file_path, 'r', encoding='utf-8') as csv_file:
        file = csv.reader(csv_file, lineterminator='\n')
        next(file, None)
        for line in file:
            data.append(line)
    data = np.array(data)
    x_train = data[:, 4:]
    y_train = data[:, 2]
    clf = RandomForestClassifier(n_estimators=n_rf_estimators)
    clf.fit(x_train, y_train)
    return clf


def create_hashtag_profile(doc_hashtags, hashtag_blacklist=None, min_score=0):
    if hashtag_blacklist is None:
        hashtag_blacklist = []
    score_sum = sum(h['score'] for h in doc_hashtags if h not in hashtag_blacklist and h['score'] >= min_score)
    hashtag_weight_tuples = [
        (h['hashtag'], h['score'] / score_sum) for h in doc_hashtags
        if h not in hashtag_blacklist and h['score'] >= min_score
    ]
    # sanity check
    if len(hashtag_weight_tuples) > 0 and not np.isclose(sum(w for (h, w) in hashtag_weight_tuples), 1, atol=1e-6):
        _logger.error("the normalization of the hashtag weights went wrong")
    return dict(hashtag_weight_tuples)


_l2r_classifier = train_rf_classifier("All", n_rf_estimators=config.CLASSIFIER_N_ESTIMATORS)
# --------------------------------------- END HASHTAG RECOMMENDATION FUNCTIONS --------------------------------------- #
