"""
after cloning the code, one must first run
git update-index --assume-unchanged hashtagger_config.py
to tell git not to track the changes to the config file and then put the config info

git ls-files -v|grep '^h'  # to get a list of dirs/files that are 'assume-unchanged'
git update-index --no-assume-unchanged hashtagger_config.py  # to undo dirs/files that are set to assume-unchanged
"""

import logging.config
import time
# import os


class Config:
    """
    The 'default' values are not the optimal ones, but the ones used in the online version of Hashtagger+
    described in https://ieeexplore.ieee.org/abstract/document/8046071 DOI:10.1109/TKDE.2017.2754253 .

    The default values with a following '*' indicate that this value is not used in online Hashtagger+ as is,
    but is equivalent to those resulting from the periodic processes in Hashtagger+ (crawling tweets from Twitter
    every 5 minutes, assigning tweets to article every 15 minutes, tagging articles every 15 minutes).
    """
    def __init__(self):
        # for Elasticsearch
        self.ES_HOST_ARTICLE = {"host": "your_elasticsearch_host_for_article_index", "port": 9200}
        self.ES_ARTICLE_INDEX_NAME = "elasticsearch_article_index_name"
        self.ES_ARTICLE_RECOMMENDATION_MAPPING_TYPE = 'flat'  # takes two values: ['flat', 'pc'] ... 'pc' is for a legacy system from our lab
        self.ARTICLE_BULKSIZE = 100
        self.TAGOFARTICLE_BULKSIZE = 2500  # for legacy system from our lab
        self.ARTICLE_BATCH_SIZE = 200
        self.HASHTAG_BUCKET_SIZE = 20  # for legacy system from our lab
        self.ES_ARTICLE_MATCH_FIELDS = ["headline^3", "subheadline^2", "content^1"]  # this parameter is not used in hashtagger_plus_offline tool, but provides the search function great flexibility for a use in other projects
        self.ES_ARTICLE_BIGRAM_MATCH_FIELDS = ["headline^9", "subheadline^6", "content^2"]  # this parameter is not used in hashtagger_plus_offline tool, but provides the search function great flexibility for a use in other projects
        self.ES_ARTICLE_PHRASE_MATCH_FIELDS = ["headline^15", "subheadline^10", "content^5"]  # this parameter is not used in hashtagger_plus_offline tool, but provides the search function great flexibility for a use in other projects
        self.ES_TAGS_FIELDS = ["good_hashtags^6", "all_hashtags^3"]  # this parameter is not used in hashtagger_plus_offline tool, but provides the search function great flexibility for a use in other projects
        self.ES_HOST_TWEETS = {"host": "your_elasticsearch_host_for_tweet_index", "port": 9200}
        self.ES_TWEET_INDEX_NAME = "elasticsearch_tweet_index_name"
        self.TWEET_BULKSIZE = 25000
        self.TWEET_BATCH_SIZE = 50000

        # for text processing
        self.STANFORD_NER_TAGGER_SERVER = {"host": "localhost", "port": 9199}
        # self.STANFORD_CLASSPATH = 'path/to/stanford_NER_folder/stanford-ner.jar:' + \
        #                           'path/to/stanford_postagger_folder/stanford-postagger.jar:'
        # self.STANFORD_MODELS = 'path/to/stanford_NER_folder/classifiers:'
        # alternatively set environment variables as described at https://stackoverflow.com/a/34112695/2262424

        # for preprocessing the articles
        self.ADDITIONAL_STOPWORDS = [
            ".", "...", "a", "able", "about", "across", "after", "all", "almost", "also", "although", "am", "among",
            "an", "and", "any", "anyone", "are", "as", "at", "bbc", "be", "because", "been", "best", "but", "by", "can",
            "cannot", "cent", "could", "dear", "did", "do", "does", "dont", "either", "else", "ever", "every", "finds",
            "first", "for", "former", "from", "get", "got", "had", "has", "have", "having", "he", "her", "hers", "him",
            "his", "how", "however", "huge", "i", "if", "image", "images", "immense", "in", "inside", "into", "is",
            "it", "its", "just", "large", "latest", "least", "let", "like", "likely", "may", "me", "might", "most",
            "must", "my", "neither", "new", "news", "no", "nor", "not", "of", "off", "often", "on", "only", "or",
            "other", "our", "own", "per", "proud", "rather", "re", "regardless", "rt", "said", "say", "says", "she",
            "should", "simple", "since", "so", "some", "still", "than", "that", "the", "their", "them", "then", "there",
            "these", "they", "this", "tis", "to", "today", "too", "twas", "us", "ve", "video", "videos", "want",
            "wants", "was", "we", "went", "were", "were", "what", "when", "where", "which", "while", "who", "whom",
            "why", "will", "with", "would", "yes", "yet", "you", "your"
        ]
        self.MAX_N_ARTICLE_KEYWORDS = 5  # default: 5 # maximum number of keywords to extract from a pseudoarticle
        self.NLTK_DATA_PATH = None
        self.NAMED_ENTITY_BOOST = 2
        self.TFIDF_VECTORIZER_MIN_DF = 2  # default: 2 # used in coldstart data TFIDF vectorization
        self.TFIDF_VECTORIZER_MAX_DF = 0.5  # default: 0.5 # used in coldstart data TFIDF vectorization

        # for L2R classifier
        self.CLASSIFIER_DATA_TRAIN_FILE = "./Data/User_Label_14_Training.csv"  # training data for L2R classifier
        self.CLASSIFIER_DATA_ALL_FILE = "./Data/User_Label_14_N_B.csv"  # training data for L2R classifier
        self.CLASSIFIER_N_ESTIMATORS = 100  # default: 100 # number of RandomForestClassifier estimators for L2R

        # for global time window stats and data
        self.SLIDING_GLOBAL_TIME_WINDOW_MARGIN = 24 * 3600  # default: 15 minutes* # the tolerance of the global window shift w.r.t. an article
        self.GLOBAL_TWEET_WINDOW_BEFORE = 4 * 24 * 3600  # default: 6 hours # used to define the global tweet window w.r.t. an article
        self.GLOBAL_TWEET_WINDOW_AFTER = 4 * 24 * 3600  # default: 24 hours* # used to define the global tweet window w.r.t. an article
        self.GLOBAL_TWEET_SAMPLE_SIZE = 25000  # default: 10000 # sample size of tweets (.order_by('?') in django) in the global window, it needs to be big not for stats but for enough tweets per hashtag
        self.GLOBAL_TWEET_SAMPLE_RANDOM_FLAG = True  # default: True # defines whether self.GLOBAL_TWEET_SAMPLE_SIZE are random or not, the maximum size for a random sample is 10000
        self.GLOBAL_HASHTAG_TWEET_SAMPLE_SIZE = None  # default: 5000 # sample size of tweets containing a hashtag in the global time window
        # for article profiles which are computed once during each article creation (updated in every global window)
        self.GLOBAL_ARTICLE_WINDOW_BEFORE = 24 * 3600  # default: 24 hours # used to define the global article window w.r.t. an article for computing article profile IDF
        self.GLOBAL_ARTICLE_WINDOW_AFTER = 1 * 3600  # default: 0 # used to define the global article window w.r.t. an article for computing article profile IDF
        # for coldstart (used once in every global window)
        self.COLDSTART_FLAG = True  # default: True # turns on/off the coldstart procedure
        # self.COLDSTART_ARTICLE_MAX_N_TWEETS = 100  # default: 100 # maximum number of tweets for an article to be helped with coldstart
        self.COLDSTART_ARTICLE_WINDOW_BEFORE = 60 * 24 * 3600  # default: 60 days # used in coldstart kNN training article selection
        self.COLDSTART_ARTICLE_WINDOW_AFTER = 0 * 3600  # default: 0 # used in coldstart kNN training article selection
        self.COLDSTART_KNN_N_TRAINING_NEIGHBOURS = 10  # default: 10 # used in coldstart kNN training
        self.COLDSTART_N_NEIGHBOURS = 6  # default: 6 # used in coldstart kNN nearest neighbour selection
        self.COLDSTART_N_TWEETS_PER_NEIGHBOUR_ARTICLE = 1000  # default: 1000 # used in coldstart post-kNN per article most recent tweet selection
        self.COLDSTART_TWEET_FIELD_NAME = "tweetcontent_clean"  # default: "tweetcontent_clean" # the name of the field to which the "stream_keywords" n-grams will be matched to
        self.COLDSTART_TWEET_NGRAM_MATCH_MODE = "must"  # default: "must" # "must" means all the n-gram terms must appear in a tweet in any order, "should" means any of the terms must appear in any order

        # for local time window tweet params
        self.LOCAL_TWEET_WINDOW_BEFORE = 2 * 24 * 3600  # default: 24 hours # used to define the overall article "tracking" time
        self.LOCAL_TWEET_WINDOW_AFTER = 2 * 24 * 3600  # default: 24 hours* # used to define the overall article "tracking" time
        self.LOCAL_TWEET_SAMPLE_SIZE = 500  # default: 5000 # sample size of the article tweets for local window feature computation
        self.LOCAL_TWEET_SAMPLE_TYPE = "coldstart+elbow+random"  # default: "random" # takes values ["random", "top", "elbow", "coldstart+elbow", "coldstart+elbow+random"], if "elbow" and the actual number of tweets > 2 * SAMPLE_SIZE the SAMPLE_SIZE will be ignored
        self.LOCAL_TWEET_MAX_N_HASHTAGS = 3  # default: 3 # (lte) used as a tweet filtering criterion in coldstart and main search for articles
        self.LOCAL_TWEET_MIN_N_HASHTAGS = 1  # default: 1 # (gte) used as a tweet filtering criterion in coldstart and main search for articles
        self.LOCAL_TWEET_MIN_N_TOKENS = 4  # default: 4 # (gte) used as a tweet filtering criterion in coldstart and main search for articles
        self.LOCAL_TWEET_FIELD_NAME = "tweetcontent_clean"  # default: "tweetcontent_clean" # the name of the field to which the "stream_keywords" n-grams will be matched to
        self.LOCAL_TWEET_NGRAM_MATCH_MODE = "must"  # default: "must" # "must" means all the n-gram terms must appear in a tweet in any order, "should" means any of the terms must appear in any order
        self.LOCAL_MIN_N_ARTICLE_TWEETS_PER_TAG = 3  # default 3 # used to filter the prospective hashtags which are not frequent enough
        self.LOCAL_MIN_N_ARTICLE_TWEETS_PER_TAG_RELATIVE_TO_MAX = 0.2  # default: 1/5 # used to filter the prospective hashtags which are not frequent enough relative to the most frequent hashtag within the article tweet set
        self.MIN_WORD_LENGTH = 2  # default: 2 # used to compute hashtag's local tweet profile, which is used in local time window features' computation
        self.TOP_WORDS_LIMIT = 20  # default: 20 # used to compute hashtag's local tweet profile, which is used in local time window features' computation
        self.HASHTAG_WINDOW_TWEET_SAMPLE_SIZE = None  # default: 100 # used to compute hashtag's local tweet profile, which is used in local time window features' computation

        # for hashtag recommendation
        self.N_KEYWORDS_FOR_TWEET_MATCH = 10  # default: 5 # the number of 'streaming_keywords' n-grams used to match tweets to articles (also in coldstart)
        self.RECOMMENDATION_CONF_THRES = 0.4  # default: 0.5 # hashtag's minimum L2R classifier score for an article
        self.HASHTAG_LIMIT = 15  # default: 10 # maximum number of most relevant (ranked by L2R classifier score) hashtags per article
        self.BURSTINESS_INTERVAL = 30 * 60  # default: 30 minutes # used to compute the 'trending' feature

        self.HASHTAG_PROFILE_MIN_SCORE = 0.5  # the lower threshold of a tag's score given by the L2R classifier
        self.HASHTAG_BLACKLIST = [
            "#news", "#business", "#breaking", "#politics", "#jobs", "#world", "#rt", "#sport", "#breakingnews",
            "#follow", "#new", "#update", "#bbc"
        ]


config = Config()

# set environment variables (for Stanford NLP tools) as described in https://stackoverflow.com/a/34112695/2262424
# os.environ['CLASSPATH'] = config.STANFORD_CLASSPATH
# os.environ['STANFORD_MODELS'] = config.STANFORD_MODELS

# docs are here https://docs.python.org/3/library/logging.config.html#logging-config-dictschema
logging.config.dictConfig(
    {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'verbose': {
                'format': '%(levelname)s %(asctime)s %(module)s %(process)d %(thread)d %(message)s'
            },
            'simple': {
                'format': '%(asctime)s %(relativeCreated)d %(levelname)5s %(name)s : %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'short': {
                'format': '%(asctime)s %(levelname)5s %(name)s : %(message)s',
                'datefmt': '%H:%M:%S'
            },
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'formatter': 'short',
            },
            'file': {
                'class': 'logging.FileHandler',
                'filename': 'logs/hashtagger_offline_features_%d.log' % int(time.time()),
                'mode': 'a',
                'formatter': 'short',
            }
        },
        'loggers': {
            'hashtagger_plus_offline': {
                'handlers': ['console'],
                'level': 'DEBUG',
                'propagate': True,
            },
            'functions': {
                'handlers': ['console'],
                'level': 'DEBUG',
                'propagate': False,
            },
            'es_article': {
                'handlers': ['console'],
                'level': 'DEBUG',
                'propagate': False,
            },
            'es_tweet': {
                'handlers': ['console'],
                'level': 'DEBUG',
                'propagate': False,
            },
            'common': {
                'handlers': ['console'],
                'level': 'DEBUG',
                'propagate': False,
            },
            'features': {
                'handlers': ['file'],
                'level': 'DEBUG',
                'propagate': False,
            },
        },
    }
)
