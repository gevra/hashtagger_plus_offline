import logging
import tarfile
import bz2
import json
import time
from datetime import datetime, timezone
from elasticsearch import Elasticsearch, helpers
from hashtagger_config import config
from common.text_functions import process_tweet_content

_logger = logging.getLogger(__name__)
_file_logger = logging.getLogger("features")

ES_HOST = config.ES_HOST_TWEETS
INDEX_NAME = config.ES_TWEET_INDEX_NAME
TYPE_NAME = 'Tweet'

es = Elasticsearch(hosts=[ES_HOST])
BULKSIZE = config.TWEET_BULKSIZE
BATCH_SIZE = config.TWEET_BATCH_SIZE

request_body_old = {
    "settings": {
        "similarity": {
            "my_similarity_dfr": {
                "type": "DFR",
                "basic_model": "g",
                "after_effect": "l",
                "normalization": "h2",
                "normalization.h2.c": "3.0"
            }
        },

        "analysis": {
            "analyzer": {
                "my_english": {
                    "tokenizer": "standard",
                    "filter": [
                        "lowercase",
                        "stop",
                        "kstem"  # stemmer, snowball, kstem, or porter_stem
                    ]
                },
                "hashtag_analyzer": {
                    "tokenizer": "standard",
                    "filter": [
                        "lowercase",
                    ]
                }
            }
        },
        "number_of_shards": 1,
        "number_of_replicas": 0,
    },
    "mappings": {
        TYPE_NAME: {
            "properties": {
                "tweetid": {
                    "type": "long"
                },
                "user": {
                    "type": "keyword",
                    "index": True
                },
                "tweetcontent": {
                    "type": "text",
                    "analyzer": "my_english",  # stemmer
                    "similarity": "BM25"
                },
                "tweetcontent_clean": {
                    "type": "text",
                    "analyzer": "my_english",  # stemmer
                    "similarity": "BM25"
                },
                "datetime": {
                    "type": "date"
                },
                "urls": {
                    "type": "text",
                },
                "image": {
                    "type": "text",
                },
                "mentions": {
                    "type": "text",
                    "analyzer": "my_english",  # stemmer
                    "similarity": "BM25"
                },
                "hashtags": {
                    "type": "text",
                    "analyzer": "my_english",  # stemmer
                    "similarity": "BM25"
                },
                "hashtags_list": {
                    "type": "text",
                    "analyzer": "hashtag_analyzer"
                },
                "n_hashtags": {
                    "type": "integer"
                },
                "follower": {
                    "type": "integer"
                },
                "handled": {
                    "type": "boolean"
                },
                # "rdbms_id": {
                #     "type": "long"
                # },
                "rdbms_id": {
                    "type": "keyword",
                    "index": True
                },
            }
        },
    }
}


request_body = {
    "settings": {
        "similarity": {
            "my_similarity_dfr": {
                "type": "DFR",
                "basic_model": "g",
                "after_effect": "l",
                "normalization": "h2",
                "normalization.h2.c": "3.0"
            }
        },

        "analysis": {
            "analyzer": {
                "my_english": {
                    "tokenizer": "standard",
                    "filter": [
                        "lowercase",
                        "stop",
                        "kstem"  # stemmer, snowball, kstem, or porter_stem
                    ]
                },
                "hashtag_analyzer": {
                    "tokenizer": "standard",
                    "filter": [
                        "lowercase",
                    ]
                }
            }
        },
        "number_of_shards": 1,
        "number_of_replicas": 0,
    },
    "mappings": {
        TYPE_NAME: {
            "properties": {
                "tweetid": {
                    "type": "long"
                },
                "user": {
                    "type": "keyword",
                    "index": True
                },
                "tweetcontent": {
                    "type": "text",
                    "analyzer": "my_english",  # stemmer
                    "similarity": "BM25"
                },
                "tweetcontent_clean": {
                    "type": "text",
                    "store": True,
                    "analyzer": "my_english",  # stemmer
                    "similarity": "BM25"
                },
                "datetime": {
                    "type": "date"
                },
                "urls": {
                    "type": "text",
                },
                "image": {
                    "type": "text",
                },
                "mentions": {
                    "type": "text",
                    "analyzer": "my_english",  # stemmer
                    "similarity": "BM25"
                },
                "hashtags": {
                    "type": "text",
                    "analyzer": "my_english",  # stemmer
                    "similarity": "BM25"
                },
                "hashtags_list": {
                    "type": "text",
                    "store": True,
                    "analyzer": "hashtag_analyzer"
                },
                "n_hashtags": {
                    "type": "integer"
                },
                "follower": {
                    "type": "integer"
                },
                "handled": {
                    "type": "boolean"
                },
                "raw_json": {
                    "type": "object",
                    "enabled": False
                },
            }
        },
    }
}


def update_esindex(es_docs, es_instance=es):
    bulk_data = []
    for data_dict in es_docs:
        op_dict = {
            "index": {
                "_index": INDEX_NAME,
                "_type": TYPE_NAME,
                "_id": data_dict['tweetid']
            }
        }
        bulk_data.append(op_dict)
        bulk_data.append(data_dict)

    __bulk_index__(bulk_data, bulksize=BULKSIZE, es_instance=es_instance)


def __bulk_index__(bulk_data, bulksize=BULKSIZE, es_instance=es):
    # bulk index the data
    _logger.debug("bulk indexing...")

    for i in range(0, len(bulk_data), bulksize):
        _logger.debug("writing records from %d to %d in %s es instance" % (i, i + bulksize, es_instance))
        # 60 seconds timeout
        res = es_instance.bulk(index=INDEX_NAME, body=bulk_data[i:i + bulksize], refresh=True, request_timeout=240)
        _logger.debug("took: %d, \terrors: %s, \tnumber of items: %d" % (res["took"], res["errors"], len(res["items"])))
        if res["errors"]:
            _logger.error("errors occurred with %d tweets" % len(res["items"]))
            _file_logger.error(
                "errors occurred with %s" %
                [item["index"]["_id"] for item in res["items"] if item["index"]["status"] != 200]
            )


def if_not_exist(es_instance=es, body=None, index_name=INDEX_NAME):
    if body is None:
        body = request_body
    if not es_instance.indices.exists(index_name):
        _logger.info("creating '%s' index..." % index_name)
        res = es_instance.indices.create(index=index_name, body=body)
        _logger.info(" response: '%s'" % res)


def create_new_index(es_instance=es, body=None, index_name=INDEX_NAME):
    if body is None:
        body = request_body
    if es_instance.indices.exists(index_name):
        _logger.info("deleting '%s' index..." % index_name)
        res = es_instance.indices.delete(index=index_name)
        _logger.info(" response: '%s'" % res)

    _logger.info("creating '%s' index..." % index_name)
    res = es_instance.indices.create(index=index_name, body=body)
    _logger.info(" response: '%s'" % res)


def batch_qs(qs, batch_size=BATCH_SIZE):
    """
    Returns a (start, end, total, queryset) tuple for each batch in the given
    queryset.

    Usage:
        # Make sure to order your queryset
        article_qs = Article.objects.order_by('id')
        for start, end, total, qs in batch_qs(article_qs):
            print "Now processing %s - %s of %s" % (start + 1, end, total)
            for article in qs:
                print article.body
    """
    total = qs.count()
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        yield (start, end, total, qs[start:end])


def create_tweet_dict_from_tweet(t):
    hashtags_list = ["#" + h['text'].lower() for h in t['entities']['hashtags']]
    t_dict = {
        'tweetid': t['id'],
        'user': "@" + t['user']['screen_name'],
        'follower': t['user']['followers_count'],
        'tweetcontent': t['text'],
        'tweetcontent_clean': process_tweet_content(t['text'], min_word_length=2),
        'datetime': datetime.fromtimestamp(int(int(t['timestamp_ms']) / 1000), tz=timezone.utc),
        'urls': ";".join(url['expanded_url'] if url['expanded_url'] else "" for url in t['entities'].get('urls', [])),
        'mentions': ";".join("@" + m['screen_name'].lower() for m in t['entities'].get('user_mentions', [])),
        'hashtags': ";".join(hashtags_list),
        'image': ";".join(m['media_url'] for m in t['entities'].get('media', [])),
        'epoch': t['timestamp_ms'],
        'hashtags_list': hashtags_list,
        'n_hashtags': len(hashtags_list),
        'raw_json': json.dumps(t)
    }
    return t_dict


def create_tweet_dict_from_pre_2014_08_tweet(t):
    hashtags_list = ["#" + h['text'].lower() for h in t['entities']['hashtags']]
    t_datetime = datetime.strptime(t['created_at'], "%a %b %d %H:%M:%S %z %Y")
    t_dict = {
        'tweetid': t['id'],
        'user': "@" + t['user']['screen_name'],
        'follower': t['user']['followers_count'],
        'tweetcontent': t['text'],
        'tweetcontent_clean': process_tweet_content(t['text'], min_word_length=2),
        'datetime': t_datetime,
        'urls': ";".join(url['expanded_url'] if url['expanded_url'] else "" for url in t['entities'].get('urls', [])),
        'mentions': ";".join("@" + m['screen_name'].lower() for m in t['entities'].get('user_mentions', [])),
        'hashtags': ";".join(hashtags_list),
        'image': ";".join(m['media_url'] for m in t['entities'].get('media', [])),
        'epoch': int(t_datetime.timestamp()),
        'hashtags_list': hashtags_list,
        'n_hashtags': len(hashtags_list),
        'raw_json': json.dumps(t)
    }
    return t_dict


def import_web_archive_tweets(list_of_file_paths, pre_2014_08=False, delete_old_index=False,
                              langs=None, min_n_hashtags=1,
                              indexed_tweet_ids=None, update_existing_docs=False, update_report_interval=25000):
    """
    Index tweets from WebArchive tweet collection .tar files.
    Archived tweets are available for download at https://archive.org/details/twitterstream.
    :param (list|tuple) list_of_file_paths: list of WebArchive .tar file paths
    :param (bool) pre_2014_08: tweets before 01.08.2014 miss the 'timestamp_ms' field and need to parsed differently
    :param (bool) delete_old_index: !!! if True, deletes the existing index and indexes the tweets into a fresh index !
    :param (str|list|tuple|dict) langs: list of tweet languages, matches the 'lang' field in a tweet, None for any
    :param (int) min_n_hashtags: minimum number of contained hashtags for a tweet to be indexed (default: 1)
    :param (set) indexed_tweet_ids: set of tweet ids to skip... a list will result in a huuuuge slowdown!
    :param (bool) update_existing_docs: if True, the existing tweets will be updated, otherwise skipped if False
            (ignored if *indexed_tweet_ids* is not given)
    :param (int) update_report_interval: reporting interval for the skipped/updated tweets
            (ignored if *indexed_tweet_ids* is not given)
    :return:
    """
    if pre_2014_08:
        tweet_dict_creator = create_tweet_dict_from_pre_2014_08_tweet
    else:
        tweet_dict_creator = create_tweet_dict_from_tweet
    if indexed_tweet_ids is None:
        indexed_tweet_ids = []
        if not update_existing_docs:
            _logger.warning("'update_existing_docs=False' setting will be ignored because indexed_tweet_ids==[]")
    start_time = time.time()
    _logger.info("started importing from %d tar file(s)\n" % len(list_of_file_paths))
    if delete_old_index:
        create_new_index()
    es_docs = []
    all_tweets_count = 0
    good_tweets_count = 0
    other_doc_count = 0
    existing_tweets_count = 0
    for file_path in list_of_file_paths:
        _logger.info("started importing from %s\n" % file_path.split("/")[-1])
        file_tweets_count = 0
        file_good_tweets_count = 0
        file_start_time = time.time()
        tar_file_iterator = tarfile.open(file_path, mode='r|')
        for member in tar_file_iterator:
            if member.isfile():
                if member.name[-4:] == ".bz2":
                    f = tar_file_iterator.extractfile(member)
                    try:
                        bz2_file = bz2.BZ2File(f)
                        for line in bz2_file.readlines():
                            try:
                                tweet = json.loads(line.decode('utf8'))
                            except ValueError:
                                _logger.error("couldn't read the following line:\n%s" % line.decode('utf8'))
                                continue
                            if 'id' in tweet:
                                file_tweets_count += 1
                                all_tweets_count += 1
                                # if es.exists(index=INDEX_NAME, doc_type=TYPE_NAME, id=tweet['id']):
                                if tweet['id_str'] in indexed_tweet_ids:  # the document has been indexed before, skip?
                                    existing_tweets_count += 1
                                    if existing_tweets_count % update_report_interval == 0:
                                        _logger.info(
                                            "%s %d already indexed docs by now" %
                                            ("updated" if update_existing_docs else "skipped", existing_tweets_count)
                                        )
                                    if not update_existing_docs:
                                        continue
                                try:
                                    if len(tweet['entities']['hashtags']) >= min_n_hashtags:
                                        if langs:
                                            if tweet['lang'] in langs:
                                                es_docs.append(tweet_dict_creator(tweet))
                                                good_tweets_count += 1
                                                file_good_tweets_count += 1
                                        else:
                                            es_docs.append(tweet_dict_creator(tweet))
                                            good_tweets_count += 1
                                            file_good_tweets_count += 1
                                except KeyError as e:
                                    # _logger.warning(
                                    #     "skipping tweet #%d id:%s - raised a KeyError: %s" %
                                    #     (file_tweets_count, tweet['id'], e)
                                    # )
                                    continue  # logging was too heavy on I/O because of too many tweets missing 'lang'
                                except Exception as ee:
                                    _logger.warning(
                                        "skipping tweet #%d id:%s - raised %s" % (file_tweets_count, tweet['id'], ee)
                                    )
                            else:
                                # these are usually tweet delete requests ...
                                other_doc_count += 1
                                continue
                    except EOFError as eee:
                        _logger.error("the file %s raised an EOFError: %s" % (member.name[-4:], eee))
                    except Exception as eeee:
                        _logger.error("the file %s raised %s" % (member.name[-4:], eeee))
                    if len(es_docs) >= BATCH_SIZE:
                        _logger.info(
                            ("tweets in %s & with >=%d tags \tpassed to ES: %d, "
                             "passed to ES overall: %d (from which %s: %d), "
                             "encountered in the file: %d, encountered: %d") %
                            (langs, min_n_hashtags, len(es_docs), file_good_tweets_count,
                             "updated" if update_existing_docs else "skipped", existing_tweets_count,
                             file_tweets_count, all_tweets_count)
                        )
                        update_esindex(es_docs)
                        # good_tweets_counter += len(es_docs)
                        # file_good_tweets_counter += len(es_docs)
                        es_docs = []
        if len(es_docs):  # write the remaining docs from the .bz2 file to Elasticsearch index
            _logger.info(
                ("the last batch of tweets in %s & with >=%d tags \tpassed to ES: %d, "
                 "passed to ES overall: %d (from which %s: %d), "
                 "encountered in the file: %d, encountered: %d") %
                (langs, min_n_hashtags, len(es_docs), file_good_tweets_count,
                 "updated" if update_existing_docs else "skipped", existing_tweets_count,
                 file_tweets_count, all_tweets_count)
            )
            update_esindex(es_docs)
            # good_tweets_counter += len(es_docs)
            # file_good_tweets_counter += len(es_docs)
            es_docs = []
        _logger.info(
            "finished indexing %d tweets (seen %d) from %s in %d seconds\n" %
            (file_good_tweets_count, file_tweets_count, file_path.split("/")[-1], round(time.time() - file_start_time))
        )
    _logger.info(
        "finished importing %d tweets (seen %d) from %d tar file(s) in %d seconds" %
        (good_tweets_count, all_tweets_count, len(list_of_file_paths), round(time.time() - start_time))
    )


def search_es(query=None, size=None, time_start=None, time_end=None, filter_id_list=None, min_hashtags=0,
              max_hashtags=None, _source=None, _source_exclude=None, return_random_sample_flag=False,
              return_generator=False):
    time_now = datetime.now(timezone.utc)
    if time_start is None:
        time_start = time_now.replace(year=time_now.year - 1)
    if time_end is None:
        time_end = time_now
    if filter_id_list is None:
        filter_id_list = []
    n_hashtags_range_query = {}
    if min_hashtags > 0:
        n_hashtags_range_query["gte"] = min_hashtags
    if max_hashtags:
        n_hashtags_range_query["lte"] = max_hashtags
    if query is not None or return_random_sample_flag:
        body = {
            "query": {
                "function_score": {
                    "query": {
                        "bool": {
                            "filter": {
                                "bool": {
                                    "must": [
                                        {
                                            "range": {
                                                "datetime": {
                                                    "gte": time_start.strftime("%Y-%m-%dT%H:%M:%S"),
                                                    "lte": time_end.strftime("%Y-%m-%dT%H:%M:%S"),
                                                }
                                            }
                                        }
                                    ],
                                    "must_not": [
                                        {
                                            "ids": {"type": TYPE_NAME, "values": filter_id_list}
                                        },
                                    ]
                                }
                            }
                        }
                    }
                }
            }
        }
        if query is not None:
            body["query"]["function_score"]["query"]["bool"].update(
                {"must": {"bool": query}}
            )
        if return_random_sample_flag:
            # if "random_score" is used with a filter (returns 0 scores), the "boost_mode" must be "replace" or "sum"
            body["query"]["function_score"].update({"random_score": {}, "boost_mode": "replace"})
            if size > 10000:
                _logger.warning(
                    "a random sample of tweets can have a maximum size 10000, retrieving 10000 instead of %d" % size
                )
                size = 10000
        if min_hashtags > 0 or max_hashtags:
            body["query"]["function_score"]["query"]["bool"]["filter"]["bool"]["must"].append(
                {"range": {"n_hashtags": n_hashtags_range_query}}
            )
        # _logger.debug(body)
        res_count = es.count(index=INDEX_NAME, body=body, request_timeout=120)
        if size is None:
            size = res_count['count']

        if size <= 10000:
            res = es.search(
                index=INDEX_NAME, size=size, body=body, _source=_source, _source_exclude=_source_exclude,
                request_timeout=60 + size // 1000
            )
            return res['hits']['hits'], res_count['count']
        else:
            res = helpers.scan(
                es, index=INDEX_NAME, query=body, scroll=u'5m', raise_on_error=True, preserve_order=True,
                _source=_source, _source_exclude=_source_exclude, doc_type=TYPE_NAME
            )
            # helpers.scan() returns a generator, so it needs to be converted to a list
            if not return_generator:
                tweets = [a for a in res]
                return tweets, res_count['count']
            else:
                return res, res_count['count']
    else:
        body = {
            "query": {
                "bool": {
                    "filter": {
                        "bool": {
                            "must": [
                                {
                                    "range": {
                                        "datetime": {
                                            "gte": time_start.strftime("%Y-%m-%d"),
                                            "lte": time_end.strftime("%Y-%m-%d")
                                        }
                                    }
                                }
                            ],
                            "must_not": [
                                {
                                    "ids": {"type": TYPE_NAME, "values": filter_id_list}
                                },
                            ]
                        }
                    }
                }
            }
        }
        if min_hashtags > 0 or max_hashtags:
            body["query"]["bool"]["filter"]["bool"]["must"].append({"range": {"n_hashtags": n_hashtags_range_query}})
        res_count = es.count(index=INDEX_NAME, body=body, request_timeout=120)
        res = helpers.scan(
            es, index=INDEX_NAME, query=body, scroll=u'5m', raise_on_error=True, preserve_order=True,
            _source=_source, _source_exclude=_source_exclude, doc_type=TYPE_NAME
        )
        # helpers.scan() returns a generator, so it needs to be converted to a list
        if not return_generator:
            return list(res), res_count['count']
        else:
            return res, res_count['count']


def get_by_id(list_of_ids, size=None, query=None, min_hashtags=0, max_hashtags=None, return_generator=False):
    body = {
        "query": {
            "bool": {
                "filter": [
                    {
                        "ids": {
                            "values": list_of_ids
                        }
                    }
                ]
            }
        }
    }
    if query is not None:
        body["query"]["bool"].update({"must": {"bool": query}})
    n_hashtags_range_query = {}
    if min_hashtags > 0:
        n_hashtags_range_query["gte"] = min_hashtags
    if max_hashtags:
        n_hashtags_range_query["lte"] = max_hashtags
    if min_hashtags > 0 or max_hashtags:
        body["query"]["bool"]["filter"].append({"range": {"n_hashtags": n_hashtags_range_query}})
    res_count = es.count(index=INDEX_NAME, body=body, request_timeout=30)
    if size is None:
        size = res_count['count']

    if size <= 10000:
        res = es.search(index=INDEX_NAME, size=size, body=body, request_timeout=60 + size // 1000)
        return res['hits']['hits'], res_count['count']
    else:
        res = helpers.scan(es, index=INDEX_NAME, query=body, scroll=u'5m', raise_on_error=True, preserve_order=True,
                           doc_type=TYPE_NAME)
        # helpers.scan() returns a generator, so it needs to be converted to a list
        if not return_generator:
            return list(res), res_count['count']
        else:
            return res, res_count['count']


class Object(object):
    def __init__(self):
        pass
