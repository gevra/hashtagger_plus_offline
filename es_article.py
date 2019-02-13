import logging
import re
import time
import datetime
from pytz import timezone
from elasticsearch import Elasticsearch, helpers
from hashtagger_config import config
from common.functions import sliding_window


_logger = logging.getLogger(__name__)

ES_HOST = config.ES_HOST_ARTICLE
INDEX_NAME = config.ES_ARTICLE_INDEX_NAME
DOC_TYPE_NAME = "doc"
TYPE_NAME = 'Article'
TYPE_CHILD_TAGS = 'TagOfArticle'
JOIN_NAME = 'article2hashtag'

es = Elasticsearch(hosts=[ES_HOST])
HASHTAG_BUCKET_SIZE = config.HASHTAG_BUCKET_SIZE
BULKSIZE = config.ARTICLE_BULKSIZE
TOABULKSIZE = config.TAGOFARTICLE_BULKSIZE
BATCH_SIZE = config.ARTICLE_BATCH_SIZE


# upgrading to ES 6.2.4 was facilitated by http://kimjmin.net/2018/01/2018-01-parent-child-to-join/ the docs and also
# https://stackoverflow.com/questions/47713400/having-trouble-creating-parent-child-relationship-in-elasticsearch-6
request_body_pc = {
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
                "my_english_folding": {
                    "tokenizer": "standard",
                    "filter": [
                        "lowercase",
                        "stop",
                        "kstem",  # stemmer, snowball, kstem, or porter_stem
                        "asciifolding"
                    ]
                },
                "hashtag_analyzer": {
                    "tokenizer": "standard",
                    "filter": [
                        "lowercase",
                    ]
                },
                "raw_analyzer": {
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
        DOC_TYPE_NAME: {
            "properties": {
                JOIN_NAME: {
                    "type": "join",
                    "relations": {
                        TYPE_NAME: TYPE_CHILD_TAGS
                    }
                },
                "id": {
                    "type": "integer"
                },
                "headline": {
                    "type": "text",
                    "analyzer": "my_english_folding",  # stemmer
                    "similarity": "BM25",
                    "fields": {
                        "nonfolded": {
                            "type": "text",
                            "analyzer": "my_english",
                            "similarity": "BM25"
                        }
                    }
                },
                "subheadline": {
                    "type": "text",
                    "analyzer": "my_english_folding",  # stemmer
                    "similarity": "BM25",
                    "fields": {
                        "nonfolded": {
                            "type": "text",
                            "analyzer": "my_english",
                            "similarity": "BM25"
                        }
                    }
                },
                "url": {
                    "type": "text",
                    "analyzer": "my_english",  # stemmer
                    "similarity": "BM25",
                    "fields": {
                        "raw": {
                            "type": "keyword",
                            "index": "true"
                        }
                    }
                },
                "keywords": {
                    "type": "text",
                    "analyzer": "my_english_folding",  # stemmer
                    "similarity": "BM25",
                    "fields": {
                        "nonfolded": {
                            "type": "text",
                            "analyzer": "my_english",
                            "similarity": "BM25"
                        }
                    }
                },
                'stream_keywords': {
                    "type": "text",
                    "analyzer": "my_english_folding",  # stemmer
                    "similarity": "BM25",
                    "fields": {
                        "nonfolded": {
                            "type": "text",
                            "analyzer": "my_english",
                            "similarity": "BM25"
                        }
                    }
                },
                "content": {
                    "type": "text",
                    "analyzer": "my_english_folding",  # stemmer
                    "similarity": "BM25",
                    "fields": {
                        "nonfolded": {
                            "type": "text",
                            "analyzer": "my_english",
                            "similarity": "BM25"
                        }
                    }
                },
                "source": {
                    "type": "keyword",
                    "index": True
                },
                "datetime": {
                    "type": "date"
                },
                "numbertweets": {
                    "type": "integer"
                },
                "updatedatetime": {
                    "type": "date"
                },
                "havehashtag": {
                    "type": "boolean"
                },
                "first_sentence": {
                    "type": "text",
                    "analyzer": "my_english_folding",  # stemmer
                    "similarity": "BM25",
                    "fields": {
                        "nonfolded": {
                            "type": "text",
                            "analyzer": "my_english",
                            "similarity": "BM25"
                        }
                    }
                },
                "all_hashtags": {
                    "type": "text",
                    "analyzer": "hashtag_analyzer"
                },
                "hashtag_profile": {
                    "type": "keyword",
                    "index": True
                },
                "good_hashtags": {
                    "type": "text",
                    "analyzer": "hashtag_analyzer"
                },
                "n_hashtags": {
                    "type": "integer"
                },
                "n_good_hashtags": {
                    "type": "integer"
                },
                "nes": {
                    "type": "text",
                    "analyzer": "my_english_folding",  # stemmer
                    "similarity": "BM25",
                    "fields": {
                        "nonfolded": {
                            "type": "text",
                            "analyzer": "my_english",
                            "similarity": "BM25"
                        }
                    }
                },
                "noun_tokens": {
                    "type": "text",
                    "analyzer": "raw_analyzer",  # no stemming
                    "similarity": "BM25",
                },
                "tokens": {
                    "type": "text",
                    "analyzer": "raw_analyzer",  # no stemming
                    "similarity": "BM25",
                },
                "unique_tokens": {
                    "type": "text",
                    "analyzer": "raw_analyzer",  # no stemming
                    "similarity": "BM25",
                },
                "processed_pseudoarticle": {
                    "type": "text",
                    "analyzer": "raw_analyzer",  # no stemming
                    "similarity": "BM25",
                },
                "unique_id": {
                    "type": "keyword",
                    "index": True
                },
                # here come the child type fields
                "rdbms_tag_id": {"type": "integer"},  # the same "id" will NOT be used to avoid clashes as both are int
                # "updatedatetime" was already defined before or Article
                "non_empty_tag_numbers": {
                    "type": "keyword",
                    "index": True
                },

            }
        },
        # TYPE_CHILD_TAGS: {
        #     "_parent": {"type": TYPE_NAME},
        #     "properties": hashtagProperties
        # }
    }
}

request_body_flat = {
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
                "my_english_folding": {
                    "tokenizer": "standard",
                    "filter": [
                        "lowercase",
                        "stop",
                        "kstem",  # stemmer, snowball, kstem, or porter_stem
                        "asciifolding"
                    ]
                },
                "hashtag_analyzer": {
                    "tokenizer": "standard",
                    "filter": [
                        "lowercase",
                    ]
                },
                "raw_analyzer": {
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
                "id": {
                    "type": "keyword"
                },
                "headline": {
                    "type": "text",
                    "analyzer": "my_english_folding",  # stemmer
                    "similarity": "BM25",
                    "fields": {
                        "nonfolded": {
                            "type": "text",
                            "analyzer": "my_english",
                            "similarity": "BM25"
                        }
                    }
                },
                "subheadline": {
                    "type": "text",
                    "analyzer": "my_english_folding",  # stemmer
                    "similarity": "BM25",
                    "fields": {
                        "nonfolded": {
                            "type": "text",
                            "analyzer": "my_english",
                            "similarity": "BM25"
                        }
                    }
                },
                "url": {
                    "type": "text",
                    "analyzer": "my_english",  # stemmer
                    "similarity": "BM25",
                    "fields": {
                        "raw": {
                            "type": "keyword",
                            "index": "true"
                        }
                    }
                },
                "keywords": {
                    "type": "text",
                    "analyzer": "my_english_folding",  # stemmer
                    "similarity": "BM25",
                    "fields": {
                        "nonfolded": {
                            "type": "text",
                            "analyzer": "my_english",
                            "similarity": "BM25"
                        }
                    }
                },
                'stream_keywords': {
                    "type": "text",
                    "analyzer": "my_english_folding",  # stemmer
                    "similarity": "BM25",
                    "fields": {
                        "nonfolded": {
                            "type": "text",
                            "analyzer": "my_english",
                            "similarity": "BM25"
                        }
                    }
                },
                "content": {
                    "type": "text",
                    "analyzer": "my_english_folding",  # stemmer
                    "similarity": "BM25",
                    "fields": {
                        "nonfolded": {
                            "type": "text",
                            "analyzer": "my_english",
                            "similarity": "BM25"
                        }
                    }
                },
                "source": {
                    "type": "keyword",
                    "index": True
                },
                 "profile": {
                    "type": "object",
                    "enabled": False
                },
                "datetime": {
                    "type": "date"
                },
                "numbertweets": {
                    "type": "integer"
                },
                "havehashtag": {
                    "type": "boolean"
                },
                "first_sentence": {
                    "type": "text",
                    "analyzer": "my_english_folding",  # stemmer
                    "similarity": "BM25",
                    "fields": {
                        "nonfolded": {
                            "type": "text",
                            "analyzer": "my_english",
                            "similarity": "BM25"
                        }
                    }
                },
                "recommendations": {
                    "type": "object",
                    "enabled": False
                },
                "all_hashtags": {
                    "type": "text",
                    "analyzer": "hashtag_analyzer"
                },
                "hashtag_profile": {
                    "type": "keyword",
                    "index": True
                },
                "good_hashtags": {
                    "type": "text",
                    "analyzer": "hashtag_analyzer"
                },
                "n_hashtags": {
                    "type": "integer"
                },
                "n_good_hashtags": {
                    "type": "integer"
                },
                "nes": {
                    "type": "text",
                    "analyzer": "my_english_folding",  # stemmer
                    "similarity": "BM25",
                    "fields": {
                        "nonfolded": {
                            "type": "text",
                            "analyzer": "my_english",
                            "similarity": "BM25"
                        }
                    }
                },
                "noun_tokens": {
                    "type": "text",
                    "analyzer": "raw_analyzer",  # no stemming
                    "similarity": "BM25",
                },
                "tokens": {
                    "type": "text",
                    "analyzer": "raw_analyzer",  # no stemming
                    "similarity": "BM25",
                },
                "unique_tokens": {
                    "type": "text",
                    "analyzer": "raw_analyzer",  # no stemming
                    "similarity": "BM25",
                },
                "processed_pseudoarticle": {
                    "type": "text",
                    "analyzer": "raw_analyzer",  # no stemming
                    "similarity": "BM25",
                },
                "unique_id": {
                    "type": "keyword",
                    "index": True
                },
            }
        },
    }
}

if config.ES_ARTICLE_RECOMMENDATION_MAPPING_TYPE == 'flat':
    DOC_TYPE_NAME = TYPE_NAME  # this is fo the fields using DOC_TYPE_NAME later in the code
    request_body = request_body_flat
elif config.ES_ARTICLE_RECOMMENDATION_MAPPING_TYPE == 'pc':
    request_body = request_body_pc
    for ii in range(1, HASHTAG_BUCKET_SIZE + 1):
        request_body["mappings"][DOC_TYPE_NAME]["properties"]["tag" + str(ii)] = {
            "type": "text", "analyzer": "hashtag_analyzer"
        }
else:
    raise Exception("the article-recommendation mapping is supported for 'flat' and 'pc' modes only")


def update_esindex(es_docs, es_instance=es):
    bulk_data = []
    for data_dict in es_docs:
        op_dict = {
            "index": {
                "_index": INDEX_NAME,
                "_type": DOC_TYPE_NAME,  # TYPE_NAME,
                "_id": data_dict['id']
            }
        }
        data_dict[JOIN_NAME] = {"name": TYPE_NAME}
        bulk_data.append(op_dict)
        bulk_data.append(data_dict)

    __bulk_index__(bulk_data, bulksize=BULKSIZE, es_instance=es_instance)


def __bulk_index__(bulk_data, bulksize=BULKSIZE, es_instance=es):
    # bulk index the data
    _logger.info("bulk indexing...")

    for i in range(0, len(bulk_data), bulksize):
        _logger.info("writing records from %d to %d in %s es instance" % (i, i + bulksize, es_instance))
        try:
            res = es_instance.bulk(index=INDEX_NAME, body=bulk_data[i:i + bulksize], refresh=True, request_timeout=120)
        except (ConnectionError, ConnectionResetError) as e:
            _logger.debug("\n%s : %s" % (datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'), e))
            for try_again in range(5):
                time.sleep(5)
                _logger.debug("trying to index these docs again in the very same way after 5 sec delay")
                try:
                    res = es_instance.bulk(
                        index=INDEX_NAME, body=bulk_data[i:i + bulksize], refresh=True, request_timeout=120
                    )
                except Exception as e:
                    _logger.debug(
                        "trying again didn't work...\n%s : %s" %
                        (datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'), e)
                    )
                    _logger.debug("the number of docs is: %d" % len(bulk_data))
                    if len(bulk_data):
                        _logger.debug("the first doc in the batch is %s" % bulk_data[0])
        except Exception as e:
            _logger.debug("\n%s : the error is %s" % (datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'), e))
            for try_again in range(5):
                time.sleep(5)
                _logger.debug("trying to index these docs again in the very same way after 5 sec delay")
                try:
                    res = es_instance.bulk(
                        index=INDEX_NAME, body=bulk_data[i:i + bulksize], refresh=True, request_timeout=120
                    )
                except Exception as e:
                    _logger.debug(
                        "trying again didn't work...\n%s : %s" %
                        (datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'), e)
                    )
                    _logger.debug("the number of docs is: %d" % len(bulk_data))
                    if len(bulk_data):
                        _logger.debug("the first doc in the batch is %s" % bulk_data[0])
            if len(bulk_data):
                _logger.debug("the first doc in the batch is %s" % bulk_data[0])
            _logger.debug("the failed doc ids are written to file")
            with open("failed_import_docs_ids_%s.txt" % datetime.datetime.utcnow().strftime('%H_%M_%S'), "w") as f:
                list_of_ids_in_the_bulk = []
                for doc in bulk_data:
                    if 'id' in doc.keys():
                        list_of_ids_in_the_bulk.append(doc['id'])
                f.write(str(list_of_ids_in_the_bulk))
        # 60 seconds timeout
        # print(res)


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


def search_es(query=None, hashtag_query=None, query_not=None, hashtag_query_not=None,
              require_all_terms_to_occur_flag=False, tags_fields=None,
              article_match_fields=None, article_bigram_match_fields=None, article_phrase_match_fields=None,
              size=None, time_start=None, time_end=None, filter_id_list=None, sources=None, recency_ranking=True,
              must_have_profile=False, only_tagged=False, _source=None, _source_exclude=None, return_generator=False):
    if filter_id_list is None:
        filter_id_list = []
    if time_start is None:
        time_start = timezone('UTC').localize(datetime.datetime.utcnow() - datetime.timedelta(days=30))
    if time_end is None:
        time_end = timezone('UTC').localize(datetime.datetime.utcnow())
    time_end_next_day = time_end + datetime.timedelta(days=1)
    if query is not None:
        if hashtag_query is None:
            tags_lst = re.findall(r'#\w*', query)
            query = " ".join([term for term in query.split(" ") if term not in tags_lst])
            # for tag in tags_lst:
            #     query = query.replace(tag, '')  # this seems wrong as #mu will spoil #mufc !!!
            txt_query = query
            hashtag_query = " ".join(tags_lst)
            _logger.debug(
                "hashtag_query==None and query is '%s'... now 'hashtag_query'='%s'" % (query, hashtag_query)
            )
        else:
            txt_query = query
            _logger.debug("hashtag_query!=None and query is '%s'" % query)
    else:
        txt_query = ""
    if article_match_fields is None:
        article_match_fields = config.ES_ARTICLE_MATCH_FIELDS
    if article_bigram_match_fields is None:
        article_bigram_match_fields = config.ES_ARTICLE_BIGRAM_MATCH_FIELDS
    if article_phrase_match_fields is None:
        article_phrase_match_fields = config.ES_ARTICLE_PHRASE_MATCH_FIELDS
    if tags_fields is None:
        tags_fields = config.ES_TAGS_FIELDS

    if (query is not None) or (hashtag_query is not None):
        if len(txt_query.split()) > 1 and require_all_terms_to_occur_flag:
            # this means that all the query terms must appear in at least one of the fields
            # this is way too restrictive for story summarization retrieval, but may be handy for a safe query expansion
            # this makes sense especially when the query is matched only on headlines or pseudoarticles
            # in this case it makes sense not to elbow-cut the articles
            parent_query_dict = {
                "bool": {
                    "must": []
                }
            }
            for term in txt_query.split():
                parent_query_dict["bool"]["must"].append(
                    {
                        "multi_match": {
                            "query": term,
                            "type": "most_fields",
                            "fields": article_match_fields
                        }
                    }
                )
        else:
            parent_query_dict = {
                "multi_match": {
                    "query": txt_query,
                    "type": "most_fields",
                    "fields": article_match_fields
                }
            }

        if config.ES_ARTICLE_RECOMMENDATION_MAPPING_TYPE == 'pc':
            tags_fields = [
                "tag" + str(i + 1) + "^" + str(6.0 - (2 / HASHTAG_BUCKET_SIZE) * i) for i in range(HASHTAG_BUCKET_SIZE)
            ]
            children_query_dict = {
                "has_child": {
                    "type": TYPE_CHILD_TAGS,
                    "score_mode": "sum",
                    "query": {
                        "multi_match": {
                            "query": hashtag_query,
                            "type": "most_fields",
                            "fields": tags_fields
                        }
                    }

                }
            }
        elif config.ES_ARTICLE_RECOMMENDATION_MAPPING_TYPE == 'flat':
            children_query_dict = {
                "multi_match": {
                    "query": hashtag_query,
                    "type": "most_fields",
                    "fields": tags_fields
                }
            }
        else:
            raise Exception("the article-recommendation mapping is supported for 'flat' and 'pc' modes only")

        exact_phrase_dicts = []
        if len(txt_query.split()) > 1:
            # create a phrase match query for each bigram
            for bigram_tuple in sliding_window(txt_query.split(), 2):
                exact_phrase_dicts.append(
                    {
                        "multi_match": {
                            "query": " ".join(bigram_tuple),
                            "type": "phrase",
                            "fields": article_bigram_match_fields
                        }
                    }
                )
            if len(txt_query.split()) > 2:
                # reward the full match even higher
                exact_phrase_dicts.append(
                    {
                        "multi_match": {
                            "query": txt_query,
                            "type": "phrase",
                            "fields": article_phrase_match_fields
                        }
                    }
                )

        body = {
            "query": {
                "function_score": {
                    "query": {
                        "bool": {
                            "must": {
                                "bool": {"should": [parent_query_dict, children_query_dict, *exact_phrase_dicts]}
                            },
                            "filter": {
                                "bool": {
                                    "must": [
                                        {
                                            "range": {
                                                "datetime": {
                                                    "gte": time_start.strftime("%Y-%m-%dT%H:%M:%S"),
                                                    "lte": time_end_next_day.strftime("%Y-%m-%dT%H:%M:%S"),
                                                }
                                            }
                                        }
                                    ],
                                    "must_not": [
                                        {
                                            "ids": {"type": DOC_TYPE_NAME, "values": filter_id_list}
                                        },
                                    ]
                                }
                            }
                        }
                    },
                    # "gauss": {
                    #     "datetime": {
                    #         "scale": "25d",
                    #         "decay": 0.5
                    #     }
                    # }
                }
            }
        }
        if recency_ranking:
            body["query"]["function_score"].update({
                "gauss": {
                    "datetime": {
                        "scale": "30d",
                        "decay": 0.5  # the bigger, the faster is the decay
                    }
                }
            })
        if sources is not None:
            body["query"]["function_score"]["query"]["bool"]["filter"]["bool"]["must"].append(
                {"terms": {"source": sources}}
            )
            # print(body)
        if must_have_profile:
            body["query"]["function_score"]["query"]["bool"]["filter"]["bool"]["must_not"].append(
                {"term": {"hashtag_profile": "{}"}}
            )
            # note that "havehashtag"=True is a weaker condition than hashtag_profile!={}
        else:
            if only_tagged:
                body["query"]["function_score"]["query"]["bool"]["filter"]["bool"]["must"].append(
                    {"term": {"havehashtag": True}}
                )
            # print("search body is  %s" % body)
        if query_not is not None:
            body["query"]["function_score"]["query"]["bool"]["filter"]["bool"]["must_not"].append(
                {"multi_match": {"query": query_not, "fields": [f.split("^")[0] for f in article_match_fields]}}
            )
        if hashtag_query_not is not None:
            body["query"]["function_score"]["query"]["bool"]["filter"]["bool"]["must_not"].append(
                {"multi_match": {"query": hashtag_query_not, "fields": ['good_hashtags', 'all_hashtags']}}
            )

        # res = es.search(index=INDEX_NAME, size=size, body=body, request_timeout=60)
        res_count = es.count(index=INDEX_NAME, body=body, request_timeout=120)
        if size is None:
            size = res_count['count']

        if size <= 10000:
            res = es.search(
                index=INDEX_NAME, size=size, body=body, _source=_source, _source_exclude=_source_exclude,
                request_timeout=60 + size // 50
            )
            if size == 0:
                # return res['hits']['total'], res_count['count']
                # the line above will retrieve (0, 0) in case if no limit was set and there were no articles to retrieve
                # and that will crash the code in later stages
                return [], res_count['count']
            else:
                return res['hits']['hits'], res_count['count']
        else:
            _logger.info("there are %d documents matching the query, will try to get them all :)" % res_count['count'])
            res = helpers.scan(
                es, index=INDEX_NAME, query=body, scroll=u'5m', raise_on_error=True, preserve_order=True,
                doc_type=DOC_TYPE_NAME, _source=_source, _source_exclude=_source_exclude,
                request_timeout=70 + 3 * (size // 100)
            )
            # helpers.scan() returns a generator, so it needs to be converted to a list
            if not return_generator:
                articles = [a for a in res]
                return articles, res_count['count']
            else:
                return res, res_count['count']
        # print('Query on ES (txt): ' + txt_query + ', ' + ', '.join(article_match_fields))
        # print('Query on ES (hashtag): ' + hashtag_query + ', ' + ', '.join(tags_fields))
        # print(res['hits']['hits'][0])
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
                                            "lte": time_end_next_day.strftime("%Y-%m-%d")
                                        }
                                    }
                                }
                            ],
                            "must_not": [
                                {
                                    "ids": {"type": DOC_TYPE_NAME, "values": filter_id_list}
                                },
                            ]
                        }
                    }
                }
            }
        }
        if sources is not None:
            body["query"]["bool"]["filter"]["bool"]["must"].append(
                {"terms": {"source": sources}}
            )
        if query_not is not None:
            body["query"]["bool"]["filter"]["bool"]["must_not"].append(
                {"multi_match": {"query": query_not, "fields": [f.split("^")[0] for f in article_match_fields]}}
            )
        if hashtag_query_not is not None:
            body["query"]["bool"]["filter"]["bool"]["must_not"].append(
                {"multi_match": {"query": hashtag_query_not, "fields": ['good_hashtags', 'all_hashtags']}}
            )
        if must_have_profile:  # note that "havehashtag"=True is a weaker condition than hashtag_profile!={}
            body["query"]["bool"]["filter"]["bool"]["must_not"].append(
                {"term": {"hashtag_profile": "{}"}}
            )
        else:
            if only_tagged:
                body["query"]["bool"]["filter"]["bool"]["must"].append(
                    {"term": {"havehashtag": True}}
                )

        res_count = es.count(index=INDEX_NAME, body=body, request_timeout=120)
        res = helpers.scan(
            es, index=INDEX_NAME, query=body, scroll=u'5m', raise_on_error=True, preserve_order=True,
            _source=_source, _source_exclude=_source_exclude, doc_type=DOC_TYPE_NAME
        )
        if not return_generator:  # helpers.scan() returns a generator, so it needs to be converted to a list
            articles = [a for a in res]
            return articles, res_count['count']
        else:
            return res, res_count['count']


def get_by_id(list_of_ids, size=None, query=None, return_generator=False):
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


# if_not_exist()

class Object(object):
    pass
