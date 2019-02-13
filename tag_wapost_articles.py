import logging
from bs4 import BeautifulSoup

from hashtagger_config import config
from common.text_functions import deaccent_and_replace_quotes

from functions import create_article_dict_without_profile, load_articles_from_json_lines

_logger = logging.getLogger(__name__)


########################################################################################################################
# -------------------------------------------------- WAPOST ARTICLES ------------------------------------------------- #
########################################################################################################################
def wapost_epoch_accessor(a): return int(a['published_date'] / 1000)


def create_dict_from_washingtonpost_article(a):
    if a['title'] is None:
        title = ""
    else:
        title = deaccent_and_replace_quotes(a['title'])
    body = " ".join(
        deaccent_and_replace_quotes(BeautifulSoup(c['content'], "lxml").text) for c in a['contents']
        if c and c['type'] == 'sanitized_html' and c['subtype'] == 'paragraph'
        # the check for 'c' is for e.g. "5f8dadba-0279-11e2-8102-ebee9c66e190" which has one of the c==None
        # it is safe to rely on the order of the statement evaluation in the 'if' chain
        # https://docs.python.org/3/library/stdtypes.html#boolean-operations-and-or-not
    ).strip()
    # WashingtonPost articles don't have subheadlines
    a_dict = create_article_dict_without_profile(
        a_id=a['id'], title=title, subtitle="", body=body, unix_timestamp=a['published_date'] / 1000,
        url=a['article_url'], source=a['source'], a_type=a['type'])
    return a_dict


wapost_json_path = "TREC_Washington_Post_collection_sample.jl"
test_article_ids = ['24153e8d6707e02bc16262f32bcce760', '9aa9c956-5ac2-11e6-9aee-8075993d73a2']

if __name__ == "__main__":
    _logger.info("\nloading the Washington Post articles...\n")
    load_articles_from_json_lines(
        articles_json_path=wapost_json_path, id_list=test_article_ids, article_epoch_accessor=wapost_epoch_accessor,
        article_constructor_function=create_dict_from_washingtonpost_article, id_field_name="id",
        overwrite_existing_articles=True, tag_articles_flag=True,
        export_file_name=None, export_es_instance=None,
        local_tweet_window_before=config.LOCAL_TWEET_WINDOW_BEFORE,
        local_tweet_window_after=config.LOCAL_TWEET_WINDOW_AFTER,
        global_tweet_window_before=config.GLOBAL_TWEET_WINDOW_BEFORE,
        global_tweet_window_after=config.GLOBAL_TWEET_WINDOW_AFTER,
        global_article_window_before=config.GLOBAL_ARTICLE_WINDOW_BEFORE,
        global_article_window_after=config.GLOBAL_ARTICLE_WINDOW_AFTER,
        current_global_stats_time_window_margin=config.SLIDING_GLOBAL_TIME_WINDOW_MARGIN
    )
# ----------------------------------------------- END WAPOST ARTICLES ------------------------------------------------ #
