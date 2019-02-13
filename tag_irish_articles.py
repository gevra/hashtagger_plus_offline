import logging

from hashtagger_config import config
from common.text_functions import deaccent_and_replace_quotes

from functions import create_article_dict_without_profile, load_articles_from_json_lines

_logger = logging.getLogger(__name__)


########################################################################################################################
# -------------------------------------------------- IRISH ARTICLES -------------------------------------------------- #
########################################################################################################################
def irish_epoch_accessor(a): return a['epoch']


def create_dict_from_irish_article(a):
    title = deaccent_and_replace_quotes(a['headline'])
    subtitle = deaccent_and_replace_quotes(a['subheadline'])
    body = deaccent_and_replace_quotes(a['content']).strip()
    a_dict = create_article_dict_without_profile(
        a_id=a['id'], title=title, subtitle=subtitle, body=body, unix_timestamp=a['epoch'],
        url=a['url'], source=a['source'], a_type=a['type'])
    return a_dict


_irish_json_path = "irish_articles_sample.json"
if __name__ == "__main__":
    _logger.info("\nloading the irish articles...\n")
    load_articles_from_json_lines(
        articles_json_path=_irish_json_path, id_list=None, article_epoch_accessor=irish_epoch_accessor,
        article_constructor_function=create_dict_from_irish_article, id_field_name="id",
        overwrite_existing_articles=False, tag_articles_flag=True,
        export_file_name=None, export_es_instance=None,
        local_tweet_window_before=config.LOCAL_TWEET_WINDOW_BEFORE,
        local_tweet_window_after=config.LOCAL_TWEET_WINDOW_AFTER,
        global_tweet_window_before=config.GLOBAL_TWEET_WINDOW_BEFORE,
        global_tweet_window_after=config.GLOBAL_TWEET_WINDOW_AFTER,
        global_article_window_before=config.GLOBAL_ARTICLE_WINDOW_BEFORE,
        global_article_window_after=config.GLOBAL_ARTICLE_WINDOW_AFTER,
        current_global_stats_time_window_margin=config.SLIDING_GLOBAL_TIME_WINDOW_MARGIN
    )
# ----------------------------------------------- END IRISH ARTICLES ------------------------------------------------- #
