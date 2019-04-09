[//]: # (For easier comprehension and better looks, make sure to render this file with a markdown reader.)

# Table of contents
1. [About](#About)
2. [Setup](#Setup)
3. [Input](#Input)
4. [Usage](#Usage)
5. [Performance](#Performance)
6. [Future plans regarding this tool ...](#Future-plans-regarding-this-tool)

___

# About
Given a collection of documents and a collection of tweets from the same time period, this tool tags each document with relevant hashtags using the <i>Hashtagger+</i> algorithm adapted to an offline setting.
<br>
Unlike <i>Hashtagger+</i> (which was designed for tagging a <b>stream</b> of documents in real-time), this tool operates on offline collections.


The details of real-time <i>Hashtagger+</i> can be found in 
* [<b>Hashtagger+: Efficient high-coverage social tagging of streaming news.</b>](https://doi.org/10.1109/TKDE.2017.2754253)
  <br>Bichen Shi, Gevorg Poghosyan, Georgiana Ifrim, Neil Hurley 
  <br>IEEE Transactions on Knowledge and Data Engineering 30.1 (2018): 43-58.
* [<b>Learning-to-rank for real-time high-precision hashtag recommendation for streaming news.</b>](https://doi.org/10.1145/2872427.2882982)
  <br>Bichen Shi, Georgiana Ifrim, Neil Hurley
  <br>Proceedings of the 25th International Conference on World Wide Web (2016): 1191-1202.
  

# Setup
### Installing the packages in the virtual environment with Anaconda
```bash
conda create -n hashtagger_offline python=3.7
conda activate hashtagger_offline # use 'source activate hashtgagger_offline' for older anaconda versions 
conda install beautifulsoup4 lxml nltk numpy pytz scikit-learn
pip install --upgrade pip
pip install elasticsearch stemming sner
```

### Setting up Stanford NLP tools
Download Stanford Named Entity Recognizer at 
[https://nlp.stanford.edu/software/CRF-NER.shtml#Download](https://nlp.stanford.edu/software/CRF-NER.shtml#Download).
<br>
Download Stanford Log-linear Part-Of-Speech Tagger at 
[https://nlp.stanford.edu/software/tagger.html#Download](https://nlp.stanford.edu/software/tagger.html#Download).
Although [basic English Stanford Tagger](https://nlp.stanford.edu/software/stanford-postagger-2018-10-16.zip) should work fine, 
this code has been tested only with [full Stanford Tagger](https://nlp.stanford.edu/software/stanford-postagger-full-2018-10-16.zip).

Unpack both the downloaded archives into your desired location.

Start Stanford NER Tagger server
```bash
cd your_stanford_ner_dir
java -Djava.ext.dirs=./lib -cp stanford-ner.jar edu.stanford.nlp.ie.NERServer -port 9199 -loadClassifier ./classifiers/english.all.3class.distsim.crf.ser.gz
```
The Stanford NER Tagger server port ("9199" in the example above) must be set in the configuration file 
along with the host (most likely "localhost" if you run the NER server on the same machine). 

<b>Using the Stanford tagger with NLTK is about 100 times slower!</b>

### Configuration
All the parameters are set in <tt>hashtagger_config.py</tt> file as attributes of a <tt>Config</tt> class object.

Set <tt>STANFORD_CLASSPATH</tt>, <tt>STANFORD_MODELS</tt>, <tt>NLTK_DATA_PATH</tt> to the corresponding paths.
<br>
Set <tt>ES_HOST_ARTICLE</tt>, <tt>ES_ARTICLE_INDEX_NAME</tt>, <tt>ES_HOST_TWEETS</tt>, <tt>ES_TWEET_INDEX_NAME</tt>.
<br>

The other parameters are configured to sensible values, but feel free to change them.
The parameters that have a significant effect on the tagging consistency and speed are the following: 
<br><tt>GLOBAL_TWEET_WINDOW_BEFORE</tt>, 
<br><tt>GLOBAL_TWEET_WINDOW_AFTER</tt>, 
<br><tt>GLOBAL_TWEET_SAMPLE_SIZE</tt>, 
<br><tt>GLOBAL_TWEET_SAMPLE_RANDOM_FLAG</tt>, 
<br><tt>GLOBAL_HASHTAG_TWEET_SAMPLE_SIZE</tt>, 
<br><tt>GLOBAL_ARTICLE_WINDOW_BEFORE</tt>, 
<br><tt>GLOBAL_ARTICLE_WINDOW_AFTER</tt>, 
<br><tt>COLDSTART_FLAG</tt>, 
<br><tt>COLDSTART_ARTICLE_WINDOW_BEFORE</tt>, 
<br><tt>COLDSTART_ARTICLE_WINDOW_AFTER</tt>, 
<br><tt>COLDSTART_N_TWEETS_PER_NEIGHBOUR_ARTICLE</tt>, 
<br><tt>LOCAL_TWEET_WINDOW_BEFORE</tt>, 
<br><tt>LOCAL_TWEET_WINDOW_AFTER</tt>, 
<br><tt>LOCAL_TWEET_SAMPLE_SIZE</tt>, 
<br><tt>HASHTAG_WINDOW_TWEET_SAMPLE_SIZE</tt>.

The 'default' values are not the optimal ones, but the ones used in the online version of Hashtagger+
described in [DOI:10.1109/TKDE.2017.2754253](https://ieeexplore.ieee.org/abstract/document/8046071).
<br>
The default values with a following '*' indicate that this value is not used in online Hashtagger+ as is,
but is equivalent to those resulting from the periodic processes in Hashtagger+ (crawling tweets from Twitter
every 5 minutes, assigning tweets to article every 15 minutes, tagging articles every 15 minutes).

<b>NOTE! The tagging quality is strongly dependent on the keyword extraction from articles and tweets.</b>

Logging level, console/file output and format can be configured in the very same file.

### Tweet index
This implementation requires a collection of tweets indexed in [Elasticsearch](https://www.elastic.co/products/elasticsearch).
See <tt>es_tweets.py</tt> for the required mapping 
(although not all the fields are necessary and the current mapping is supporting also another tool in our lab). 
<br>
<tt>es_tweets.py</tt> includes <tt>import_web_archive_tweets()</tt> function for indexing tweets from [WebArchive tweet collections](https://archive.org/details/twitterstream).
<br>
[\#lpt](https://www.reddit.com/r/LifeProTip) Downloading the WebArchive tweets is usually much faster with the torrent than with the direct download link.
<br>
N.B. Twitter’s language classification metadata is available in the archive beginning on [March 26, 2013](https://developer.twitter.com/en/docs/tweets/data-dictionary/guides/tweet-timeline.html).
Nevertheless, the 'lang' field appears in tweets as early as December 2012. 

A collection of high quality news-related hashtagged tweets is available for 15.07.2015-24.05.2017 at [https://doi.org/10.6084/m9.figshare.7932422](https://doi.org/10.6084/m9.figshare.7932422).

# Input
The expected format is text file where each line is [JSON object](https://www.w3schools.com/js/js_json_objects.asp) of a document that needs to be tagged.
For efficient execution, the documents must be <b>ordered in time</b>, which will allow less frequent computation of global time window stats.

The expected fields are <i>"id"</i>, <i>"headline"</i>, <i>"subheadline"</i>, <i>"content"</i>, <i>"epoch"</i>, 
<i>"url"</i>, <i>"source"</i>, <i>"type"</i>.

Note that these fields are <b>not mandatory</b> at the input file level, but are <b>necessary</b> for the code to run. 
If a subset of these is missing or has different names, 
the corresponding mappings can be defined in a function and passed as <tt>article_constructor_function</tt> argument 
to the <tt>load_articles_from_json_lines()</tt> function. 
The document's unix timestamp accessor can be defined as a function and passed as <tt>article_epoch_accessor</tt> argument 
to the <tt>load_articles_from_json_lines()</tt> function, so don't worry if your dataset has document timestamps in milliseconds or in a textual format.
Example mapping functions can be found in the main script.  


# Usage
For now it's recommended to create a dedicated script for each input document collection.
<br>
2 example scripts [<tt>tag_irish_articles.py</tt>](./tag_irish_articles.py) and 
[<tt>tag_wapost_articles.py</tt>](tag_wapost_articles.py) are provided as templates.
<br>
For each input document collection, please, specify the input file path (<tt>articles_json_path</tt>), 
document's unix timestamp accessor (<tt>article_epoch_accessor</tt>) 
and article field mapping function (<tt>article_constructor_function</tt>) in the script 
and run the code from the terminal, e.g.
```bash
python tag_irish_articles.py
```  
or 
```bash
python tag_wapost_articles.py
```

The logs are saved in the <tt>logs/</tt> folder. 
The logs in the file and the ones printed in the console window may not match depending on the configuration.


# Performance
It is worth repeating that <b>the tagging quality and coverage are strongly dependent on the keyword extraction from articles and tweets</b>.
The tagging algorithm is of secondary importance...

Another aspect (related to keyword extraction) influencing the tagging performance is the way that tweets are getting matched to the articles.
Tweets are being matched to the n-grams in article's <tt>"stream_keywords"</tt> field 
(the name comes from the original Hashtagger+ where these n-grams were used to stream tweets from Twitter with the Streaming API).
<br>
The original Hashtagger+ filtered the tweets by requiring <i>all</i> of the n-gram terms to appear in a tweet in any order.
This corresponds to setting <tt>COLDSTART_TWEET_NGRAM_MATCH_MODE="must"</tt> and  <tt>LOCAL_TWEET_NGRAM_MATCH_MODE="must"</tt> in the configuration file.
To relax the constraint and allow matching <i>any</i> of the n-gram terms matching a tweet, set the earlier mentioned parameters to <tt>"should"</tt> instead.
<br>
Currently, none of the available modes require preserving the matching terms' order in the n-grams.

The original real-time Hashtagger+ was providing multiple hashtag recommendations per article. 
More precisely, up to 10 hashtags would be recommended ever 15 minutes for a duration of 24 hours, resulting in 96 recommendations of up to 10 distinct hashtags each.
<br>
For each recommendation, a new set of tweets was being added to the article's tweet set.
Each following recommendation was being done on a bigger tweet collection which included all the tweets from the previous recommendation.
This means that although the newer recommendations would be capturing the change is social discussions, nevertheless this change would be coming with some inertia.

In offline setting the multiple recommendations can be done both with and without inertia by selecting a corresponding overlapping/non-overlapping tweet windows respectively.


# Future plans regarding this tool
* fix the bug in the <tt>stream_keywords</tt> extraction and article profile extraction
* add an examples of tagging [Spinn3r](https://www.icwsm.org/data/) documents, [Signal Media One-Million News Articles](https://github.com/signal-ai/Signal-1M-Tools), [Signi-Trend](http://signi-trend.appspot.com/) articles and more
* improve the tweet assignment for cases when there are less than <SOME_THRESHOLD> tweets assigned to an article... increase the time window and/or include more keywords to make sure all articles have (enough) tweets assigned to them 
* fix the suboptimal routine in the global window article sample retrieval for cases when there are less articles than requested in the random sample
* add some labeled data on which the tool's performance can be evaluated quantitatively
* add a pseudocode and a diagram explaining the tagging process and the key differences from the original online (real-time) Hashtagger+
* index tweets in their native format, instead of the custom one, possibly with some additional fields like <tt>n_hashtags</tt> [breaking the backward-compatibility with legacy code]
* move to an object-oriented code

<div align="right"><a href="#Table-of-contents">↥ back to top<a/></div>
