import pandas as pd

df=pd.read_csv("G:/Project NLP/qa_Electronics.csv")
list(df.columns)
df.shape
df.info()

##answer##
qes=df.iloc[:,4]
qes=pd.DataFrame(qes)
qes.size

qes["question"].isnull().sum() ##checking Null values
# dropping null value columns to avoid errors
qes.dropna(subset=['question'], inplace=True) 

pd.value_counts(qes['question']).head()
print(df['question'].nunique())

qes.duplicated().sum() ###39368 duplicated rows in question columns
qes['question'][2]

##Cleaning The Data
import nltk
#nltk.download()
from nltk.tokenize import sent_tokenize, word_tokenize

# Apply a first round of text cleaning techniques
import re
import string

## Make text lowercase, remove text in square brackets,
## remove punctuation and remove words containing numbers.

def clean_text_round1(text):
 
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

round1 = lambda x: clean_text_round1(x)

# Let's take a look at the updated text
data_clean = pd.DataFrame(qes.question.apply(round1))
data_clean

##Apply a second round of cleaning,Get rid of some additional punctuation
##non-sensical text that was missed the first time around
def clean_text_round2(text):

    text = re.sub('[‘’“”…]', '', text)
    text = re.sub('\n', '', text)
    return text

round2 = lambda x: clean_text_round2(x)

# Let's take a look at the updated text
data_clean = pd.DataFrame(data_clean.question.apply(round2))
data_clean

pd.value_counts(data_clean['question']).head()
data_clean.duplicated().sum()##33944 duplicated rows in data_clean data
data=data_clean.drop_duplicates(['question'])

##Remove stop words
from nltk.corpus import stopwords
stop = stopwords.words('english')
stop_qes = data.question.apply(lambda x: " ".join(x for x in x.split() if x not in stop))
stop_qes.dtype

stop_words = pd.DataFrame(stop_qes)
stop_words.question.nunique()
stop_words.duplicated().sum() ###20697
stop_question = stop_words.drop_duplicates(['question'])
stop_question.question.head()

##Lemmatization
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()
def lemmatize_text(text):
    return [lemmatizer.lemmatize(w,'v') for w in w_tokenizer.tokenize(str(text))]

lemm_question=stop_question.question.apply(lemmatize_text)
lemm_question.head(20)
lem_qes=pd.DataFrame(lemm_question)

##doing stemming part
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
stem_qes = lem_qes.question.apply(lambda x: ' '.join([stemmer.stem(y) for y in x]))
stem_qes.dtype

ste_question = pd.DataFrame(stem_qes)
ste_question.question.nunique()
ste_question.duplicated().sum() ##2917
stem_question = ste_question.drop_duplicates(['question'])
stem_question["question"].head()

#Creating word Cloud
# Joinining all the reviews into single paragraph 
import matplotlib.pyplot as plt
from wordcloud import WordCloud

cloud = " ".join(stem_question.question)

wordcloud= WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(cloud)

plt.imshow(wordcloud)

##For positive world cloud 
with open("G:/Assignments/Text Mining/Positive Words.txt","r") as pos:
  poswords = pos.read().split("\n")

stemqes=stem_question.question.tolist()###covertin data frame into list

qes_pos = ' '.join([w for w in stemqes if w in poswords])

wordcloud_pos = WordCloud(
                           background_color = 'black',
                           width =1800,
                           height =1400
                           ).generate(str(qes_pos))
plt.imshow(wordcloud_pos)

##For negative word cloud
with open("G:/Assignments/Text Mining/Negative Words.txt","r") as nos:
    negwords = nos.read().split("\n")  

qes_neg =' '.join([w for w in stemqes if w in negwords])

wordcloud_neg = WordCloud(
                           background_color = 'black',
                           width =1800,
                           height =1400
                           ).generate(str(qes_neg))
plt.imshow(wordcloud_neg)

##N-Gram
corpus=stem_question["question"]
from sklearn.feature_extraction.text import CountVectorizer

def get_top_n_words(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in      
                   vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], 
                       reverse=True)
    return words_freq[:n]

#Convert most freq words to dataframe for plotting bar plot
top_words = get_top_n_words(corpus, n=25)
top_df = pd.DataFrame(top_words)
top_df.columns=["Word", "Freq"]
top_df.head(25)

#Barplot of most freq words
import seaborn as sns
sns.set(rc={'figure.figsize':(13,8)})
g = sns.barplot(x="Word", y="Freq", data=top_df)
g.set_xticklabels(g.get_xticklabels(), rotation=30)

#Most frequently occuring Bi-grams
def get_top_n2_words(corpus, n=None):
    vec1 = CountVectorizer(ngram_range=(2,2),  
            max_features=2000).fit(corpus)
    bag_of_words = vec1.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in     
                  vec1.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], 
                reverse=True)
    return words_freq[:n]

#Convert most freq words to dataframe for plotting bar plot
top2_words = get_top_n2_words(corpus, n=20)
top2_df = pd.DataFrame(top2_words)
top2_df.columns=["Bi-gram", "Freq"]
print(top2_df)

#Barplot of most freq Bi-grams
import seaborn as sns
sns.set(rc={'figure.figsize':(13,8)})
h=sns.barplot(x="Bi-gram", y="Freq", data=top2_df)
h.set_xticklabels(h.get_xticklabels(), rotation=45)

#Most frequently occuring Tri-grams
def get_top_n3_words(corpus, n=None):
    vec1 = CountVectorizer(ngram_range=(3,3), 
           max_features=2000).fit(corpus)
    bag_of_words = vec1.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in     
                  vec1.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], 
                reverse=True)
    return words_freq[:n]

#Convert most freq words to dataframe for plotting bar plot
top3_words = get_top_n3_words(corpus, n=20)
top3_df = pd.DataFrame(top3_words)
top3_df.columns=["Tri-gram", "Freq"]
print(top3_df)

#Barplot of most freq Tri-grams
import seaborn as sns
sns.set(rc={'figure.figsize':(13,8)})
j=sns.barplot(x="Tri-gram", y="Freq", data=top3_df)
j.set_xticklabels(j.get_xticklabels(), rotation=45)

##Named Entity Recognition with NLTK
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

##Now we apply word tokenization and part-of-speech tagging to the sentence
corpus.question.dtype
corpus=pd.DataFrame(corpus)
def preprocess(sent):
    sent = nltk.word_tokenize(sent)
    sent = nltk.pos_tag(sent)
    return sent

##O/p we get a list of tuples containing the individual words in the 
##sentence and their associated part-of-speech
sent = preprocess(str(corpus["question"]))
print(sent) 

##Now we implement noun phrase chunking to identify named entities using 
##a regular expression consisting of rules that indicate how sentences should be chunked
pattern = 'NP: {<DT>?<JJ>*<NN>}'

cp = nltk.RegexpParser(pattern)
cs = cp.parse(sent)
print(cs)

from nltk.chunk import conlltags2tree, tree2conlltags
from pprint import pprint
iob_tagged = tree2conlltags(cs)
pprint(iob_tagged)

##With the function nltk.ne_chunk(), 
##we can recognize named entities using a classifier  
ne_tree = nltk.ne_chunk(pos_tag(word_tokenize(str(corpus))))
print(ne_tree)

