import pandas as pd

# loading the data
df=pd.read_csv("G:/Project NLP/qa_Electronics.csv")

df=df.drop(['unixTime'],axis=1)

# to get top rows
df.head()
df.columns

df.shape
df.size
df.info()

###Count the Null Columns
null_columns=df.columns[df.isnull().any()]
df[null_columns].isnull().sum()

#Single Column Is Null
df["answer"].isnull().sum()
df["question"].isnull().sum()

df.notnull()

#All Null Columns
print(df[df.isnull().any(axis=1)][null_columns])

# More checks
df['answer'][:2]
df['question'][:2]

# Apply a first round of text cleaning techniques
import pandas as pd
import nltk
nltk.download()
from bs4 import BeautifulSoup
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

## Make text lowercase, remove text in square brackets
##remove punctuation and remove words containing numbers. 
def clean_text_round1(text):
    
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

round1 = lambda x: clean_text_round1(x)

#applying the created method to column for answer and questionType and 
#making as qadata_cleanqt and qadata_cleanqt
df_clean=(df.answer.apply(round1))

df.answer.info



#################################

def clean_text_round1(text):
    '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

round1 = lambda x: clean_text_round1(x)