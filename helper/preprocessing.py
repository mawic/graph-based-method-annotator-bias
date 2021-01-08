import re
import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")
stop_words_en = stopwords.words('english')

def preprocess(text):
    #print(text)
    text = text.replace("NEWLINE_TOKEN"," ")
    text = text.replace("TAB_TOKEN"," ")
    # remove URLS
    #print("before url")
    text = re.sub("http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", ' <url> ', text)
    # remove HTML tags
    #print("before html tags")
    text = re.sub("([a-z]+=``.+``)+", ' ', text)
    #print("before html tags")
    text = re.sub("[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+", ' <ip> ', text)
    #remove mails
    #print("before mails")
    text = re.sub("([a-zA-Z0-9._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+)", ' <mail> ', text)
    #remove images
    #print("before images")
    text = re.sub("Image:([a-zA-Z0-9\s?\,?\-?]+)\.(jpg|jpeg|JPG|png|gif)", ' <image> ', text)
    text = re.sub("File:([a-zA-Z0-9\s?\,?\-?]+)\.(jpg|jpeg|JPG|png|gif)", ' <image> ', text)
    
    #print("before date")
    text = re.sub("\d{1,2}\s+(Jan(uary)?|Feb(ruary)?|Mar(ch)?|Apr(il)?|May|Jun(e)?|Jul(y)?|Aug(ust)?|Sep(tember)?|Oct(ober)?|Nov(ember)?|Dec(ember)?)\s+\d{4}\s+\d{1,2}:\d{1,2}", ' <date> ', text)
    text = re.sub("\d{1,2}\s+(Jan(uary)?|Feb(ruary)?|Mar(ch)?|Apr(il)?|May|Jun(e)?|Jul(y)?|Aug(ust)?|Sep(tember)?|Oct(ober)?|Nov(ember)?|Dec(ember)?)\s+\d{4}", ' <date> ', text)
    text = re.sub("\d{1,2}:\d{1,2}", ' <date> ', text)
    text = re.sub("\d{1,4}", ' <number> ', text)
    
    
    #print("before smiley")
    #text = text.replace(":-)"," <smile> ")
    #text = text.replace(":-("," <sad> ")
    
    #print("before UTC")
    text = text.replace("(UTC)"," ")
    text = text.replace(":"," ")
    text = text.replace("="," ")
    text = text.replace("<3"," <love> ")
    text = text.replace("e.g.","eg")
    text = text.replace("i.e.","ie")
    text = text.replace("'s"," is")
    text = text.replace("´s"," is")
    text = text.replace("'re"," are")
    text = text.replace("'nt"," not")
    text = text.replace("'ve"," have")
    text = text.replace("``","")
    text = text.replace("`","")

    #rermove_repeating_chars(text):
    to_remove = ".,?!"
    text = re.sub("(?P<char>[" + re.escape(to_remove) + "])(?P=char)+", r"\1", text)

    
    #print("before mapping")
    mapping = [ ('!', ' '),('?', ' '),('|', ' '),('~', ' '),("'", ' '),('/', ' '),('(', ' '),(')', ' '),('[', ' '),(']', ' '),('—',' '),('-',' '),(',',' '),(':',' '),('→',' '),('*',' '),(';',' '),('.',' '),('•', ' '),('^', ' '),('_', ' '),('{', ' '),('}', ' '),('♥', '<love>'),('#', ' '),('&', ' '),('\xa0',' '),('%',' '),('←',' ')]
    for k, v in mapping:
        text = text.replace(k, v)
      
    text = re.sub(' +', ' ', text)
    text = text.strip()
    text = text.lower()
    text = remove_stopwords(text.split(" "))

    return text

def remove_stopwords(tweet):
    txt = []
    for w in tweet:
        if w in stop_words_en:
            continue
        txt.append(w)

    return(txt)