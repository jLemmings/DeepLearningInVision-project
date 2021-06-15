import pickle
import numpy as np
import pandas as pd
import re
from bs4 import BeautifulSoup
from keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
import warnings
pd.set_option("display.max_colwidth", 200)
warnings.filterwarnings("ignore")



contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",
                           "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
                           "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                           "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",
                           "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
                           "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",
                           "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have",
                           "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",
                           "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
                           "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",
                           "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",
                           "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",
                           "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",
                           "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",
                           "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",
                           "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",
                           "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",
                           "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",
                           "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",
                           "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",
                           "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
                           "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
                           "you're": "you are", "you've": "you have"}


def text_cleaner(text,num):
    stop_words = set(stopwords.words('english'))
    newString = text.lower()
    newString = BeautifulSoup(newString, "lxml").text
    newString = re.sub(r'\([^)]*\)', '', newString)
    newString = re.sub('"','', newString)
    newString = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in newString.split(" ")])
    newString = re.sub(r"'s\b","",newString)
    newString = re.sub("[^a-zA-Z]", " ", newString)
    newString = re.sub('[m]{2,}', 'mm', newString)
    if(num==0):
        tokens = [w for w in newString.split() if not w in stop_words]
    else:
        tokens=newString.split()
    long_words=[]
    for i in tokens:
        if len(i)>1:                                                 #removing short word
            long_words.append(i)
    return (" ".join(long_words)).strip()


def prepare_data(data):
    cleaned_text = []

    for t in data['Text']:
        cleaned_text.append(text_cleaner(t,0))

    data['cleaned_text']=cleaned_text


    cleaned_text =np.array(data['cleaned_text'])

    short_text=[]
    max_text_len = 30
    for i in range(len(cleaned_text)):
        if (len(cleaned_text[i].split()) <= max_text_len):
            short_text.append(cleaned_text[i])

    df = pd.DataFrame({'text': short_text})

    return df



def seq2summary(input_seq,target_word_index,reverse_target_word_index):
    newString=''
    for i in input_seq:
        if((i!=0 and i!=target_word_index['sostok']) and i!=target_word_index['eostok']):
            newString=newString+reverse_target_word_index[i]+' '
    return newString

def seq2text(input_seq,reverse_source_word_index):
    newString=''
    for i in input_seq:
        if(i!=0):
            newString=newString+reverse_source_word_index[i]+' '
    return newString


def decode_sequence(input_seq,encoder_model,decoder_model,target_word_index,reverse_target_word_index,max_summary_len):
    # Encode the input as state vectors.
    e_out, e_h, e_c = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1))

    # Populate the first word of target sequence with the start word.
    target_seq[0, 0] = target_word_index['sostok']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:

        output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_word_index[sampled_token_index]

        if (sampled_token != 'eostok'):
            decoded_sentence += ' ' + sampled_token

        # Exit condition: either hit max length or find stop word.
        if (sampled_token == 'eostok' or len(decoded_sentence.split()) >= (max_summary_len - 1)):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        # Update internal states
        e_h, e_c = h, c

    return decoded_sentence



data = pd.DataFrame(columns=["Text"])
data.loc[len(data)] = ["I get tonnes of compliments when I wear this to work. Currently one of my favorite shirts. I have 2 colors and will purchase the others. Def recommend (just don't do like me and order the wrong sleeve length lol)"]
data.loc[len(data)] = ["This shirt is a slim fit for an obese man, perhaps. I’m 5’11 and 181 pounds and this thing is way too big. Tried to shrink it down in the wash to no avail. It’s a bigger fit than normal cut dress shirts I buy. Waste of money."]
data.loc[len(data)] = ["Great pool, easy set up just make sure the ground is level. Dislike the description does not give you the filter size. It’s a crap shoot to figure out what size you need. You have to call Coleman."]
df = prepare_data(data)
max_text_len=30
max_summary_len=8

# loading
with open('tokenizer.pickle', 'rb') as handle:
    x_tokenizer = pickle.load(handle)
with open('y_tokenizer.pickle', 'rb') as handle:
    y_tokenizer = pickle.load(handle)

x_tr= np.array(df['text'])
x_tr_seq = x_tokenizer.texts_to_sequences(x_tr)
x_tr = pad_sequences(x_tr_seq,  maxlen=max_text_len, padding='post')

reverse_target_word_index=y_tokenizer.index_word
reverse_source_word_index=x_tokenizer.index_word
target_word_index=y_tokenizer.word_index

# encoder_model = tensorflow.saved_model.load('C:/Users/JumpStart/PycharmProjects/DeepLearningInVision-project/encoder_model/')
encoder_model = keras.models.load_model('C:/Users/JumpStart/PycharmProjects/DeepLearningInVision-project/encoder_model/')
decoder_model = keras.models.load_model('C:/Users/JumpStart/PycharmProjects/DeepLearningInVision-project/decoder_model/')


for i in range(0, 3):
    try:
        print("Review:", seq2text(x_tr[i],reverse_source_word_index))
        # print("Original summary:", seq2summary(y_tr[i]))
        print("Predicted summary:", decode_sequence(x_tr[i].reshape(1, max_text_len),encoder_model,decoder_model,target_word_index,reverse_target_word_index,max_summary_len))
        print("\n")
    except:
        print("Exeption")