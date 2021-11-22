import pickle
from flask.helpers import flash
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import preprocessor as p
import re
import string
from nltk import word_tokenize
from flask import Flask, render_template, request
# from ...src.nlpProject import word2vec

app = Flask(__name__)


# text = word2vec(["hello world "])
# print(text[0])
stopword = nltk.corpus.stopwords.words('english')
stopword += ['yr', 'year', 'woman', 'man', 'girl','boy','one', 'two', 'sixteen', 'yearold', 'fu', 'weeks', 'week',
            'treatment', 'associated', 'patients', 'may','day', 'case','old']
mapping = {0:'&#x2764',1:'&#x1F60D',2:'&#x1F602',3:'&#x1F495',4:'&#x1F525', 5:'&#x1F60A',6:'&#x1F60E',7:'&#x2728',8:'&#x1F499',9:'&#x1F618',10:'&#x1F4F7',11:'&#x1F1F8',12:'&#x2600',13:'&#x1F49C',14:'&#x1F609',15:'&#x1F4AF',16:'&#x1F601',17:'&#x1F384',18:'&#x1F4F8',19:'&#x1F61C'}
def clean_text(text):
    text = re.sub(r'@user','',text)
    text  = "".join([char.lower() for char in text if char not in string.punctuation])
    text  = "".join([char.lower() for char in text if char not in ['…','\n']])
    text = re.sub('[0-9]+', '', text)
    # text = ''.join(i for i in text if ord(i)<128)
    text = re.sub('\s+', ' ', text)
    text = word_tokenize(text)    
    for i in range(len(text)):
        word = text[i]
        newword = word[0]
        prev = word[0]
        prev_cnt = 0
        for j in range(1,len(word)):
            if word[j] == prev:
                prev_cnt+=1
                if prev_cnt < 2:
                    newword += word[j]
            else:
                newword += word[j]
                prev = word[j]
                prev_cnt = 0
                
        text[i] = newword

    text = [word for word in text if word not in stopword]
    #text = [str(TextBlob(word).correct()) for word in text]
    text = [word for word in text if word not in stopword]
    # text = [ps.stem(word) for word in text]
    # text = [word for word in text if 10>len(word)>2]
    # for word in text:
    #     if len(word)<=2:
    #         print(word)
    new_text = " ".join([word for word in text])
    return "tweet " + new_text



def get_prediction(text):
    vec = pickle.load(open('./tfvec','rb'))
    p_processed = p.clean(text)
    finPre = clean_text(p_processed)

    print(finPre)

    bow_test =  vec.transform([finPre]).toarray()
    model = pickle.load(open('../finalModels/SVM/models/noWE_SVM_linear_0.3','rb'))
    prediction = model.predict(bow_test)

    print(prediction)

    return prediction



@app.route('/',methods = ['GET','POST'])
def hello_world():
    prediction = [0]
    models = ['SVM','NOTSVM']
    if request.method == 'POST':
        model = request.form['models']
        print(model)
        prediction = get_prediction(request.form['tweet'])
    return render_template('index.html',prediction=mapping[prediction[0]],models=models)


if __name__ == '__main__':
      app.run(host='127.0.0.1', port=8000)