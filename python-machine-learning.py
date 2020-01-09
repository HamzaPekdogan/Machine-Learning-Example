import numpy as np
import pandas as pd

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, GRU, Embedding, CuDNNGRU
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequnces

//int verip kelime halini dönecektir.
def tokens_to_string(tokens):
    words = [inverse_map[token] for token in tokens if token != 0]
    text = ' '.join(words)
    return text
	
//dataseti yükleme
dataset = pd.read_csv('hepsiburada.csv')

//dataseti ayırma
result = dataset['Rating'].values.tolist()
content = dataset['Review'].values.tolist()

//dataseti %80 ne 20 ye göre ayırma
cutoff = int(len(content) * 0.80)
x_training, testing_x = content[:cutoff], content[cutoff:]
y_training, testing_y = result[:cutoff], result[cutoff:]

//en sık geçen 10000 kelime alınmaktadır ve bunlara birer sayı atanamaktadır.
num_words = 10000
general_token = Tokenizer(num_words=num_words)
general_token.fit_on_texts(content)

//tokenleştirme işlemi yapılır, kelimeler yerine sayı atama yapar
x_train_tokens = general_token.texts_to_sequences(x_training)
x_test_tokens = general_token.texts_to_sequences(testing_x)

//verilerin kaç kelimelei olduğunu buluyor, array ile token sayılarına ulaşılmaktadır.
num_tokens = [len(tokens) for tokens in x_train_tokens + x_test_tokens]
num_tokens = np.array(num_tokens)
max_tokens = np.mean(num_tokens) + 2 * np.std(num_tokens)
max_tokens = int(max_tokens)
np.sum(num_tokens < max_tokens) / len(num_tokens)
x_train_pad = pad_sequences(x_train_tokens, maxlen=max_tokens)
x_test_pad = pad_sequences(x_test_tokens, maxlen=max_tokens)

//ters çevirme işlemi yapılmaktadır
idx = general_token.word_index
inverse_map = dict(zip(idx.values(),idx.keys()))

//ardısık model oluşturuyorum
model = Sequential()
//matrisin size ı her kelimeye karşılık gelen matris uzunluğu
embedding_size = 50

//model e ekleme yapıyor rastgele ekleme yapıyor. kelime sayıları,kelime vektörleri uzunluğu,(10000 e 50 matris oldu), gelen inputun boyutu 59,sonradan çağırmak için ısım ver
model.add(Embedding(input_dim=num_words,output_dim=embedding_size,input_length=max_tokens,name='embedding_layer'))

//yinelenen sinir ağını oluşturmak için
//ilk CuDNNGRU daha hızlı çalışır ekran kartı için çalışır
//16 nöron sayısı 16 aout put verecek  return_sequences=True tamamı döndürülüyor
//return_sequences=false olursa tek output olur
//dense layer sinaptik layer tek nöronda olusur sigmoid olduğu için 1-0 arası olur
model.add(CuDNNGRU(units=16, return_sequences=True))
model.add(CuDNNGRU(units=8, return_sequences=True))
model.add(CuDNNGRU(units=4))
model.add(Dense(1,activation='sigmoid'))

//optimizasyon algorıtmsı 
optimizer = Adam(lr=1e-3)
//model derle
model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])
//eğitimi başlatıyor
model.summary()
//model eğitmeye hazır

//model eğitmeye başlıyoruz epochs=5 eğitim kaç defa eğitim olduğu
model.fit(x_train_pad, y_training, epochs=5, batch_size=256)

//modeli test edelim
result = model.evaluate(x_test_pad, testing_y)
          
//modelin ilk 1000 elemanını test ediyoruz
y_pred = model.predict(x=x_test_pad[0:1000])
//sütunu satıra çeviriyor          
y_pred = y_pred.T[0]
//1 ler ve sıfırlardan olusan bir değer atadık
cls_pred = np.array([1.0 if p>0.5 else 0.0 for p in y_pred])
//test etmek için 1000 elamanın sonucunu aldık
cls_true = np.array(testing_y[0:1000])
//elimideki iki veriyi karşılaştırıyoruz
incorrect = np.where(cls_pred != cls_true)
//hangi yorumlar yanlış onları tutuyor
incorrect = incorrect[0]

//twitterdan gelen verileri modelimize verip sonuçlarını alıyoruz
texts = ["mükemmel","bu parti çok iyi","böyle parti mi olur"]

tokens = general_token.texts_to_sequences(texts)
tokens_pad = pad_sequnces(tokens, maxlen=max_tokens)
tokens_pad.shape

print(model.predict(tokens_pad))
