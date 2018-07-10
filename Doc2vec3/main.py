# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 17:25:10 2018

@author: mirza
"""

#Import all the dependencies
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import re
'''
data = ["I love machine learning. Its awesome.",
        "I love coding in python",
        "I love building chatbots",
        "they chat amagingly well"]
'''


with open("data/train.txt", "r") as f:
    train = f.readlines()


tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(train)]

max_epochs = 10
vec_size = 20
alpha = 0.025

model = Doc2Vec(size=vec_size,
                alpha=alpha, 
                min_alpha=0.00025,
                min_count=1,
                dm =1)

print ('-----------------')
#print model.corpus_count
#print len(lines)
  
model.build_vocab(tagged_data)

for epoch in range(max_epochs):
    print('iteration {0}'.format(epoch))
    model.train(tagged_data,
                total_examples=len(train),
                epochs=model.epochs)
    # decrease the learning rate
    model.alpha -= 0.0002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha

model.save("model\d2v.model")
print("Model Saved")


#model= Doc2Vec.load("model\d2v.model")

#to find the vector of a document which is not in training data
test_data = word_tokenize("I love chatbots".lower())

with open("data/test.txt", "r") as f:
    test = f.readlines()

fw = open("vector/test_vectors.txt", "w")

for t in test:
    X = model.infer_vector(word_tokenize(t.lower()))
    print X
    v = str(X).replace("[", "").replace("]", "")
    #v = str(model.docvecs[q]).replace("[", "").replace("]", "")        
    v = re.sub(r'[\r\n]+', '', v)
    v = re.sub(r"[ ]+", ",", v)
    print v
    fw.write(v[1:] + "\n")
    print ('--------------------------------------------')
    
fw.close

'''
v1 = model.infer_vector(test_data)
print("V1_infer", v1)

# to find most similar doc using tags
similar_doc = model.docvecs.most_similar('1')
print(similar_doc)


# to find vector of doc in training data using tags or in other words, printing the vector of document at index 1 in training data
print(model.docvecs['1'])
'''