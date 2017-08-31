
# coding: utf-8

# In[1]:


from __future__ import print_function
import gensim
print(gensim.__version__)


# In[2]:


from gensim.models import KeyedVectors


# In[3]:



# Creating the model
en_model = KeyedVectors.load_word2vec_format('wiki.te.vec')


# In[4]:


# Getting the tokens 
words = []
for word in en_model.vocab:
    words.append(word)

# Printing out number of tokens available
print("Number of Tokens: {}".format(len(words)))

# Printing out the dimension of a word vector 
print("Dimension of a word vector: {}".format(
    len(en_model[words[0]])
))


# In[55]:


# Pick a word 
find_similar_to = 'chilakamma'

# Finding out similar words [default= top 10]
for similar_word in en_model.similar_by_word(find_similar_to):
    print("Word: {0}, Similarity: {1:.2f}".format(
        similar_word[0], similar_word[1]
))


# In[60]:



# Test words 
word_add = ['ganga', 'amma']
word_sub = ['gangamma']

# Word vector addition and subtraction 
for resultant_word in en_model.most_similar(
    positive=word_add, negative=word_sub
):
    print("Word : {0} , Similarity: {1:.2f}".format(
        resultant_word[0], resultant_word[1]
))


# 

# In[ ]:




