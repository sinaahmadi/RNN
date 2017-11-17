# -*- coding: utf-8 -*-
"""
Created on Mon Avril 3 14:31:57 2017
@author: sina
"""
# %matplotlib inline
from random import choice, randrange
import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('No display found. Using non-interactive Agg backend.')
    mpl.use('Agg')
import matplotlib.pyplot as plt
#import _gdynet as dy
#dy.init()
import dynet as dy
from datetime import datetime
import pickle
import codecs

#==============================================================================
# Recurrent Neural Network
#==============================================================================
class RecurrentNN:
    def __init__(self, rnn_num_of_layers, embeddings_size, state_size):
        self.model = dy.Model()
        self.embeddings = self.model.add_lookup_parameters((VOCAB_SIZE, embeddings_size))
        self.RNN = RNN_BUILDER(rnn_num_of_layers, embeddings_size, state_size, self.model)
        self.output_w = self.model.add_parameters((VOCAB_SIZE, state_size))
        self.output_b = self.model.add_parameters((VOCAB_SIZE))
        self.model.save("models/rnn_character_20epochs", [self.embeddings, self.RNN, self.output_b, self.output_w])
    
    def _add_eos(self, string):
        string = list(string) + [EOS]
        return [char2int[c] for c in string]
    
    def _preprocess_input(self, string):
        return self._add_eos(string)
    
    def _preprocess_output(self, string):
        return self._add_eos(string)
    
    def _embed_string(self, string):
        return [self.embeddings[char] for char in string]

    def _run_rnn(self, init_state, input_vecs):
        s = init_state
        states = s.add_inputs(input_vecs)
        rnn_outputs = [state.output() for state in states]
        return rnn_outputs
    
    def _get_probs(self, rnn_output):
        output_w = dy.parameter(self.output_w)
        output_b = dy.parameter(self.output_b)
        probs = dy.softmax(output_w * rnn_output + output_b)
        return probs

    def get_loss(self, input_string, output_string):
        input_string = self._preprocess_input(input_string)
        output_string = self._preprocess_output(output_string)

        dy.renew_cg()
        embedded_string = self._embed_string(input_string)
        rnn_state = self.RNN.initial_state()
        rnn_outputs = self._run_rnn(rnn_state, embedded_string)
        lost = list()
        for rnn_output, output_char in zip(rnn_outputs, output_string):
            probs = self._get_probs(rnn_output)
            loss.append(-dy.log(dy.pick(probs, output_char)))
        loss = dy.esum(loss)
        return loss

    def _predict(self, probs):
        probs = probs.value()
        predicted_char = int2char[probs.index(max(probs))]
        return predicted_char
    
    def generate(self, input_string):
        input_string = self._preprocess_input(input_string)

        dy.renew_cg()
        embedded_string = self._embed_string(input_string)
        rnn_state = self.RNN.initial_state()
        rnn_outputs = self._run_rnn(rnn_state, embedded_string)
        
        output_string = []
        for rnn_output in rnn_outputs:
            probs = self._get_probs(rnn_output)
            predicted_char = self._predict(probs)
            output_string.append(predicted_char)
        output_string = ''.join(output_string)
        return output_string.replace('<EOS>', '')
#==============================================================================
# SGD for back-propagation
#==============================================================================
def train(network, train_set, val_set, epochs):
    global TEXTE 
    TEXTE +=  "<ul>"
    MAX_STRING_LEN = 50
    
    def get_val_set_loss(network, val_set):
        loss = [network.get_loss(input_string, output_string).value() for input_string, output_string in val_set]
        return sum(loss)
    
    trainer = dy.SimpleSGDTrainer(network.model)
    losses = []
    iterations = []
    occurences = 0

    for i in range(epochs):
        print "Epoch ", i
        for training_example in train_set:
            occurences += 1
            input_string, output_string = training_example
            
            loss = network.get_loss(input_string, output_string)
            # performing a forward through the network.
            loss_value = loss.value()
            # an optimization step            
            loss.backward()
            trainer.update()
    
            if occurences%((len(train_set) * epochs)/100) == 0:
                val_loss = get_val_set_loss(network, val_set)
                losses.append(val_loss)
                iterations.append(occurences/(((len(train_set)*epochs)/100)))
   
        plot_name = 'plots/' + str(network).split()[0].split('.')[1] + '.png'
        plt.ioff()
        fig = plt.figure()
        plt.plot(iterations, losses)
        plt.axis([0, 100, 0, len(val_set)*MAX_STRING_LEN])
        if not os.path.exists("plots"):
            os.makedirs("plots")
        plt.savefig(plot_name)
        plt.close(fig)
        TEXTE += "<il>Epoche %d - loss on validation set is %.9f </il>"%(i, val_loss)
    TEXTE += '</ul><img src="%s">'%plot_name
#==============================================================================
# the main scope
#==============================================================================
if __name__ == "__main__":
    from nltk.probability import FreqDist 
    from itertools import chain
    
    from Utility import Utility
    Utility = Utility()    
    
    global TEXTE 
    TEXTE = "" 
    start = datetime.now()    
    
    corpus_dir_train = "./corpus/QALB-Train2014.m2"
    corpus_dir_test = "./corpus/QALB-Test2014.m2"
    corpus_dir_dev =  "./corpus/QALB-Dev2014.m2"
    
    EOS = '<EOS>' 
    TEXTE += "<h2>Pre-processing</h2>"
    unknown_token = "<UNKNOWN_TOKEN>"

    if(not os.path.isfile('word_vars.pickle')):
                
        phrase_bank_train = Utility.data_set(corpus_dir_train)
        phrase_bank_test = Utility.data_set(corpus_dir_test)
        phrase_bank_dev = Utility.data_set(corpus_dir_dev)

        corrected_tokens = list()
        for source in [phrase_bank_train, phrase_bank_test, phrase_bank_dev]:
            for phrase in source:
                corrected_tokens.append( phrase[1].split() )
            
        flattened_corrected_tokens = list(chain.from_iterable(corrected_tokens))
        word_freq = FreqDist(flattened_corrected_tokens)
        print "Found %d unique words." % len(word_freq.items())
        
        # counting the words with a frequency less than 10.
#        word_freq_threshold = 0
#        for val in range(len(word_freq.values())):
#            if(word_freq.values()[val] <10):
#                word_freq_threshold += 1
        
        word_freq_threshold = 91291
        VOCAB_SIZE = len(word_freq.items()) - word_freq_threshold
        vocab = word_freq.most_common(VOCAB_SIZE - 2)
        vocab.append((EOS, 0))
        vocab.append((unknown_token, 0))
        
        index2word = [x[0] for x in vocab]
        word2index = dict([(w,i) for i,w in enumerate(index2word)])
       
        print "Using vocabulary size %d." % VOCAB_SIZE
        
        for phrase_bank in [phrase_bank_train, phrase_bank_test, phrase_bank_dev]:
            for j in range(len(phrase_bank)):
                each_phrase = phrase_bank[j]
                each_phrase_list = list()
                for i, sent in enumerate(each_phrase):
                    each_phrase_list.append(" ".join( [w if w in word2index else unknown_token for w in sent.split()]) )
                phrase_bank[j] = (each_phrase_list[0], each_phrase_list[1])
        # for local machine
#        train_set = phrase_bank_train[0 : int(len(phrase_bank_train)/15)] # 90% training set, 10% validation set
#        val_set = phrase_bank_dev[int(len(phrase_bank_dev)/10) : int(len(phrase_bank_dev)/8)]
#        test_set = phrase_bank_test[int(len(phrase_bank_test)/ 10) : int(len(phrase_bank_test)/8)]
        # for server
        train_set = phrase_bank_train
        val_set = phrase_bank_dev
        test_set = phrase_bank_test
        
        # Saving all variables 
        with open('word_vars.pickle', 'w') as var_file:  
            pickle.dump([phrase_bank_train, phrase_bank_test, phrase_bank_dev, index2word, word2index, VOCAB_SIZE], var_file)
        print "Variables pickled"
    else:
        print "Variables unpickled"
        with open('word_vars.pickle') as vars_file:  
            train_set, test_set, val_set, index2word, word2index, VOCAB_SIZE = pickle.load(vars_file)
    
    #----------------------------------------
    print "Data sets created succesfully."
    TEXTE += "<p>Data sets created succesfully.</p>"
    TEXTE += "<h3>Statistics of the corpus</h3><ul>"
    TEXTE += "<li>Vocabulary size + UNKNOWN: %d</li>"%VOCAB_SIZE
    TEXTE += "<li>Size of the training set: %d</li>"%len(train_set)
    TEXTE += "<li>Size of the validation set: %d</li>"%len(val_set)
    TEXTE += "<li>Size of the test set: %d</li>"%len(test_set)
    TEXTE += "<p>Time lapsed: (%s)</p>"%str(datetime.now() - start)
    
    start = datetime.now()    
    if not os.path.exists("word_models"):
        os.makedirs("word_models")
    if not os.path.exists("word_system_output"):
        os.makedirs("word_system_output")
    if not os.path.exists("word_html_output"):
        os.makedirs("word_html_output")
        
   RNN_BUILDER = dy.LSTMBuilder
   EPOCHS = 20
   TEXTE += "<h2>Training with simple RNN</h2>"

   RNN_NUM_OF_LAYERS = 2 
   EMBEDDINGS_SIZE = 4 
   STATE_SIZE = 128 
   Utility.training_display(RNN_NUM_OF_LAYERS, EMBEDDINGS_SIZE, STATE_SIZE, EPOCHS)
   
   rnn = RecurrentNN(RNN_NUM_OF_LAYERS, EMBEDDINGS_SIZE, STATE_SIZE)
   train(rnn, train_set, val_set, EPOCHS)

   system_output = codecs.open("system_output/system_output_rnn_wlevel.txt", 'wb', "utf-8")
   for test_phrase in test_set:
       system_output.write(rnn.generate(test_phrase[0])+"\n")
       
   TEXTE += "<p>Time lapsed: (%s)</p>"%str(datetime.now() - start)   
   start = datetime.now()
   print "RNN done."

   Utility.write_html(TEXTE + Utility.TEXTE, "html_output/rnn_wl_sortie.html")
   print "All outputs saved in sortie.html."