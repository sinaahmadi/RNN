#!/usr/bin/python
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
import codecs
from datetime import datetime
import pickle
#==============================================================================
#   MLP class
#==============================================================================
class MLP:
    def __init__(self, HIDDEN_SIZE, input_size, output_size):        
        # Parameters of the model and training
        self.HIDDEN_SIZE = HIDDEN_SIZE # 100
        self.input_size = input_size # 128
        self.output_size = output_size # len(lexicon)
        self.model = dy.Model()
        # Define the model and SGD optimizer
        self.w_xh_p = self.model.add_parameters((self.HIDDEN_SIZE, self.input_size))
        self.b_h_p = self.model.add_parameters(self.HIDDEN_SIZE)
        self.W_hy_p = self.model.add_parameters((self.output_size, self.HIDDEN_SIZE))
        self.b_y_p = self.model.add_parameters(self.output_size)
        
        self.x_val = dy.vecInput(self.input_size)
                
    def calc_function(self, encoded_string):
        dy.renew_cg()
        w_xh = dy.parameter(self.w_xh_p)
        b_h = dy.parameter(self.b_h_p)
        W_hy = dy.parameter(self.W_hy_p)
        b_y = dy.parameter(self.b_y_p)
        self.x_val.set(encoded_string)
        h_val = dy.tanh(w_xh * self.x_val + b_h)
        y_val = W_hy * h_val + b_y 
        return y_val
#==============================================================================
# Encoder_MLP class that gets the output of the last layer and passes it through the MLP network         
#==============================================================================
class RnnMlp(RecurrentNN):
    def __init__(self, enc_layers, embeddings_size, enc_state_size, mlp_hidden_size, mlp_output_size):
        self.model = dy.Model()
        self.embeddings = self.model.add_lookup_parameters((VOCAB_SIZE, embeddings_size))
        self.ENC_RNN = RNN_BUILDER(enc_layers, embeddings_size, enc_state_size, self.model)
        self.mlp = MLP(mlp_hidden_size, enc_state_size, mlp_output_size)

    def _encode_string(self, embedded_string):
        initial_state = self.ENC_RNN.initial_state()
        hidden_states = self._run_rnn(initial_state, embedded_string)
        return hidden_states
        
    def get_loss(self, input_string, output_string):
        input_string = self._add_eos(input_string)
        output_string = self._add_eos(output_string)
        embedded_string = self._embed_string(input_string)
        encoded_string = self._encode_string(embedded_string)[-1]
        y = self.mlp.calc_function(encoded_string)
        loss = dy.softmax(y - dy.scalarInput(output_string))
        loss = dy.esum(loss)
        
        return loss
    
    def _predict(self, probs):
        probs = probs.value()
        predicted_word = lexicon[probs.index(max(probs))]
        return predicted_word
        
    def generate(self, input_string):
        input_string = self._add_eos(input_string)
        dy.renew_cg()

        embedded_string = self._embed_string(input_string)
        encoded_string = self._encode_string(embedded_string)[-1]
        mlp_output = self.mlp.calc_function(encoded_string)
        predicted_word = self._predict(mlp_output)
        return predicted_word.replace('<EOS>', '')
#==============================================================================
#       Trainer based on SGD algorithm
#==============================================================================
def train(network, train_set, val_set, epochs):
    global TEXTE 
    TEXTE +=  "<ul>"
    MAX_STRING_LEN = 50 # for the scale of the plot of gradient descent
    
    def get_val_set_loss(network, val_set):
        loss = [network.get_loss(input_string, output_string).value() for input_string, output_string in val_set]
        return sum(loss)
    
    trainer = dy.SimpleSGDTrainer(network.model)
    losses = list()
    iterations = list()
    occurences = 0

    for i in range(epochs):
        print "Epoch ", i
        for training_example in train_set:
            occurences += 1
            input_string, output_string = training_example
            
            loss = network.get_loss(input_string, output_string)
            loss_value = loss.value()          
            loss.backward()
            trainer.update()
    
            if occurences%((len(train_set) * epochs)/100) == 0:
                val_loss = get_val_set_loss(network, val_set)
                losses.append(val_loss)
                iterations.append(occurences/(((len(train_set)*epochs)/100)))
    
        plt.ioff()
        fig = plt.figure()
        plt.plot(iterations, losses)
        plt.axis([0, 100, 0, len(val_set)*MAX_STRING_LEN])
        if not os.path.exists("plots"):
            os.makedirs("plots")
        plt.savefig('plots/plot.png')
        plt.close(fig)
        TEXTE += "<li>Epoche %d - loss on validation set is %.9f </li>"%(i, val_loss)
    TEXTE += '</ul><img src="plots/plot.png">'
#==============================================================================
# the main scope
#==============================================================================
if __name__ == "__main__":
    
    from Utility import Utility
    Utility = Utility()    
    
    global TEXTE 
    TEXTE = "" 
    start = datetime.now()    
    
    corpus_dir_train = "./corpus/QALB-Train2014.m2"
    corpus_dir_test = "./corpus/QALB-Test2014.m2"
    corpus_dir_dev =  "./corpus/QALB-Dev2014.m2"
    
    EOS = '<EOS>' # all strings will end with EOS
    TEXTE += "<h2>Pre-processing</h2>"

    if(not os.path.isfile('vars_mlp.pickle')):
        from nltk.probability import FreqDist
        from itertools import chain
        #==============================================================================
        #       Extracting characters
        #==============================================================================
        characters = list()
        
        phrase_bank_train = Utility.data_set(corpus_dir_train)
        phrase_bank_test = Utility.data_set(corpus_dir_test)
        phrase_bank_dev = Utility.data_set(corpus_dir_dev)

        for element in phrase_bank_train:
            for ch in element[0]:
                if ch not in characters:
                    characters.append(ch)
            for ch in element[1]:
                if ch not in characters:
                    characters.append(ch)
        #Creating the data set (train and validation)
        characters.append(EOS)
    
        int2char = list(characters)
        char2int = {c:i for i,c in enumerate(characters)}
        
        VOCAB_SIZE = len(characters)
        #==============================================================================
        #       Lexicon      
        #==============================================================================
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
        lexicon = word_freq.most_common(len(word_freq.items()) - word_freq_threshold)
        #==============================================================================
        #       Preparing data sets
        #==============================================================================
        # for local machine
#        train_set = phrase_bank_train[0 : int(len(phrase_bank_train)/15)] # 90% training set, 10% validation set
#        val_set = phrase_bank_dev[int(len(phrase_bank_dev)/10) : int(len(phrase_bank_dev)/8)]
#        test_set = phrase_bank_test[int(len(phrase_bank_test)/ 10) : int(len(phrase_bank_test)/8)]
        # for server
        train_set = phrase_bank_train
        val_set = phrase_bank_dev
        test_set = phrase_bank_test
        #==============================================================================
        #         Pickling all variables 
        #==============================================================================
        with open('vars_mlp.pickle', 'w') as var_file:  
            pickle.dump([phrase_bank_train, phrase_bank_test, phrase_bank_dev, characters, int2char, char2int, VOCAB_SIZE, lexicon], var_file)
        print "Variables pickled"
    else:
        print "Variables unpickled"
        with open('vars_mlp.pickle') as vars_file:  
            train_set, test_set, val_set, characters, int2char, char2int, VOCAB_SIZE, lexicon = pickle.load(vars_file)
    
    #==============================================================================
    #     Training
    #==============================================================================
    print "Data sets created succesfully."
    TEXTE += "<p>Data sets created succesfully.</p>"
    TEXTE += '<div class="well">Extracted characters are: ' + " ".join(characters) + '</div>'
    TEXTE += "<h3>Statistics of the corpus</h3><ul>"
    TEXTE += "<li>Number of characters (+ EOF): %d</li>"%(len(characters)-1)
    TEXTE += "<li>Size of the training set: %d</li>"%len(train_set)
    TEXTE += "<li>Size of the validation set: %d</li>"%len(val_set)
    TEXTE += "<li>Size of the test set: %d</li>"%len(test_set)
    TEXTE += "<p>Time lapsed: (%s)</p>"%str(datetime.now() - start)
    
    start = datetime.now()    
    if not os.path.exists("models"):
        os.makedirs("models")
    if not os.path.exists("system_output"):
        os.makedirs("system_output")
    if not os.path.exists("html_output"):
        os.makedirs("html_output")
        
    RNN_BUILDER = dy.LSTMBuilder
    EPOCHS = 20
   
    TEXTE += "<h2>Training with Encoder-MLP</h2>"   
    ENC_RNN_NUM_OF_LAYERS = 1
    EMBEDDINGS_SIZE = 4
    ENC_STATE_SIZE = 128
    MLP_HIDDEN_SIZE = 5
    MLP_OUTPUT_SIZE = len(lexicon)
    Utility.training_display(ENC_RNN_NUM_OF_LAYERS, EMBEDDINGS_SIZE, ENC_STATE_SIZE, EPOCHS)
    
    rnn_mlp = RnnMlp(ENC_RNN_NUM_OF_LAYERS, EMBEDDINGS_SIZE, ENC_STATE_SIZE, MLP_HIDDEN_SIZE, MLP_OUTPUT_SIZE)
    train(rnn_mlp, train_set, val_set, EPOCHS)
    #==============================================================================
    #       Generating from the test set
    #==============================================================================
    system_output = codecs.open("system_output/system_output_rnn_mlp.txt", 'wb', "utf-8")
    for test_phrase in test_set:
        system_output.write(rnn_mlp.generate(test_phrase[0])+"\n")
        
    TEXTE += "<p>Time lapsed: (%s)</p>"%str(datetime.now() - start)   
    start = datetime.now()
    print "Encoder-decoder_mlp done."

    Utility.write_html(TEXTE + Utility.TEXTE, "html_output/rnn_mlp_sortie.html")
    print "All outputs saved in sortie.html."