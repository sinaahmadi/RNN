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
# Bidirectional Recurrent Neural Network
#==============================================================================
class BRecurrentNN(RecurrentNN):
    def __init__(self, rnn_num_of_layers, embeddings_size, state_size):
        self.model = dy.Model()
        self.embeddings = self.model.add_lookup_parameters((VOCAB_SIZE, embeddings_size))
        self.fwd_RNN = RNN_BUILDER(rnn_num_of_layers, embeddings_size, state_size, self.model)
        self.bwd_RNN = RNN_BUILDER(rnn_num_of_layers, embeddings_size, state_size, self.model)
        self.output_w = self.model.add_parameters((VOCAB_SIZE, state_size*2))
        self.output_b = self.model.add_parameters((VOCAB_SIZE))
        self.model.save("models/birnn_character_20epochs", [self.fwd_RNN, self.bwd_RNN, self.embeddings, self.output_b, self.output_w])
    
    def get_loss(self, input_string, output_string):
        input_string = self._preprocess_input(input_string)
        output_string = self._preprocess_output(output_string)

        dy.renew_cg()
        embedded_string = self._embed_string(input_string)
        
        rnn_fwd_state = self.fwd_RNN.initial_state()
        rnn_fwd_outputs = self._run_rnn(rnn_fwd_state, embedded_string)
        rnn_bwd_state = self.bwd_RNN.initial_state()
        rnn_bwd_outputs = self._run_rnn(rnn_bwd_state, embedded_string[::-1])[::-1]
        
        rnn_outputs = [dy.concatenate([fwd_out, bwd_out]) for fwd_out, bwd_out in zip(rnn_fwd_outputs, rnn_bwd_outputs)]
        loss = list()
        for rnn_output, output_char in zip(rnn_outputs, output_string):
            probs = self._get_probs(rnn_output)
            loss.append(-dy.log(dy.pick(probs, output_char)))
        loss = dy.esum(loss)
        return loss
    
    def generate(self, input_string):
        input_string = self._preprocess_input(input_string)

        dy.renew_cg()
        embedded_string = self._embed_string(input_string)
        rnn_fwd_state = self.fwd_RNN.initial_state()
        rnn_fwd_outputs = self._run_rnn(rnn_fwd_state, embedded_string)
        rnn_bwd_state = self.bwd_RNN.initial_state()
        rnn_bwd_outputs = self._run_rnn(rnn_bwd_state, embedded_string[::-1])[::-1]

        rnn_outputs = [dy.concatenate([fwd_out, bwd_out]) for fwd_out, bwd_out in zip(rnn_fwd_outputs, rnn_bwd_outputs)]
        
        output_string  = list()
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
    losses  = list()
    iterations  = list()
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

    if(not os.path.isfile('vars.pickle')):
        
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
        # for local machine
#        train_set = phrase_bank_train[0 : int(len(phrase_bank_train)/15)] # 90% training set, 10% validation set
#        val_set = phrase_bank_dev[int(len(phrase_bank_dev)/10) : int(len(phrase_bank_dev)/8)]
#        test_set = phrase_bank_test[int(len(phrase_bank_test)/ 10) : int(len(phrase_bank_test)/8)]
        # for server
        train_set = phrase_bank_train
        val_set = phrase_bank_dev
        test_set = phrase_bank_test
    
        # Saving all variables 
        with open('vars.pickle', 'w') as var_file:  
            pickle.dump([phrase_bank_train, phrase_bank_test, phrase_bank_dev, characters, int2char, char2int, VOCAB_SIZE], var_file)
        print "Variables pickled"
    else:
        print "Variables unpickled"
        with open('vars.pickle') as vars_file:  
            train_set, test_set, val_set, characters, int2char, char2int, VOCAB_SIZE = pickle.load(vars_file)
    #----------------------------------------
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
   
    TEXTE += "<h2>Training with Bidirectional RNN</h2>"
    RNN_NUM_OF_LAYERS = 2
    EMBEDDINGS_SIZE = 4
    STATE_SIZE = 64
    Utility.training_display(RNN_NUM_OF_LAYERS, EMBEDDINGS_SIZE, STATE_SIZE, EPOCHS)

    birnn = BRecurrentNN(RNN_NUM_OF_LAYERS, EMBEDDINGS_SIZE, STATE_SIZE)
    train(birnn, train_set, val_set, EPOCHS)
    
    system_output = codecs.open("system_output/system_output_birnn.txt", 'wb', "utf-8")
    for test_phrase in test_set:
        system_output.write(birnn.generate(test_phrase[0])+"\n")
    
    TEXTE += "<p>Time lapsed: (%s)</p>"%str(datetime.now() - start)   
    start = datetime.now()
    print "BiRNN done."

    Utility.write_html(TEXTE + Utility.TEXTE, "html_output/birnn_sortie.html")
    print "All outputs saved in sortie.html."