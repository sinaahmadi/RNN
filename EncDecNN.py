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
# encoder-decoder model
#==============================================================================
class EncDecNN(RecurrentNN):
    def __init__(self, enc_layers, dec_layers, embeddings_size, enc_state_size, dec_state_size):
        self.model = dy.Model()

        self.embeddings = self.model.add_lookup_parameters((VOCAB_SIZE, embeddings_size))
        self.ENC_RNN = RNN_BUILDER(enc_layers, embeddings_size, enc_state_size, self.model)
        self.DEC_RNN = RNN_BUILDER(dec_layers, enc_state_size, dec_state_size, self.model)
        self.output_w = self.model.add_parameters((VOCAB_SIZE, dec_state_size))
        self.output_b = self.model.add_parameters((VOCAB_SIZE))
        self.model.save("models/encoderDecoder_character_20epochs", [self.ENC_RNN, self.DEC_RNN, self.embeddings, self.output_b, self.output_w])

    def _encode_string(self, embedded_string):
        initial_state = self.ENC_RNN.initial_state()
        hidden_states = self._run_rnn(initial_state, embedded_string)
        return hidden_states

    def get_loss(self, input_string, output_string):
        input_string = self._add_eos(input_string)
        output_string = self._add_eos(output_string)

        dy.renew_cg()
        embedded_string = self._embed_string(input_string)
        encoded_string = self._encode_string(embedded_string)[-1]
        rnn_state = self.DEC_RNN.initial_state()

        loss = list()
        for i in range(len(output_string)):
            output_char = output_string[i]
            embedded_string_output = self._embed_string(output_string)
            encoded_string_output = self._encode_string(embedded_string_output)[i]
            encoded = dy.concatenate(encoded_string, encoded_string_output)
            rnn_state = rnn_state.add_input(encoded)
            probs = self._get_probs(rnn_state.output())
            loss.append(-dy.log(dy.pick(probs, output_char)))
        loss = dy.esum(loss)
        return loss

    def generate(self, input_string):
        input_string = self._add_eos(input_string)

        dy.renew_cg()
        embedded_string = self._embed_string(input_string)
        encoded_string = self._encode_string(embedded_string)[-1]
        rnn_state = self.DEC_RNN.initial_state()
        output_string = list()
        while True:
            embedded_string_output = self._embed_string(output_string[-1])
            encoded_string_output = self._encode_string(embedded_string_output)
            encoded = dy.concatenate(encoded_string, encoded_string_output)
            
            rnn_state = rnn_state.add_input(encoded)
            probs = self._get_probs(rnn_state.output())
            predicted_char = self._predict(probs)
            output_string.append(predicted_char)
            if predicted_char == EOS or len(output_string) > 2*len(input_string):
                break
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

    if(not os.path.isfile('tiny_vars.pickle')):
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
        with open('tiny_vars.pickle', 'w') as var_file:  
            pickle.dump([phrase_bank_train, phrase_bank_test, phrase_bank_dev, characters, int2char, char2int, VOCAB_SIZE], var_file)
        print "Variables pickled"
    else:
        print "Variables unpickled"
        with open('tiny_vars.pickle') as vars_file:  
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
   
    TEXTE += "<h2>Training with Encoder-decoder RNN</h2>"   
    ENC_RNN_NUM_OF_LAYERS = 1
    DEC_RNN_NUM_OF_LAYERS = 1
    EMBEDDINGS_SIZE = 4
    ENC_STATE_SIZE = 64
    DEC_STATE_SIZE = 80
    Utility.training_display(ENC_RNN_NUM_OF_LAYERS, EMBEDDINGS_SIZE, ENC_STATE_SIZE, EPOCHS)
    
    encoder_decoder = EncDecNN(ENC_RNN_NUM_OF_LAYERS, DEC_RNN_NUM_OF_LAYERS, EMBEDDINGS_SIZE, ENC_STATE_SIZE, DEC_STATE_SIZE)
    train(encoder_decoder, train_set, val_set, EPOCHS)

    system_output = codecs.open("system_output/system_output_encoder_decoder.txt", 'wb', "utf-8")
    for test_phrase in test_set:
        system_output.write(encoder_decoder.generate(test_phrase[0])+"\n")
        
    TEXTE += "<p>Time lapsed: (%s)</p>"%str(datetime.now() - start)   
    start = datetime.now()
    print "Encoder-decoder done."

    Utility.write_html(TEXTE + Utility.TEXTE, "html_output/encoder_decoder_sortie.html")
    print "All outputs saved in sortie.html."