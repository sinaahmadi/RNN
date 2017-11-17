# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 20:51:02 2017

@author: sina
"""

import codecs
from preprocessing import Alignment

class Utility:
#==============================================================================
#             
#==============================================================================
    
    def __init__(self):    
        global TEXTE
        self.TEXTE = ""
    
    def data_set(self, corpus_dir, saving_file = False):
        Algmnt = Alignment(corpus_dir)
        corpus_lexicon = Algmnt.make()
    
        original_tokens = [None] * len(corpus_lexicon)
        corrected_tokens = [None] * len(corpus_lexicon)
        action_list = [None] * len(corpus_lexicon)
        
        phrase_bank = list()
    
        for element_index in range(len(corpus_lexicon)):
            element = corpus_lexicon[element_index]
            original = list()
            corrected = list()
            actions = list()
            for token in element:
                original.append(token[0])
                corrected.append(token[1])
                actions.append(token[2])
             
            phrase_double = " ".join(original), " ".join(corrected) 
            phrase_bank.append( phrase_double )
                
            original_tokens[element_index] = original
            corrected_tokens[element_index] = corrected
            action_list[element_index] = actions
            
        dataset = list()
        piled_actions_list = list()
        
        for line_index in range(len(original_tokens)):
            for token_index in range(len(original_tokens[line_index])):
                element = original_tokens[line_index][token_index], corrected_tokens[line_index][token_index]
                dataset.append(element)
                
                piled_actions_list.append((action_list[line_index][token_index]))
        # ----------------- Saving file        
        if(saving_file):
            output_file_original_sentences = codecs.open("output_file_original_sentences.txt", 'wb', "utf-8")
            output_file = codecs.open("data_sets.txt", 'wb', "utf-8")
            
            for i in range(len(original_tokens)):
                for j in range(len(original_tokens[i])):
                    output_file.write("".join(original_tokens[j]) + "\t" + "".join(corrected_tokens[j]) + "\t" + "".join(action_list[j]) + "\n" )
                output_file.write("\n")
            
            
            for i in range( 1000 ):
                sentence_input = original_tokens[i]
                sentence_output = corrected_tokens[i]
                print " ".join(sentence_input) + "\n"
                print " ".join(sentence_output) + "\n"
                output_file_original_sentences.write(" ".join(sentence_input) + "\n")
            
            
            output_file_original_sentences.close()
            output_file.close()    
        # -----------------
        
        return phrase_bank
    #==============================================================================
    #         
    #==============================================================================
    def training_display(self, layers, embeddings, state, epochs):
        global TEXTE 
    
        self.TEXTE += "<ul><li>Number of the layers: %d</li>"%layers
        self.TEXTE += "<li>Number of the embeddings size: %d</li>"%embeddings
        self.TEXTE += "<li>Number of the states: %d</li>"%state
        self.TEXTE += "<li>Number of the epochs: %d</li></ul>"%epochs
    #==============================================================================

    #==============================================================================
    def write_html(self, text, file_name):
        header = '<!DOCTYPE html><html lang="en"><head> <title>Bootstrap Example</title> <meta charset="utf-8"> <meta name="viewport" content="width=device-width, initial-scale=1"> <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css"> <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.2.1/jquery.min.js"></script> <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/js/bootstrap.min.js"></script> <style> .navbar { margin-bottom: 0; border-radius: 0; }  .row.content {height: 450px} .sidenav { padding-top: 20px; background-color: #f1f1f1; height: 100%; } footer { background-color: #555; color: white; padding: 15px; }  @media screen and (max-width: 767px) { .sidenav { height: auto; padding: 15px; } .row.content {height:auto;} } table, th, td {border: 1px solid black; border-collapse: collapse;}th, td {padding: 5px; text-align: left;}table#t01 { width: 100%;background-color: #f1f1c1;} </style></head><body><nav class="navbar navbar-inverse"> <div class="container-fluid"> <div class="navbar-header"> <button type="button" class="navbar-toggle" data-toggle="collapse" data-target="#myNavbar"> <span class="icon-bar"></span> <span class="icon-bar"></span> <span class="icon-bar"></span> </button> </div> <div class="collapse navbar-collapse" id="myNavbar"> <ul class="nav navbar-nav"> <li class="active"><a href="#">Results</a></li> <li><a href="#">About</a></li> </ul> </div> </div></nav> <div class="container-fluid text-center"> <div class="row content"> <div class="col-sm-8 text-left">'
        
        footer = '<br><br><br><br><br></div> </div></div><footer class="navbar-fixed-bottom text-center">  <p><a href="http://sinaahmadi.github.io/" target="_blank">Sina Ahmadi</a>- LIPN 2017</p></footer></body></html> '
        sortie_file = codecs.open(file_name, "w", "utf-8")
        sortie_file.write(header)
        sortie_file.write(text)
        sortie_file.write(footer)
        #os.system("gnome-open sortie.html")