# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 11:46:39 2017
This script creates the needed data base from the QALB corpus. It returns a lexicon containing lists of correction sets in the format of [original_token, correct_token, action]. The output has a #phrases*#tokens#3 dimension.
@author: sina
"""
import codecs
      
class Alignment:
    def __init__(self, directory, level="character"):
        self.corpus_dir = directory
        self.corpus_lexicon = list()
        self.level = level
        
    def make(self):
        input_file = codecs.open(self.corpus_dir, "r", "utf-8")
        input_text = input_file.read()
        input_file.close()
        corpus = input_text.split("\n\n")
        
        print "Processing..."
        
        for sentence in corpus:
            sentence = sentence.split("\n")  
            annotations = self.one_to_one(sentence)
            self.corpus_lexicon.append(self.extract(sentence, annotations))
        return self.corpus_lexicon
#==============================================================================
# one_to_one method unifies the actions in the correction lines (A). In the case of having more than one action for a given token, the method concatenates all the available actions into one action. This method returns a dictionary containing the unique actions (Key: ID, Value: [original_token, correct_token, action])
#==============================================================================
    def one_to_one(self, sentence):
        phrase = sentence[0].replace("S ", " ").split()
        annotations = sentence[1:]
        one_to_one_dict = dict()
        for line in annotations:
            ID = line.split("|||")[0].replace("A ", "")
            start_index = int(ID.split()[0])
            end_index = int(ID.split()[1])
            
            original_token = " ".join(phrase[start_index:end_index])
            action = line.split("|||")[1]
            correct_token = line.split("|||")[2]

            if(ID not in one_to_one_dict.keys()):
                one_to_one_dict[ID] = [original_token, correct_token, action]
            else:
                if(action == one_to_one_dict[ID][2]):
                    one_to_one_dict[ID] = [original_token, one_to_one_dict[ID][1] + " " +correct_token, action]
                else:
                    one_to_one_dict[ID] = [original_token, one_to_one_dict[ID][1] +" " + correct_token, 'Other']
        return one_to_one_dict
#==============================================================================
# getting the unified annotaitons and the sentence in input, extract method creates the final data set for each phrase.    
#==============================================================================
    def extract(self, sentence, annotations):
        phrase = sentence[0].replace("S ", " ").split()
        
        lexicon = list()
        word_index = 0            
        ID_log = list()
    
        for word in sentence[1:]:
            ID = word.split("|||")[0].replace("A ", "")
            start_index = int(ID.split()[0])
            end_index = int(ID.split()[1])
            
            while(word_index < start_index):
                lexicon.append([phrase[word_index], phrase[word_index], "OK"])
                word_index += 1
    
            word_index += end_index - start_index
            if(ID not in ID_log):
                lexicon.append(annotations[ID])
                ID_log.append(ID)
        return lexicon