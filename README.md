
# Recurrent Neural Network Models for Error Correction

This repository provides the source code of various models that have been described in [my master's thesis](http://sinaahmadi.github.io/files/[SinaAhmadi]Masters_thesis.pdf). This project aims at implementing and evaluating neural network models, in particular, Recurrent Neural Network (RNN), Bidirectional Recurrent Neural Network (BRNN), Sequence-to-Sequence (seq-to-seq) models and finally, attention-based mechanism in Sequence-to-Sequence models. The following figure illustrates an encoder-decoder model predicting the corrected form of the given incorrect phrase. 

![An encoder-decoder model for error correction](imgs/encoder_decoder.png "Recurrent Neural Network" )

### DyNet library

In the implementation of the current project we have been using DyNet. The *Dynamic Neural Network Toolkit*, or DyNet, is a neural network library suited to networks that have dynamic structures.  DyNet supports both static and dynamic declaration strategies used in neural networks computations.  In the dynamic declaration, each network is built by using a directed and acyclic computation graph that is composed of expressions and parameters that define the model. Working efficiently on CPU or GPU, DyNet has powered a number of NLP research papers and projects recently. You may find more information about DyNet [here](http://dynet.readthedocs.io/en/latest/index.html#). 

### Data set

Our approach is language-independent. Specifically for our project, we have trained and evaluated the models using the [QALB corpus](http://nlp.qatar.cmu.edu/qalb/) which is an Arabic corpus annotated based on the annotation style of the CoNLL-2013 shared task. A pickled version of a part of the corpus is also provided. 

### Training the models

Assuming the task of grammatical and spelling error correction as a monolingual translation task, we train each model using a potentially incorrect phrase with its gold-standard correction as training instance. In the provided codes, the models can be trained in character-level or word-level. The `preprocessing` class may help you in extracting specific parts of the annotated corpus into trainable data sets as well. 

For the experiments of my thesis, I used [a dedicated cluster](http://lipn.univ-paris13.fr/rcln/wiki/index.php/Cluster_TAL) at the host laboratory which was enough efficient to train the models with. In any case, it would be a good idea to work on the **optimization** of the introduced models in the current project, such as a) using hierarchical softmax [1] instead of the simple softmax, b) replacing stochastic gradient descent (`dynet.SimpleSGDTrainer()`) by Adam for stochastic optimization [2], c) modifying the initialization values and d) adapting hyperparameters.

### Some (more) details
  * At the end of each complete execution, an HTML file including the hyper-parameters of the models, details of the data set and the cross-entropy results, is created and saved in the `html_output` folder. To have your desired information saved, you may need to apply some modifications in the `Utility` class.
  * Parameters of the models are saved in the `models` folder. This is done by `model.save()`.
  * The loss value over the validation set in each epoch is illustrated and saved as a plot in the `plot` folder.
  * The output of each model for a given test set is saved in a text file in the `system_output` folder. 
  * An RNN-MLP model is also provided which may be used for word-level models. This model is not discussed in the thesis. 

### Requirements
  * [DyNet](http://dynet.readthedocs.io/en/latest/).
  * Python 2.7


### Reference
All details regarding the discussed models are documented in [my master's thesis](http://sinaahmadi.github.io/docs/[SinaAhmadi]Masters_thesis.pdf).

[1]: Morin, F., & Bengio, Y. (2005, January). Hierarchical Probabilistic Neural Network Language Model. In Aistats (Vol. 5, pp. 246-252).

[2]: Kingma, D., & Ba, J. (2014). Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980.


### Confidentiality 
Some parts of the codes may not have been provided in their entirety due to the confidentiality of the project.
