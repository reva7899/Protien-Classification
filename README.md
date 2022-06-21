# Protien-Classification
Protien sequence classification
Protien Sequence Classification
Abstract
The CRISPR-Cas systems are made up of lustered regularly interspaced short palindromic repeats (CRISPR) and their associated (Cas) proteins, which play a critical role in the
adaptive immune system of prokaryotes against invasive foreign components. CRISPR-Cas technologies have also been built to permit target gene editing in eukaryotic
genomes in recent years. Cas protein, as one of the most important components of the CRISPR-Cas system, is indispensable. The kind of CRISPR-Cas system is determined by
the effector module, which is made up of Cas proteins. Effective Cas protein prediction and identification can aid researchers in determining the kind of CRISPR-Cas system.
Furthermore, the class 2 CRISPR-Cas systems are increasingly being used in genome editing. The discovery of the Cas protein will aid in the development of more genome
editing possibilities.
1. Introduction
Proteins (https://en.wikipedia.org/wiki/Protein) are large, complex biomolecules that play many critical roles inbiological bodies. Proteins are made up of one or more long chains
of amino acids(https://en.wikipedia.org/wiki/Amino_acid) sequences. These Sequence are the arrangement of amino acids in a protein held together by peptide
bonds(https://en.wikipedia.org/wiki/Peptide_bond). Proteins can be made from 20 (https://www.hornetjuice.com/amino-acids-types/) different kinds of amino acids, and the
structure and function of each protein are determined by the kinds of amino acids used to make it and how they are arranged. Understanding this relationship between amino
acid sequence and protein function is a long-standing problem in moleculer biology with far-reaching scientific implications. Can we use deep learning that learns the relationship
between unaligned amino acid sequences and their functional annotations.
Problem Description
Based on the Fasta Dataset, classification of a protein's amino acid sequence into one of the protein family categories. In other words, the aim is to determine which class a
protein domain belongs to based on its amino acid sequence.
1.2 Objective
Predict protein family accession from its amino acids sequence with high accuracy.
Avoding latency concerns
2. Data Exploration
2.1 Data Overview
FASTA format is a text-based format for representing either nucleotide sequences or peptide sequences, in which base pairs or amino acids are represented using single-letter
codes. A sequence in FASTA format begins with a single-line description, followed by lines of sequence data. The description line is distinguished from the sequence data by a
greater-than (">") symbol in the first column. It is recommended that all lines of text be shorter than 80 characters in length.
2.1.1 An example sequence in FASTA format is:
>gi|186681228|ref|YP_001864424.1| phycoerythrobilin:ferredoxin
oxidoreductaseMNSERSDVTLYQPFLDYAIAYMRSRLDLEPYPIPTGFESNSAVVGKGKNQEEVVTTSYAFQTAKLRQIRAAHVQGGNSLQVLNFVIFPHLNYDLPFFGADLVTLPGGHLIALDMQP
Sequences are expected to be represented in the standard IUB/IUPAC amino acid and nucleic acid codes, with these exceptions:
1. Lower-case letters are accepted and are mapped into upper-case;
2. A single hyphen or dash can be used to represent a gap of indeterminate length;
3. in amino acid sequences, U and * are acceptable letters (see below).
4. any numerical digits in the query sequence should either be removed or replaced by appropriate letter codes (e.g., N for unknown nucleic acid residue or X for unknown
amino acid residue).
The nucleic acid codes are:
 A --> adenosine M --> A C (amino)
 C --> cytidine S --> G C (strong)
 G --> guanine W --> A T (weak)
 T --> thymidine B --> G T C
 U --> uridine D --> G A T
 R --> G A (purine) H --> A C T
 Y --> T C (pyrimidine) V --> G C A
 K --> G T (keto) N --> A G C T (any)
 - gap of indeterminate length
The accepted amino acid codes are:
A ALA alanine P PRO proline
B ASX aspartate or asparagine Q GLN glutamine
C CYS cystine R ARG arginine
D ASP aspartate S SER serine
E GLU glutamate T THR threonine
F PHE phenylalanine U selenocysteine
G GLY glycine V VAL valine
H HIS histidine W TRP tryptophan
I ILE isoleucine Y TYR tyrosine
K LYS lysine Z GLX glutamate or glutamine
L LEU leucine X any
M MET methionine * translation stop
N ASN asparagine - gap of indeterminate length
With respect to the data set
sequence : These are usually the input features to the model. Amino acid sequence for this domain.There are 20 very common amino acids (frequency > 1,000,000), and 4
amino acids that are quite uncommon: X, U, B, O, Z.
Entry name : These are usually the labels for the model. Accession number in form CS12A_ACISB, where first term is defined as classs
Protein name : Sequence name, in the form "uniprot_accession_id/start_index-end_index".
Length : It contains the length of the sequence
Entry: One word name for family
There were 4 folder one named as CAS and other 3 as Non-CAS. Firstly selecting the fasta format data, dividing accordingly to there attributes by a package called Biopython
and alligning the sequences with respect to length entry name and protien names. After Preprocessing converting data into csv and saving it as the same name with file
extension as .csv file.Repeating the process for all the files and merging all the csv files into one data.csv file
2.1.2 Data split
We have been provided with already done random split(train, val, test) of fasta dataset.
Train - 80% (For training the models).
Valid - 10% (For hyperparameter tuning/model validation).
Test - 10% (For acessing the model performance).
3. Predicting Methods
3.1 Bidirectional LSTM
Bidirectional long-short term memory(Bidirectional LSTM) is the process of making any neural network o have the sequence information in both directions backwards (future to
past) or forward(past to future).
The workings of LSTMs
The LSTM, or Long Short-Term Memory network, is a form of recurrent neural network (RNN) that was created to overcome the problem of disappearing gradients. Because of
this issue, which is created by gradient chaining during error backpropagation, the most upstream layers of a neural network learn very slowly.
It's particularly troublesome when your neural network is recurrent, because the sort of backpropagation needed entails unrolling the network for each input token, effectively
chaining copies of the same model. The vanishing gradients problem gets worse as the sequence gets longer. As a result, we no longer employ traditional or vanilla RNNs as
frequently.
i.By separating memory from hidden outputs, LSTMs solve this problem. An LSTM is made up of memory cells, one of which is shown in the illustration below. As you can see,
the output from the preceding layer [latex]h[t-1][/latex] is segregated from the memory, which is labeled [latex]c[/latex]. The memory interacts with the past output and present
input through three parts, or gates:
ii.The first piece is called the forget gate. It uses a Sigmoid ([latex]sigma[/latex]) function to combine the previous output with the current input, then multiplies the result with
memory. As a result, certain short-term items are erased from memory.
iii.The third and final portion is the output gate. It multiplies a Tanh-normalized representation from memory with a Sigmoid triggered mix of current input and prior output. The
output is then shown and utilized in the following cell, which is a duplicate of the current one with the identical settings.
How unidirectionality can limit your LSTM
Suppose that you are processing the sequence [latex]\text{I go eat now}[/latex] through an LSTM for the purpose of translating it into French. Recall that processing such data
happens on a per-token basis; each token is fed through the LSTM cell which processes the input token and passes the hidden state on to itself. When unrolled (as if you utilize
many copies of the same LSTM model), this process looks as follows:
This
immediately shows that LSTMs are unidirectional. In other words, the sequence is processed into one direction; here, from left to right. This makes common sense, as - except
for a few languages - we read and write in a left-to-right fashion. For translation tasks, this is therefore not a problem, because you don't know what will be said in the future and
hence have no business about knowing what will happen after your current input word.
From unidirectional to bidirectional LSTMs
In some circumstances, a Bidirectional LSTM would be preferable. Sequences are processed both left-to-right and right-to-left in this type of network. In other words, [latex]textI
go eat now[/latex] is processed as [latex]textI rightarrow texgo rightarrow texeat rightarrow texnow[/latex] and [latex]textI leftarrow texgo leftarrow texeat leftarrow texeat leftarrow
texnow[/latex].
While conceptually bidirectional LSTMs work in a bidirectional fashion, they are not bidirectional in practice. Rather, they are just two unidirectional LSTMs for which the output is
combined. Outputs can be combined in multiple ways (TensorFlow, n.d.):
Vector summation. Here, the output equals [latex]\text{LSTM}\rightarrow + \text{LSTM}\leftarrow[/latex]. Vector averaging. Here, the output equals [latex]\frac{1}{2}
(\text{LSTM}\rightarrow + \text{LSTM}\leftarrow)[/latex] Vector multiplication. Here, the output equals [latex]\text{LSTM}\rightarrow \times \text{LSTM}\leftarrow[/latex]. Vector
concatenation. Here, the output vector is twice the dimensionality of the input vectors, because they are concatenated rather than combined.
Constructing a bidirectional LSTM involves the following steps...
1.Specifying the model imports. As you can see, we import a lot of TensorFlow modules. We're using the provided IMDB dataset for educational purposes, Embedding for learned
embeddings, the Dense layer type for classification, and LSTM/Bidirectional for constructing the bidirectional LSTM. Binary crossentropy loss is used together with the Adam
optimizer for optimization. With pad_sequences, we can ensure that our inputs are of equal length. Finally, we'll use Sequential - the Sequential API - for creating the initial
model.
2.Listing the configuration options. I always think it's useful to specify all the configuration options before using them throughout the code. It simply provides the overview that we
need. They are explained in more detail in the tutorial about LSTMs.
3.Loading and preparing the dataset. We use imdb.load_data(...) for loading the dataset given our configuration options, and use pad_sequences to ensure that sentences that
are shorter than our maximum limit are padded with zeroes so that they are of equal length. The IMDB dataset can be used for sentiment analysis: we'll find out whether a review
is positive or negative.
4.Defining the Keras model. In other words, constructing the skeleton of our model. Using Sequential, we initialize a model, and stack the Embedding, Bidirectional LSTM, and
Dense layers on top of each other.
5.Compiling the model. This actually converts the model skeleton into a model that can be trained and used for predictions. Here, we specify the optimizer, loss function and
additional metrics.
6.Generating a summary. This allows us to inspect the model in more detail.
7.Training and evaluating the model. With model.fit(...), we start the training process using our training data, with subsequent evaluation on our testing data using
model.evaluate(...).
3.2 ProtCNN
This model uses residual blocks inspired from ResNet architecture which also includes dilated convolutions offering larger receptive field without increasing number of model
parameters.
Understanding the relationship between amino acid sequence and protein function is a long-standing problem in molecular biology with far-reaching scientific implications.
Despite six decades of progress, state-of-the-art techniques cannot annotate 1/3 of microbial protein sequences, hampering our ability to exploit sequences collected from
diverse organisms. To address this, we report a deep learning model that learns the relationship between unaligned amino acid sequences and their functional classification
across all 17929 families of the Pfam database. Using the Pfam seed sequences we establish a rigorous benchmark assessment and find a dilated convolutional model that
reduces the error of both BLASTp and pHMMs by a factor of nine. Using 80% of the full Pfam database we train a protein family predictor that is more accurate and over 200
times faster than BLASTp, while learning sequence features it was not trained on such as structural disorder and transmembrane helices. Our model co-locates sequences from
unseen families in embedding space, allowing sequences from novel families to be accurately annotated. These results suggest deep learning models will be a core component
of future protein function prediction tools.
Deep learning provides an opportunity to bypass these bottlenecks and directly predict protein functional annotations from sequence data. In these frameworks, a single model
learns the distribution of multiple classes simultaneously, and can be rapidly evaluated. Besides providing highly accurate models, the intermediate layers of a deep neural
network trained with supervision can capture high-level structure of the data through learned representations [13]. These can be leveraged for exploratory data analysis or
supervised learning on new tasks, in particular those with limited data. For example, novel classes can be identified from just a few examples through few-shot learning.
To interrogate what ProtCNN learns about the natural amino acids, we add a 5-dimensional trainable representation between the one-hot amino acid input and the embedding
network (see Methods for details), and retrain our ProtCNN model on the same unaligned sequence data , achieving the same performance. The structural similarities between
these matrices suggest that ProtCNN has learned known amino acid substitution patterns from the unaligned sequence data.
