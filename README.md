# Long Short Term Memory (LSTM)

## Definition

Long short term memory refers, in biological and cognitive science, to a model of human memory that distinguishes between short term and long term storage.

Short term memory holds limited information for a short duration, while long term memory is capable of storing large amounts of information over extended periods.

The same idea is applied in artificial intelligence through an architecture called LSTM.

---

## Recurrent Neural Network (RNN)

A recurrent neural network processes sequences step by step, maintaining a hidden state that represents previous information.

### Intuition Example

Sentence: "The cat sat"

Step by step process

t equals 1  
Input is "The"  
Memory becomes context about "The"

t equals 2  
Input is "cat"  
Memory becomes context about "The cat"

t equals 3  
Input is "sat"  
Memory becomes context about the full sentence

The model then predicts the next word based on the final hidden state.

---

## Problem with RNN

RNNs suffer from two major issues during training.

Vanishing gradient  
Gradients shrink over time, making it difficult for the model to learn long term dependencies.

Exploding gradient  
Gradients grow excessively, leading to unstable training.

As a result, RNNs struggle to remember information from earlier in long sequences.

---

## Long Short Term Memory (LSTM)

LSTM is designed to solve the limitations of RNN by introducing a structured memory system.

At each time step, the model receives

x_t which represents the current input  
h_(t minus 1) which represents the previous hidden state  
C_(t minus 1) which represents the previous cell state  

---

## Gates in LSTM

LSTM uses gates to control the flow of information.

### Forget Gate

This gate decides what information should be removed from the previous cell state.

f_t equals sigma of W_f multiplied by the concatenation of h_(t minus 1) and x_t plus b_f

The output ranges between 0 and 1  
0 means complete forgetting  
1 means complete retention  

---

### Input Gate

This gate determines how much new information should be added.

i_t equals sigma of W_i multiplied by the concatenation of h_(t minus 1) and x_t plus b_i

#### Candidate Memory

A new candidate state is created using

C_t tilde equals tanh of W_C multiplied by the concatenation of h_(t minus 1) and x_t plus b_C

---

### Cell State Update

The new cell state is computed by combining the old memory and the new candidate.

C_t equals f_t multiplied element wise with C_(t minus 1) plus i_t multiplied element wise with C_t tilde

This step represents forgetting old information and adding new information.

---

### Output Gate

This gate controls what part of the memory becomes visible as output.

o_t equals sigma of W_o multiplied by the concatenation of h_(t minus 1) and x_t plus b_o

The hidden state is then computed as

h_t equals o_t multiplied element wise with tanh of C_t

---

## Memory Representation

Cell state represents long term memory. It carries information across time with minimal changes.

Hidden state represents short term memory. It is used as the output at each step and passed forward.

At any time step, the hidden state encodes the contextual understanding of the sequence up to that point.

---

## Conceptual Meaning

The model continuously decides

which information to forget  
which information to keep  
which information to output  

This allows it to maintain a meaningful representation of the sequence.

---

## Example

Sentence: "The movie was not good"

When processing the word "good", the model still retains the influence of "not".

This happens because the cell state preserves important context over time.

---

## Important Note

All operations occur within a single LSTM cell that is repeated across time steps.

The gates are described separately only for conceptual understanding.

---

## LSTM in Sequence to Sequence Models

LSTM is commonly used in encoder decoder architectures.

### Encoder

The encoder reads the input sequence from x1 to xT and produces final states h_T and C_T.

These states represent a compressed understanding of the entire sequence.

---

### Decoder

The decoder uses the encoder states to generate the output sequence step by step.

The initial states of the decoder are set as

h_0 of decoder equals h_T of encoder  
C_0 of decoder equals C_T of encoder  

At each step, the decoder predicts the next output based on its current hidden state.

---

## Intuition

The process is similar to watching a movie and then explaining it.

Only the most important information is retained and passed forward.

---

## Summary

LSTM improves upon traditional RNNs by introducing controlled memory mechanisms.

It effectively handles long term dependencies and produces stable training behavior.

This makes it a fundamental model for sequence based tasks such as language modeling and translation.
