#  Long Short-Term Memory (LSTM) Networks 

Welcome! This guide breaks down the core concepts, math, and intuition behind Long Short-Term Memory (LSTM) networks. 

##  Biological Inspiration
In biological and cognitive science, human memory is divided into two main categories:
* **Short-Term Memory (STM):** Holds a limited amount of information temporarily (for seconds to minutes).
* **Long-Term Memory (LTM):** Stores vast amounts of information for extended periods, from days to an entire lifetime.

Artificial Intelligence borrows this exact concept and applies it to neural networks to handle sequential data more effectively.

***

##  The Problem with Standard RNNs
Before LSTMs, traditional Recurrent Neural Networks (RNNs) struggled with learning long sequences due to a few major roadblocks:

* **The Vanishing Gradient:** During training (Backpropagation Through Time), gradients shrink exponentially as they move backward. This means the model literally "forgets" earlier information and cannot learn long-term dependencies.
* **The Exploding Gradient:** The opposite problem, where gradients become massive, causing the model to become unstable and crash.
* **Severe Short-Term Memory:** Because of vanishing gradients, RNNs struggle to remember information from the beginning of a long sequence.

### A Quick Intuition Example
Imagine processing the sentence: "The cat sat"
1. At $t=1$: The input is "The". The model's memory saves the context of "The".
2. At $t=2$: The input is "cat". The memory updates to the context of "The cat".
3. At $t=3$: The input is "sat". The memory holds the full sentence context.

Finally, the model uses this compiled final state to predict the next word.

***

##  The LSTM Architecture
To fix the memory problems of standard RNNs, the LSTM introduces three specialized "gates." 

At any given time step $t$, the cell takes three inputs: the current data $x_{t}$, the previous hidden state $h_{t-1}$, and the previous long-term memory state $C_{t-1}$.

*Note: All of these gates exist inside a single LSTM cell that repeats across time.*

### 1. The Forget Gate
This gate decides what old information we no longer need. It applies a sigmoid function, outputting a value between $0$ (forget completely) and $1$ (keep completely).
$$f_{t}=\sigma(W_{f}[h_{t-1},x_{t}]+b_{f})$$

### 2. The Input Gate & Candidate Memory
Next, the network decides what *new* information is worth storing. 
* The **Input Gate** uses a sigmoid function to decide how much to update ($0$ to $1$).
$$i_{t}=\sigma(W_{i}[h_{t-1},x_{t}]+b_{i})$$
* The **Candidate Memory** generates the actual new potential values using a tanh function (between $-1$ and $1$).
$$C_{t}=tanh(W_{C}[h_{t-1},x_{t}]+b_{C})$$

### 3. Updating the Cell State
Now, we combine the decisions from steps 1 and 2 to update the core memory. We multiply the old memory by the forget gate, and add the new candidate memory multiplied by the input gate.
$$C_{t}=f_{t}\odot C_{t-1}+i_{t}\odot\tilde{C_{t}}$$
*(Note: $\odot$ represents element-wise multiplication).*

### 4. The Output Gate
Finally, the cell decides what to output as the new hidden state. It filters the updated memory through an output gate.
$$o_{t}=\sigma(W_{o}[h_{t-1},x_{t}]+b_{o})$$
$$h_{t}=o_{t}\odot tanh(C_{t})$$

***

##  Internal Memory vs. Output Memory
LSTMs maintain two distinct tracking states to manage context:

1. **Cell State ($C_{t}$):** This is the **Long-Term Memory**. It flows straight through the entire chain with minor linear interactions. It acts as the core information highway.
2. **Hidden State ($h_{t}$):** This is the **Short-Term Memory / Output**. This is passed to the next time step and to the upper layers of the network. 

**Example:** In the sentence "The movie was not good", when the model reaches the word "good", the hidden state strongly encodes the influence of "not". The forget gate dropped irrelevant earlier words, the input gate focused on "not", and the output gate passed this crucial context forward.

***

##  LSTMs in Sequence-to-Sequence (Seq2Seq)
LSTMs shine in translation and summarization tasks using an Encoder-Decoder structure.

###  The Encoder
The encoder reads the input sequence step-by-step. Its job is to forget unimportant details, keep important events, and refine its understanding. By the final time step $T$, the final hidden state $h_{T}$ represents a dense, compressed summary of the entire input.

###  The Decoder
This summary state $h_{T}$ is then handed off to the decoder. Think of this like watching a movie, building an understanding in your head (encoding), and then explaining the plot to a friend (decoding).

The decoder is initialized using the final states of the encoder:
$$h_{0}^{(dec)}=h_{T}^{(enc)},C_{0}^{(dec)}=C_{T}^{(enc)}$$

Using this context and the previously generated word $y_{t-1}$, the decoder calculates the probability of the next word in the sequence:
$$P(y_{t}|h_{t}^{(dec)})$$
