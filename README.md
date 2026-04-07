# Long Short-Term Memory (LSTM) Networks

## Biological Inspiration
[cite_start]Biology Sciences Definition of LSTM[cite: 1]. [cite_start]In biological and cognitive science, human memory distinguishes between two primary types[cite: 2]:
* [cite_start]**Short-Term Memory (STM):** Holds limited information temporarily, typically for seconds to minutes[cite: 3, 4].
* [cite_start]**Long-Term Memory (LTM):** Stores vast amounts of information for extended periods, ranging from days to an entire lifetime[cite: 5, 6].

[cite_start]The same conceptual idea applies to Artificial Intelligence, which features an architecture bearing the exact same name[cite: 7, 8].

***

## The Problem with Normal Recurrent Neural Networks (RNN)
Before LSTMs, traditional RNNs faced severe limitations[cite: 10, 33]:
* [cite_start]**Vanishing Gradient:** During Backpropagation Through Time (BPTT), gradients shrink exponentially over many time steps, meaning the model cannot learn long-term dependencies[cite: 34].
* [cite_start]**Exploding Gradient:** Gradients can become extremely large, causing unstable training and weight divergence[cite: 35].
* **Short-Term Memory:** Because of vanishing gradients, RNNs struggle to remember information from far earlier in a sequence[cite: 36].

### Intuition Example
Consider processing the sentence "The cat sat" step-by-step[cite: 37, 38, 39]:
* [cite_start]At $t=1$: Input is "The", and memory becomes the context about "The"[cite: 40, 41, 42].
* [cite_start]At $t=2$: Input is "cat", and memory becomes the context about "The cat"[cite: 43, 44, 45].
* At $t=3$: Input is "sat", and memory becomes the context about the full sentence[cite: 46, 47, 48].
Finally, the model predicts the next word from the final hidden state[cite: 49].

***

## The LSTM Architecture
The LSTM architecture resolves these issues using three main gates[cite: 50, 55]. At each time step $t$, the inputs are the current input vector $x_{t}$, the previous hidden state $h_{t-1}$, and the previous cell state $C_{t-1}$[cite: 67, 68, 69, 70]. 

*Important Note: All of this happens inside one LSTM cell repeated across time, not multiple different LSTMs[cite: 132]. [cite_start]The gates are described separately for conceptual clarity[cite: 133].*

### 1. Forget Gate
[cite_start]This gate decides what information to remove from the previous cell state[cite: 56, 71, 72]. [cite_start]It uses a sigmoid function, outputting $0$ to forget completely or $1$ to keep completely[cite: 73, 74].
[cite_start]$$f_{t}=\sigma(W_{f}[h_{t-1},x_{t}]+b_{f})$$ [cite: 76]

### 2. Input Gate and Candidate Memory
[cite_start]The Input Gate decides how much new information to store, using a sigmoid function where $0$ means "do not update" and $1$ means "fully update"[cite: 60, 75, 77, 79, 80].
[cite_start]$$i_{t}=\sigma(W_{i}[h_{t-1},x_{t}]+b_{i})$$ [cite: 78]

[cite_start]Alongside this, the Candidate Memory creates new potential values (representing new candidate information) to add to the cell state, using a tanh function that outputs values between $-1$ and $1$[cite: 81, 82, 84, 85, 89].
[cite_start]$$C_{t}=tanh(W_{C}[h_{t-1},x_{t}]+b_{C})$$ [cite: 83]

### 3. Cell State Update
[cite_start]The core memory update happens here[cite: 90]. 
[cite_start]$$C_{t}=f_{t}\odot C_{t-1}+i_{t}\odot\tilde{C_{t}}$$ [cite: 98]
[cite_start]In this equation, the first term represents forgetting part of the old memory, the second term adds the new memory, and $\odot$ denotes element-wise multiplication[cite: 91].

### 4. Output Gate
[cite_start]The Output Gate controls what information becomes the output using a sigmoid function[cite: 92, 94, 95].
[cite_start]$$o_{t}=\sigma(W_{o}[h_{t-1},x_{t}]+b_{o})$$ [cite: 99]

[cite_start]The Hidden State is then computed as[cite: 93, 113]:
[cite_start]$$h_{t}=o_{t}\odot tanh(C_{t})$$ [cite: 100, 120]

[cite_start]At each time step, the LSTM produces a vector of a fixed dimension that acts as a contextual representation of everything remembered up to time $t$[cite: 97, 101, 102, 103]. [cite_start]Mathematically, $h_{t}\in\mathbb{R}^{n}$[cite: 121].

***

## Internal Memory vs. Output Memory
[cite_start]LSTMs maintain two distinct memory states[cite: 104]:

1. [cite_start]**Cell State (Long-Term Memory):** $C_{t}$ stores long-term information and decides what is remembered or forgotten[cite: 105, 106, 107, 108]. [cite_start]It flows through time with minimal modification[cite: 109].
2. [cite_start]**Hidden State (Short-Term / Output Memory):** $h_{t}$ is passed to the next time step and the output layer[cite: 111, 112, 113]. 

[cite_start]Conceptually, at time step $t$, the hidden state encodes which words were forgotten, which words were remembered, and the current context of the sequence[cite: 114, 115, 116, 117, 118, 119]. [cite_start]For example, in the sentence "The movie was not good", at the word "good", $h_{t}$ heavily encodes the influence of "not"[cite: 122, 123, 124]. [cite_start]This works because the forget gate removes irrelevant memory, the input gate adds useful information, and the output gate selects what becomes visible[cite: 126, 127, 128, 129, 130, 131].

***

## LSTM in Sequence-to-Sequence (Seq2Seq) Models
LSTMs are powerful when used as Encoders and Decoders in Seq2Seq architectures[cite: 134].

### The Encoder
The encoder reads the input sequence to represent the compressed context[cite: 135, 142, 143]. Its intuition is to forget unimportant details, keep important events, and refine understanding[cite: 144, 145, 146, 147]. By the end of the sequence, $h_{T}$ represents your final understanding of the input[cite: 148].

### The Decoder
The encoder's final state $h_{T}$ is passed to the decoder, acting as a summarized context and filtered memory of the important information[cite: 152, 164, 165, 166, 167]. It is much like explaining a movie after watching it by summarizing only the important parts[cite: 168].

The decoder LSTM is initialized using the final states from the encoder[cite: 169, 170]:
$$h_{0}^{(dec)}=h_{T}^{(enc)},C_{0}^{(dec)}=C_{T}^{(enc)}$$ [cite: 172]

Given the previous predicted word $y_{t-1}$, the decoder generates the next sequence by calculating the probability $P(y_{t}|h_{t}^{(dec)})$[cite: 171, 173].
