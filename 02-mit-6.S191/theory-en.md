# Deep Sequence Modelling
How do we model sequential data, where order and temporal context are fundamental requirements?

What are some examples of sequences in nature?
- sounds;
- videos;
- text;
- DNA;
- motion;
- etc.

There are 4 architectures for building sequential neural networks:
- **one to one**: given a static input, we associate a static output (given a student's grades, predict whether they will pass the exam or not);
- **many to one**: given a sequence, we associate a static output (given a tweet, assign it a positive or negative sentiment);
- **one to many**: given a static input, we associate a sequence (given an image, associate a caption with it);
- **many to many**: given a sequence, we associate another sequence (given a text in Italian, translate it into English).

At this point, we need to reflect on older models, which we can no longer use once time comes into play.
We must consider individual time steps.

To overcome the limitations of classical networks and allow the internal state to depend on the past and on its own previous state, **Recurrent Neural Networks (RNNs)** were born.

Why are the perceptron and the Feed-forward network (dense layer, multi-layer) no longer suitable?
- **Unidirectional flow (no cycles):** In classical networks, information travels in only one direction. The input enters on the left and information moves rightward to the output, with no backward loops. For example: a network that classifies X-ray images receives pixels, processes them layer by layer, and outputs "tumour or not tumour". Information only goes forward; there is no reason to look back, as each image is independent.
In RNNs, by contrast, the network's state travels both forward toward the output and backward into itself, so that it can be re-analysed and taken into account at the next time step. For example: in the sentence "Calcio è uno sport violento" ("Football is a violent sport"), the word *calcio* could mean the mineral calcium, the kicking action, or the sport. To understand which, we need to read *sport*, which comes later. A classical network cannot go back and re-evaluate *calcio* in light of what was read afterwards.

- **Independence Assumption (Lack of Memory):** As a direct consequence of the first point, a Feed-forward network assumes that the input at a given position is completely independent of what happened before. Every time step resets the model's "brain". For example: predicting a student's grade from their parameters (study hours, attendance, previous grades) — each student is independent of the others; there is no temporal context to remember.
In real sequences, the meaning of an element depends entirely on the previous context and on the accumulated memory, which classical networks destroy at every time step. For example: generating a response to "My name is Marco, I am 25 years old. What career do you recommend?" — without memory of the context, the model answers without remembering that the user is called Marco and is 25 years old. Every word is disconnected from the previous one.


# Recurrent Neural Networks (RNNs)
A Recurrent Neural Network (RNN) is a class of artificial neural networks designed specifically for processing sequential data.
Unlike traditional Feed-Forward networks, which process the entire input in a single isolated instant, an RNN operates step by step through time, processing the elements of a sequence one at a time in chronological order.

The term **Recurrent** derives from the presence of a feedback loop within the structure of the network.

Instead of simply transmitting information forward to the next layer, the recurrent layer takes the result of the computation performed at the current time and feeds it back into itself as input for the next step's computation.

This continuous "returning to itself" allows the network to develop an internal memory. Thanks to this mechanism, the network's response to a given input does not depend exclusively on the element it is reading at that precise moment, but is influenced by the entire history of past elements it has already encountered along the sequence.

To implement the idea of feeding information back into itself and creating a real dependency on the past, RNNs introduce an internal memory variable called the **Hidden State**, denoted $h_t$.

At each single time step $t$, the network applies a mathematical function $f_W$ that takes the current input $x_t$ and combines it with its own previous state $h_{t-1}$:

$$h_t = f_W(h_{t-1}, x_t)$$

We have just described memory mathematically. But why $f_W$?
That $W$ represents the weights of the neural network that define the function itself — it is a function of the weights. The weights do not change; it is the inputs that change at each single time step. The weights (parameters) always remain the same while the network reads the sequence step by step; they will only change when we perform backpropagation.

The phase in which we process sequentially is the **Forward pass**; the phase of backpropagation is the **Backward pass**.

At each single instant $t$ (when the network reads a new word, a new video frame, or a new millisecond of audio), the network takes the old blackboard ($h_{t-1}$), adds to it the information from the present ($x_t$), erases unnecessary details, and writes the new version of the blackboard ($h_t$).

RNNs have a state, $h_t$, which is updated at each time step as a sequence is being processed.

## How does the network work at instant t?

1. The cell receives two vectors from outside:

    $x_t$ **(Input Vector):** The data from the present (e.g. the numbers representing the current word).

    $h_{t-1}$ **(Old Hidden State):** The vector arriving from the past, containing the memory of previous steps.

2. The cell takes these two vectors and feeds them into the update formula.
It multiplies the input by its weights ($W_{xh} \cdot x_t$).
It multiplies the past by its weights ($W_{hh} \cdot h_{t-1}$).
It sums the results (and the bias) and squashes them through the tanh function.
The result of this computation is the new Hidden State ($h_t$). This vector has a double life: it is sent forward in time, becoming the past for the next time step ($t+1$). It is sent upward to compute the output of the present.
$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$

3. The network does not just remember — it must also produce a response (Output Vector).
To do so, it takes the newly computed state ($h_t$) and multiplies it by the third weight matrix, called $W_{hy}h_t$.
If the network must predict the next word, this $y_t$ will pass through a function to output the probability of the correct word.
$$y_t = W_{hy} h_t + b_y$$

### How does the complete RNN work in essence?

The three fundamental matrices never change from one step to the next: $W_{xh}$ processes the current input ($x_0, x_1, x_2, \dots, x_t$), $W_{hh}$ transmits memory from one cell to the next, $W_{hy}$ projects the internal state toward the output (${\hat{y}}_0, {\hat{y}}_1, {\hat{y}}_2, \dots, {\hat{y}}_t$).

While the sequence advances from left to right:

At step $t=0$, input $x_0$ generates prediction ${\hat{y}}_0$.

This prediction is immediately compared with the real target to compute the first partial loss: $L_0$.

At step $t=1$, input $x_1$ joins the previous memory via $W_{hh}$ and generates the second prediction ${\hat{y}}_1$, producing loss: $L_1$.

This process repeats identically for every single instant until the end of the sequence ($t$).

## What are the design criteria for building an effective RNN?

1. **Handle variable-length sequences:**
In nature, sequences never have a fixed length.
If an image classification model always receives $224 \times 224$ pixel photos, a language model must be able to read both a 3-word sentence and an entire 100,000-word novel.
The architecture must have no rigid constraints on the input size.
It must be able to process arbitrarily long sequences without redesigning the network structure or changing the number of parameters.

2. **Track long-term dependencies:**
The meaning of a word at the end of a page may depend entirely on a word written at the beginning of the chapter.
The model must have a memory structure capable of preserving information intact even when tens or hundreds of intermediate steps lie between two correlated elements.

3. **Preserve the order of data:** In sequences, order completely changes the meaning of the message.
The two sentences "The dog bites the boy" and "The boy bites the dog" contain exactly the same words, but describe two opposite situations.
The network must process information in strict chronological order, so that the final representation correctly reflects who performs the action and who receives it.

4. **Share parameters along the sequence:** For example, if the word "football" appears at the beginning or at the end of a sentence, the fundamental rule for processing and understanding it must not change. We cannot train different parameters for every single position in the text; otherwise the model would become enormous and would not know what to do if it encountered a longer sentence than expected. Therefore, the weights and computation rules of the network must be shared (the same) for every time step. This makes the model flexible and allows it to generalise grammatical or structural rules regardless of when an element appears in the sequence.

## A sequential problem: predicting the next word in a sentence.

"This morning I took my dog for a walk."

Given the input "This morning I took my dog for a", predict the next word.

An important consideration: neural networks do not take words as input, only numbers. We must therefore transform our words into fixed-size vectors of numbers.

Three phases:

- **Tokenisation:** We split a sentence into words (and in practice, the entire vocabulary of a language):
input: "This morning I took my dog for a"
output: ["This", "morning", "I", "took", "my", "dog", "for", "a"];

- **Indexing:** We assign a number to each word:
The model has a fixed internal dictionary, called the vocabulary, which contains all the words it learned during training. In this phase, each token is looked up in the vocabulary and replaced with its respective integer ID (the index).
output: [1, 2, 3, 4, 5, 6, 7, 8];

- **One-hot embedding:** We associate each number with a vector:
This layer replaces each integer ID with a vector of zeros except for the position corresponding to the word.
For example, the output for the word "dog" ($x_7$): [0, 0, 0, 0, 0, 0, 1]

Instead of one-hot embedding, we could use **learned embedding** — a kind of map for understanding how correlated words are with each other. Words that have close or related semantic relationships (e.g. "dog", "cat", "puppy") will have vectors with very similar numerical values, placing them close together in the geometric space traced by the network.

As we saw earlier, the meaning of a sentence changes if we do not preserve the original order — this was one of the criteria for an RNN.

So, can the criteria for predicting the next word be satisfied by an RNN?
- **Encoding language for a neural network:** Yes, as we just saw;
- **Handle variable sequence lengths:** Yes, RNNs are designed to accept variable input sizes;
- **Model long-term dependencies:** Yes, as we saw earlier;
- **Capture differences in sequence order:** Yes, preserving order is one of the core criteria of RNNs.

## How do we train RNNs in practice?
With backpropagation, just as with simple neural networks. But it is more complex in the case of recurrent sequential networks, and we will see why.

How did we do it for non-recurrent neural networks?
Given some inputs, we passed through the weights, produced outputs. We then computed the Loss (the discrepancy between the prediction and the real target). Backpropagation consisted of computing the gradient of this loss function with respect to the network's weights, moving in the opposite direction of the gradient (gradient descent), and updating the weights to minimise the loss.

Now, however, things are a bit more complex, because we need to extend this concept to the context of RNNs.
As we said, in RNNs we have this step-by-step sequential processing, so we can predict the output at every time step, and we can compute the loss at every time step.
We sum all these losses and obtain a single total loss.

Here, backpropagation is different: instead of going back through every piece of the network at every time step (i.e., instead of performing classical backpropagation at every time step), we perform **Backpropagation Through Time (BPTT)**, which means:
we do not update any weights until we have finished reading the entire temporal sequence, because we carry the error backwards through time. We take the total error computed at the end of the sequence, for example, and propagate it backwards through the chain with respect to the memory weight matrix ($W_{hh}$) — the only one that is shared across time.

Going back in time, a problem arises: at each step we must perform matrix multiplications of the memory weights, because it means deriving all the intermediate states backwards (to see how $h_0$ influences the loss):

$$\frac{\partial h_3}{\partial h_0} = \frac{\partial h_3}{\partial h_2} \times \frac{\partial h_2}{\partial h_1} \times \frac{\partial h_1}{\partial h_0}$$

and since each of those intermediate steps contains the weight matrix, we end up computing accumulated factors of $W_{hh}$, such as $W_{hh}^3$:

$$\frac{\partial h_3}{\partial h_0} \propto W_{hh} \times W_{hh} \times W_{hh} = W_{hh}^3$$

What does this imply?

- If values are > 1 → **exploding gradients** (the solution is to normalise the gradient values within a range so that the result has a maximum and does not explode: **gradient clipping**);
- If values are < 1 → **vanishing gradients**: the gradient tends to zero and we cannot capture problems far from the final position, only those near the end of the network. For example, with a derivative of $0.0000001$, we would only be able to update the weights of nearby words; the network would become completely blind to the words at the beginning.

There are several solutions to address the vanishing gradient:

- activation functions;
- weight initialisation;
- network architecture.

One solution to this problem is the **Gating mechanism** in neurons. That is, we use gates to decide whether to add or remove information from the memory state.

The Gating mechanism resolves this problem using logic gates regulated by the **Sigmoid** activation function ($\sigma$). Since the sigmoid returns values strictly between 0 and 1, these "gates" act like valves: if the gate returns 0 → the channel is closed (information is removed or blocked). If the gate returns 1 → the channel is open (information flows freely and is added or retained).

# The Limitations of RNNs for Sequence Modelling
- **Encoding bottleneck:** We carry all this past information as the network's internal state;
- **Too slow:** There is no parallelisation;
- **Short memory** (vanishing gradients).

So RNNs use this step-by-step method to track temporal dependencies.

Our goal would be to avoid these limitations and achieve the following advantages:
- See the sequence globally as a continuous stream of information (going from "Word 100 can only talk to word 1 through 98 intermediate memory steps" to "Word 100 looks directly at word 1, skipping all the intermediaries");
- Process the sequence in parallel rather than time step by time step;
- Have long memory.

How can we achieve these advantages?
One approach could be to compress everything together: a single input vector, taking all the data from all time steps, concatenating them, obtaining the feature vector (the single concatenated embedded vector), and producing the output.

This eliminates recurrence, but it is not scalable for very large, dense networks; we would also lose the order, and how do we have memory without the concept of time?

## Attention Is All You Need

Transformers are the solution.
But what are Transformers?

Can a neural network identify which parts of the input are most important? Which ones best represent the input?

The solution is to identify which parts are most relevant, and then extract the most important content that has been identified as highly relevant (i.e. with high attention scores).

But how do we figure out which parts are most relevant? Like a classic search.

We have a query; this query produces results; we search the keys to see if anything is linked to our query.
Once we have identified the important part of our data, we extract it.

The goal of **Attention Search** is to compute the mutual relevance between all the words of the sequence in a single pass (globally), allowing the network to focus on the most important information and extract it in a targeted way.

The process is articulated in four fundamental phases:

### 1. Injecting Order (Positional Embedding)
Before performing any attention computation, we must solve the problem of missing temporal coordinates caused by parallelisation (since the network reads the entire sentence simultaneously).

We take the input data matrix, where each row is the *learned embedding* of a word, and **add** the **Positional Embedding** to it:

These are two distinct vectors of numbers, separate at first, which are then merged together. The first vector contains the values representing the semantics of the word; the second vector contains the values representing the position of the word within the sentence.

$$\vec{x}_{\text{final}} = \vec{x}_{\text{semantic}} + \vec{x}_{\text{position}}$$

for example:
$$\vec{x}_{\text{semantic}} = [0.25, -0.87, 0.41, 0.12]$$
$$\vec{x}_{\text{position}} = [0.14, 0.95, -0.32, 0.88]$$

This positional "stamp" is applied **once only at the beginning**. Since the Query ($Q$), Key ($K$), and Value ($V$) matrices will subsequently be generated from this single unified input, all three automatically inherit the information about word order.


### 2. Splitting Roles (Query, Key, Value)
The input matrix (now enriched with positions) is multiplied by three different weight matrices ($W_Q, W_K, W_V$) learned during training.

What are these weights for? Initially, our word has a single vector ($\vec{x}_{\text{final}}$) that fuses together meaning and position. However, a word cannot do everything at once: it cannot present itself as a "subject", search for an "object", and simultaneously provide its deep meaning using the exact same numbers.

This operation projects each word into three vectors with well-defined roles that simulate a relational search system:

* **Query ($Q$) → *"What am I looking for?"***: Represents the current word interrogating the rest of the sentence (the string typed in the search bar).
* **Key ($K$) → *"Who am I and what do I offer?"***: Represents the indexing tag and characteristics of each word, used to be found by others (the index or title in the database).
* **Value ($V$) → *"My actual content"***: Contains the deep semantic information of the word, which will actually be extracted if there is a good match between the Query and its Key.


### 3. Computing Attention Weights (The Match)
To understand which parts of the input are correlated with each other, the network computes the mathematical similarity between every Query and all available Keys via a dot product ($Q \cdot K^T$).

The similarity here is the probability that this word is correlated with the word in K — not that they are identical. Geometric similarity in the dot product indicates logical, grammatical, or semantic correlation: how much two words need each other to complete the meaning of the sentence.

1. The raw scores obtained are divided by a scaling factor ($\sqrt{d_k}$, where $d_k$ is the dimension of the key vectors) to stabilise the computations and prevent gradients from vanishing during backpropagation.
2. The scaled results are fed into a **Softmax** function.

The Softmax transforms the scores into a true ranking of percentages (values strictly between $0$ and $1$, whose sum is always $1$). These are the **Attention Weights**, which numerically indicate how much relevance a given word has with respect to another in the current context.

To understand how the mathematics creates the relevance map, let us put ourselves in the Transformer's shoes while it analyses the sentence:

**"The programmer writes code."**

We want to compute attention with respect to the word **writes** (the verb), to understand which other words in the sentence are fundamental for understanding its context.

Suppose the designers chose a key vector dimension ($d_k$) of **64**. Consequently, our scaling factor will be:

$$\sqrt{d_k} = \sqrt{64} = \mathbf{8}$$

Here are the three exact steps that the GPU executes in parallel:

### Step A: Raw Scores (The dot product $Q \cdot K^T$)
The Query of the word *writes* interacts via a dot product with the Keys of all the words in the sentence. Since the vectors are very high-dimensional (64 elements), the geometric multiplications sum many numbers together, generating high raw scores that are far apart:

* `writes` $\times$ `The` $\rightarrow$ **`8.0`** *(Low: an article gives no context to the verb)*
* `writes` $\times$ `programmer` $\rightarrow$ **`56.0`** *(Very high! It is the subject that performs the action)*
* `writes` $\times$ `writes` $\rightarrow$ **`24.0`** *(Medium: the word's self-reflection)*
* `writes` $\times$ `code` $\rightarrow$ **`40.0`** *(Very high! It is the object, what is being written)*

### Step B: Scaling
If we fed the Softmax a large number like `56.0`, the exponential function of the Softmax ($e^{56}$) would skyrocket to a near-infinite value, zeroing out the gradients of all other words and causing the training to break down (*Vanishing Gradient*).

To stabilise the computations, the network divides all scores by the scaling factor, which in this case is **8**:

* **The** $\rightarrow$ $8.0 / 8 = \mathbf{1.0}$
* **programmer** $\rightarrow$ $56.0 / 8 = \mathbf{7.0}$
* **writes** $\rightarrow$ $24.0 / 8 = \mathbf{3.0}$
* **code** $\rightarrow$ $40.0 / 8 = \mathbf{5.0}$

The computations are now rescaled, protected from mathematical distortions, and ready for the final verdict.

### Step C: Softmax
The Softmax function takes the scaled numbers `[1.0, 7.0, 3.0, 5.0]`, computes the exponential, and normalises them, squashing them into a range between $0$ and $1$ so that their total sum is exactly **1** (i.e. **100%**).

The geometric scores are officially transformed into the true **Attention Weights**:

| Word (Key) | Softmax Computation | Attention Weight (Relevance) | What the Transformer understood |
| :--- | :--- | :--- | :--- |
| **The** | $e^1 / \text{total}$ | **0.2%** (`0.002`) | It is an article — negligible context for the verb. |
| **programmer** | $e^7 / \text{total}$ | **84.3%** (`0.843`) | **Subject.** This is the crucial information for making sense of "writes". |
| **writes** | $e^3 / \text{total}$ | **1.5%** (`0.015`) | Retains a small fraction of self-attention. |
| **code** | $e^5 / \text{total}$ | **14.0%** (`0.140`) | **Object.** This is the second fundamental piece of information in the sentence. |
| **TOTAL** | | **100%** (`1.00`) | |


### 4. Content Extraction (High Attention Extraction)
Once the weights have been identified, we move on to the actual extraction described in the classic search intuition. We multiply the attention weight matrix obtained from the Softmax by the **Value ($V$)** matrix.

Words that have received a high attention score will have their Value extracted almost entirely and passed forward to subsequent layers. Conversely, parts with scores close to zero will be filtered out, attenuated, and ignored by the model.

### Step D: Mathematical Extraction
We have our attention weights for the word *"writes"* computed in Step C. Now the GPU takes the **Value ($V$)** vectors of each word (which contain the deep informational content) and applies a weighted average:

$$\vec{x}_{\text{output\_writes}} = (0.002 \cdot \vec{V}_{\text{The}}) + (0.843 \cdot \vec{V}_{\text{programmer}}) + (0.015 \cdot \vec{V}_{\text{writes}}) + (0.140 \cdot \vec{V}_{\text{code}})$$

What does this new final vector — just generated by the network for the word *"writes"* — contain in terms of meaning?

* The content of **"The"** has been almost entirely filtered and zeroed out ($0.2\%$).
* The original content of **"writes"** retains only a tiny fraction of self-reflection ($1.5\%$).
* The vector is now literally **flooded** with the deep semantic information of **"programmer"** ($84.3\%$) and **"code"** ($14.0\%$).


The final result is a new output vector in which the original word is no longer isolated, but has been **enriched by the entire global context** of the sentence. The verb *"writes"* is no longer an abstract action: it now mathematically carries inside itself the information about *who* is performing the action and *what* is actually being written.

## The Unified Attention Formula

All these logical and matrix-based steps condense mathematically into the equation of **Scaled Dot-Product Attention**:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### Breaking down the formula:
* **$QK^T$** $\rightarrow$ **Phase 3 (The Match):** Geometric multiplication of every Query against all Keys to compute raw similarity and relevance.
* **$\text{softmax}\left(\frac{\dots}{\sqrt{d_k}}\right)$** $\rightarrow$ **Phase 3 (The Weights):** Scaling of values and conversion of scores into stable probabilities/percentages (from $0$ to $1$).
* **$\times V$** $\rightarrow$ **Phase 4 (The Extraction):** Selective, weighted extraction of the most important content based on the high Softmax scores.

**This is the main building block for constructing the Transformer architecture.**

This approach is incredibly parallelisable.

## The Computation Block: The "Self-Attention Head"

All these mathematical operations connect sequentially within a single hardware/software module called a **Self-Attention Head**.

Graphically, the data flow inside a single head follows this structure:
1. The input (enriched by *Positional Encoding*) enters the block.
2. It is split through three parallel linear layers to generate the three matrices: **Query**, **Key**, and **Value**.
3. Matrices $Q$ and $K$ undergo the dot product ($MatMul$) and are scaled ($\sqrt{d_k}$).
4. **Softmax** is applied to obtain the weight matrix.
5. The result is multiplied ($MatMul$) by matrix $V$, outputting the final vector enriched with global context.

This entire structure represents the fundamental building block of the Transformer architecture.
There are no temporal cycles, no recurrence: the entire sentence is processed in a single pass.

## Multi-Head Attention

There is a crucial limitation of a single head: a single Attention Head can only focus on one relationship at a time. For example, in the sentence "She threw the tennis ball to serve", the word *"serve"* needs to connect both to *"tennis"* and to *"threw"*.

To allow the network to view the sentence from multiple perspectives simultaneously, Transformers apply **Multi-Head Attention**, running multiple computation heads **in parallel**.

* **Head 1** focuses on syntactic and grammatical relationships (Subject → Verb).
* **Head 2** focuses on long-range semantic relationships (spatial context).
* **Head 3** analyses the structure of complements or punctuation.

## Fields of Application (Self-Attention Applied)

This parallel spatial computation mechanism has proven so powerful that it has extended far beyond text, revolutionising any data that can be expressed as a sequence:

* **Language Processing (NLP):** It is the foundation of models such as BERT, GPT, and all modern Large Language Models (LLMs).
* **Computer Vision:** In *Vision Transformers (ViT)*, images are divided into small squares (*patches*) treated exactly like words in a sentence.
* **Biological Sequences:** Models such as *AlphaFold* exploit Self-Attention to map interactions between amino acids and predict the three-dimensional folding of proteins and DNA.