# Historical Context: The Era of Generative AI and Edge AI
In **2022**, the official public release of **GPT-3** marked a point of no return: the way we access information and perceive technology changed drastically. Business and development processes that used to take days are now completed in minutes.

The speed of this evolution is lightning fast. Just one year later, in **2023**, the release of **GPT-4** redefined the limits of linguistic understanding in models. However, these enormous architectures shared a structural constraint: a total **dependence on the Cloud**.

## The Paradigm Shift: From Cloud to Local (Edge AI)
The true engineering revolution is happening today. Models with benchmarks similar to or superior to GPT-4 are no longer confined inside massive data centers.

Thanks to innovative architectures, such as those developed by **Liquid AI**, and advanced optimization techniques, Large Language Models (LLMs) can now live **completely on-device**. They can run offline, without an internet connection, while you are on a plane or inside a tunnel, directly on the RAM of consumer devices.

This manual was written to explore the technology behind this quantum leap: **Deep Learning**.

# What is Deep Learning and why now?
Deep Learning is a subfield of Machine Learning, which in turn is part of artificial intelligence. Its main characteristic is that patterns are extracted from raw data through the use of artificial neural networks.

Why is it exploding right now? Three factors:
- Big data: today we have access to massive datasets;
- Hardware: we have reached levels capable of supporting DL, for example highly parallelizable GPUs;
- Software: algorithms have improved enormously, and we have toolboxes like TensorFlow and PyTorch (which we will use throughout this manual).

## The Perceptron, the atom of the neural network.
The perceptron is the fundamental building block, the simplest type of network possible. It works like a simplified biological neuron: it receives inputs, processes them, and produces an output.

The output of a perceptron is defined by the following mathematical formula:

$$y = g \left( w_0 + \sum_{i=1}^{m} x_i w_i \right)$$

which in vector notation becomes:

$$y = g \left(w_0 + X^T W \right)$$

What do these terms mean? Let us break down the formula:

- X = [x_1,...,x_m]^T : the input vector;

- W = [w_1,...,w_m]^T : the weight vector, used to determine how much importance to give to each input;

- w_0 : the bias, which allows shifting the activation function to better fit the data. Think of the Perceptron as a bouncer at the entrance of a club with a dress code. The inputs are the customer's attributes (for example, how well-dressed they are and how old they are).
A very high bias means the bouncer is extremely permissive -- for example, if the club is empty, he lets almost everyone in.
A very low bias means the bouncer is extremely strict -- the club is packed, and he has very high standards, letting in only customers with exceptional inputs.

From a mathematical standpoint, without a bias, the line would always be forced to pass through the origin, meaning the model would only see black or white, with no shades of grey.

For example, if we had to distinguish sick people from healthy ones, without a bias we would probably classify someone with a cold as seriously ill, with zero tolerance;

- g : the non-linear activation function, which we will cover in detail below;

- y : the predicted output.

## The importance of non-linearity
Without the activation function g, the network would be nothing more than a linear combination of inputs, incapable of modeling complex problems.

If we connect a second layer of neurons in cascade, it will take a linear combination as input and produce another linear combination as output. From a purely mathematical standpoint, a linear combination of a linear combination is itself a linear combination. This means that even if we built a gigantic neural network with 100 layers and billions of parameters, without non-linear activation functions, that network would behave mathematically just like the simplest perceptron. We would have built a giant castle that is no more powerful than a small hut.

Imagine training a model to distinguish healthy cells from tumor cells. In the data plot, we find that tumor cells are all concentrated in the center, forming a blob (a circle), while healthy cells surround them.

**Without the activation function g (Linear only):** The model only has "straight lines" at its disposal. It can draw a diagonal, vertical, or horizontal line. But how do we isolate a circle in the center using only a straight line? It is impossible. We would always be forced to cut the dataset in half, accumulating an infinite number of errors. The model has no ability to "curve" its decision boundary.

**With the activation function g (Non-linear):** By introducing functions like ReLU or Sigmoid, which we will cover shortly, we allow the model's mathematics to bend. The network can start combining multiple curved lines to create a perfect boundary around the tumor cells.

The most common activation functions are:

- Sigmoid: $$g(z) = \frac{1}{1 + e^{-z}}$$, useful for probabilities (output between 0 and 1), for example the probability of passing or failing an exam. It is used in the output layer, the final layer, when the network's task is classification (we do not use it in hidden layers due to the Vanishing Gradient problem, which we will cover later);

- ReLU (Rectified Linear Unit): $$g(z) = \max(0, z)$$. The classic ramp, the most widely used for its computational simplicity in hidden layers, which we will see shortly. The secret of its use is computational simplicity -- for a GPU it is a simple if statement, which drastically reduces model training time;

- Tanh: $$g(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$$, very similar to Sigmoid, but it maps the output to values between -1 and 1. Having output centered on zero makes training of the next layer more stable and faster.

# Neural network architecture: the layers
To solve complex problems, perceptrons are organized into layers:

- Input Layer: the incoming data;

- Hidden Layers: intermediate layers that extract increasingly abstract features. If there are many layers, the network is called "Deep";

- Output Layer: provides the final decision, the classification.

In a Dense Layer, which is a fully connected layer, every unit receives input from all units in the previous layer.
This means the layer performs a global mapping of the information that precedes it. Each neuron applies its own set of weights to all received inputs, adds its own bias, and passes the result through the activation function.

$$z_i = w_{0,i} + \sum_{j=1}^{m} x_j w_{j,i}$$

Where $w_{j,i}$ is the weight connecting input j to neuron i.

## Single Layer Neural Network
A Single Layer Neural Network (SLNN) (also commonly called a Single Layer Perceptron Network) is simply a neural network with a single Dense Layer -- the output layer. (No hidden layers)

$$\hat{y}_i = g\!\left(\mathbf{w}_i^\top \mathbf{x} + b_i\right)$$

In vector form (all outputs together):

$$\hat{\mathbf{y}} = g\!\left(W\mathbf{x} + \mathbf{b}\right)$$

## Multi Layer Perceptron (MLP)
The MLP is a neural network composed of multiple dense layers (at least 2), with at least one hidden layer. It is the classic architecture in which information flows in a single direction, from input to output.

## Deep Neural Network
An MLP can be a Deep Network if it has many hidden layers.

$$z_{k,i} = w_{0,i}^{(k)} + \sum_{j=1}^{n_{k-1}} g(z_{k-1,j})\, w_{j,i}^{(k)}$$

where:
- $k$ : index of the current layer;

- $i$ : index of the current neuron in layer $k$;

- $n_{k-1}$ : number of neurons in the previous layer;

- $g(z_{k-1,j})$ : activation of neuron $j$ in the previous layer;

- $w_{j,i}^{(k)}$ : weight connecting neuron $j$ in layer $k-1$ to neuron $i$ in layer $k$;

- $w_{0,i}^{(k)}$ : bias of neuron $i$ in layer $k$

Each successive layer no longer processes raw data (x), but the abstract representations created by the previous layer.

The first layers might detect simple edges or gradients.
The intermediate layers combine edges into shapes.
The final layers recognize complex objects.

To understand this concept, think of LEGO bricks:
- The raw data ($X$) are the individual bricks scattered on the floor;
- The **first layers** join the bricks into small solid structures (e.g. a small wall, a column);
- The **intermediate layers** join the columns and walls to build rooms and towers;
- The **final layers** join the rooms to give the final shape to the castle.

The power of a Deep Neural Network lies in its ability to learn a hierarchy of features through these nested transformations. The deeper the network, the more complex the function it can approximate.

# The Learning Process: Loss and Optimization
Training a network means finding the weights W that minimize the error.
But how do we quantify the error?
Let us analyze the Loss function:
$\mathcal{L}(\hat{y}, y)$

This function measures how far $\hat{y}$ is from $y$. But what is this Loss function made of?

- Mean Squared Error (MSE): Used for regression (e.g. predicting the price of a house). It calculates the square of the difference between the predicted and the real value;

- Binary Cross Entropy: Used for classification (e.g. "Yes" or "No"). It measures how far the predicted probability deviates from the real target;

$$J(W) = \frac{1}{n} \sum_{i=1}^{n} \mathcal{L}\left(f\left(x^{(i)}; W\right), y^{(i)}\right)$$

$J(W)$ measures the average Loss over our neural network. The 1/n term normalizes it to get the average Loss value. Without 1/n it would be the total Loss over the network.

When we apply the **Mean Squared Error**, the network's goal is to predict a continuous numerical value. We want to quantify how far our numerical result deviates from the real one.

$$J(\mathbf{W}) = \frac{1}{n} \sum_{i=1}^{n} \left( y^{(i)} - f\left(x^{(i)}; \mathbf{W}\right)\right)^2$$

Imagine we want to predict the exact grade a student will get on an exam (a value from 18 to 30). If the network predicts $24$ ($\hat{y}$) but the student gets $28$ ($y$), the MSE measures the linear distance between these two real numbers, squares it to penalize the error, and tells the network how to correct itself.

When we apply **Binary Cross-Entropy**, the objective changes completely: we are not looking for just any number, but a probability strictly between 0 and 1. We want to measure how far the certainty computed by the network deviates from reality.

$$J(\mathbf{W}) = -\frac{1}{n} \sum_{i=1}^{n} \left[ y^{(i)} \log\left(f\left(x^{(i)}; \mathbf{W}\right)\right) + \left(1 - y^{(i)}\right) \log\left(1 - f\left(x^{(i)}; \mathbf{W}\right)\right) \right]$$

Imagine predicting whether a student will pass or fail an exam (a binary problem: 1 = Pass, 0 = Fail). If the network outputs $0.95$ ($\hat{y}$), it is saying the student has a 95% probability of passing. If the student unexpectedly fails ($y = 0$), the BCE calculates how much the network is "punished" for having assigned such a high probability to an event that turned out to be false.

## A closer look at Binary Cross Entropy
At first glance it may seem intimidating, but it is actually very simple to understand. Let us break it down to see exactly how it works.

The real target $y^{(i)}$ in binary classification can only take two values: **$1$** (true event) or **$0$** (false event).
Inside the square brackets we notice two distinct blocks separated by a $+$ sign. The value of $y^{(i)}$ acts as a true **software switch**:

- **If the real target is $y^{(i)} = 1$ (the student passes the exam):**
  The second block cancels out completely because it contains the factor $(1 - y^{(i)})$, which is $(1 - 1) = 0$.
  The formula reduces to evaluating only the first piece: **$1 \cdot \log(\hat{y})$**.
- **If the real target is $y^{(i)} = 0$ (the student fails):**
  The first block cancels out because it is multiplied directly by $y^{(i)} = 0$. The second block activates since $(1 - 0) = 1$.
  The formula reduces to evaluating only the second piece: **$1 \cdot \log(1 - \hat{y})$**.

Thanks to this algebraic trick, a single line of code can cleanly handle both positive and negative cases at the same time.

### Why do we use the Logarithm ($\log$)?

The fundamental reason we introduce the logarithm is related to the concept of **distance between probability distributions** and the geometric shape we want to give to our error surface.

The output of our network, $\hat{y} = f(x^{(i)}; \mathbf{W})$, is a probability and is therefore a number strictly confined between $0$ and $1$.
Let us see what happens to the logarithm in this interval:

* The logarithm of $1$ equals $0$: $\log(1) = 0$.
* As we approach $0$, the logarithm tends to minus infinity: $\log(\rightarrow 0) = -\infty$.

Combining this geometric property with the **minus sign ($-$)** at the beginning of the formula, we get a penalty (a Loss) that behaves in an asymmetric and extremely severe way:

#### Case A: The real target is $y = 1$
The network must output a value as close as possible to $1$.
* If the network gets it right and says $\hat{y} = 0.99$, we compute $-\log(0.99) \approx 0.01$ $\rightarrow$ **The Loss is almost zero** (the network is rewarded).
* If the network goes completely off the rails and says $\hat{y} = 0.01$ (99% sure the event is false), we compute $-\log(0.01) \approx 4.60$. As the prediction approaches zero, the value shoots toward infinity $\rightarrow$ **The Loss becomes enormous**.

#### Case B: The real target is $y = 0$
The network must output a value as close as possible to $0$, meaning the term $(1 - \hat{y})$ must approach $1$.
* If the network gets it right and says $\hat{y} = 0.01$, the term becomes $-\log(1 - 0.01) = -\log(0.99) \approx 0.01$ => **The Loss is minimal**.
* If the network is wrong and says $\hat{y} = 0.99$ (sure the student will pass, but they fail), we compute $-\log(1 - 0.99) = -\log(0.01) \approx 4.60$ => **The Loss explodes again**.

### Advantages of BCE over MSE in classification

Why not use the simple squared difference (MSE) for classification as well?
There are two crucial reasons for AI Engineering:

1. **Maximum penalty for confident wrong predictions:** If a network misclassifies using MSE, the maximum error for a single sample is $(1 - 0)^2 = 1$. The network is not punished enough for being "arrogant". BCE, thanks to the logarithmic curve, tends toward infinite punishment. This forces the weights to change immediately.
2. **Mathematical stability of Gradients:** The activation functions used for classification (like the *Sigmoid*) flatten at the extremes (when the output is close to 0 or 1), driving derivatives to zero. If we used MSE, training would stall because the gradients would become too small (Vanishing Gradient). The logarithm in BCE mathematically cancels out this flattening, keeping the gradient flow alive throughout the optimization process.

We want to find the network weights that minimize the error:

$$\mathbf{W}^* = \arg\min_{\mathbf{W}} \frac{1}{n} \sum_{i=1}^{n} \mathcal{L}\left(f\left(x^{(i)}; \mathbf{W}\right), y^{(i)}\right)$$

$$\mathbf{W}^* = \arg\min_{\mathbf{W}} J(\mathbf{W})$$

## Backpropagation
Backpropagation answers the question: if I slightly change a weight $w_j$, by how much will the final error $J(W)$ change?

Remember that our Loss is a function of the network's weights.

Imagine we have this function, and we pick a random point on it.

At this point we compute the gradient at this randomly chosen point. Once the gradient is computed (which gives us the direction in which our Loss increases -- it points toward the maximum), we move in the opposite direction of the gradient, so that we move toward the minimum, where the error tends to be lower.

We keep repeating this process until we converge to the minimum.

### The Gradient Descent Algorithm

- We assign random values to the weights drawn from a normal distribution: $\mathbf{W} \sim \mathcal{N}(0, \sigma^2)$ (e.g. with standard deviation $\sigma = 0.02$).
- We compute the gradient vector of the cost function with respect to the weights: $\nabla_{\mathbf{W}} J(\mathbf{W})$, and update the weights by moving in the opposite direction of the gradient:
     $$\mathbf{W} \leftarrow \mathbf{W} - \eta \nabla_{\mathbf{W}} J(\mathbf{W})$$
- We return the optimal weights $\mathbf{W}^*$ when the algorithm converges (i.e. when the Loss stops decreasing).

The parameter $\eta$ (eta) represents the **Learning Rate**. It controls the size of the "step" we take along the descent. If it is too large we risk overshooting the minimum; if it is too small, training will take days. There is also a proper way to determine the correct learning rate.

### The Chain Rule

Technically, to update a single weight $w_j$, we want to compute the partial derivative of the cost with respect to that weight:

$$\frac{\partial J(\mathbf{W})}{\partial w_j}$$

In a deep neural network, where the input passes through many nested layers before generating the output, computing this derivative directly is impossible. To do so, we use the **Chain Rule** from calculus.

We start from the error generated at the final output and propagate the information backward (*Backpropagation*), layer by layer. For example, the variation of the error with respect to the weight of the first layer ($w_1$) decomposes into the product of local variations:

$$\frac{\partial J(\mathbf{W})}{\partial w_1} = \frac{\partial J(\mathbf{W})}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z_1} \cdot \frac{\partial z_1}{\partial w_1}$$

### Explanation of the factors:
1. $\frac{\partial J(\mathbf{W})}{\partial \hat{y}}$: How much the global error varies as the network's final prediction changes.
2. $\frac{\partial \hat{y}}{\partial z_1}$: How much the neuron's output varies with respect to its internal activation potential ($z_1$).
3. $\frac{\partial z_1}{\partial w_1}$: How much the internal potential varies as the specific weight $w_1$ changes (which depends directly on the input $x$).

This calculation, repeated backward for every single weight in the network, tells us with pinpoint precision in which direction to "shift" the model's billions of weights to descend along the valley of error and approach the optimization objective.

### Is optimizing the Loss difficult?
Optimizing the Loss function can be very complex.

What we have understood is that Loss optimization happens through gradient descent. But how do we descend intelligently?

We need to pay close attention to the Learning Rate, the term $\eta$.
- If we choose a Learning Rate that is too small, we will converge too slowly to the minimum and will likely get stuck in a false minimum (the learning rate is essentially the size of the jump we want to make on our function to check whether the landing point could be the minimum);
- If we choose it too high, and therefore make a giant jump, we will likely diverge rather than converge. We risk exploding the model.
- Adaptive learning rates: instead of a fixed LR, we vary it based on the slope of the terrain (of the function) and the speed at which we are learning.

#### What are the ideas?
1. We try different learning rates and see which ones work best;
2. We use the adaptive learning rate we just mentioned.

So the LR will no longer be fixed but will depend on several factors, such as:
- how large the gradient is;
- how fast we are learning;
- the size of certain specific weights;
- ...

Below are some gradient descent algorithms available in TensorFlow and PyTorch:
- SGD
- Adam
- Adadelta
- Adagrad
- RMSProp

## Mini-Batches and SGD
Computing the gradient on millions of data points is extremely slow. That is why we use Stochastic Gradient Descent (SGD) on small groups of data called Mini-batches, a set of B points:
- Speed: faster computations that exploit GPU parallelism;
- Regularization: the "noise" introduced by computing the gradient on only a few examples helps the network avoid getting stuck in shallow local minima.

The goal of the algorithm is to navigate the mathematical error surface to find the optimal combination of weights $\mathbf{W}^*$ that minimizes the global cost, updating the parameters in small steps (Batches).

#### Weight Initialization
$$\mathbf{W} \sim \mathcal{N}(0, \sigma^2)$$
The algorithm starts by assigning random values to the weights drawn from a normal distribution with mean 0 and a small standard deviation $\sigma^2$ (e.g. $\sigma = 0.02$). This is done to break neuron symmetry and allow learning to begin.

#### Loop until convergence
The process enters an iterative cycle that repeats until the error stabilizes near the global minimum.

#### Mini-Batch Sampling
Instead of processing all $n$ samples in the dataset, the algorithm randomly extracts a small block of data composed of $B$ elements (called *Batch Size*, typically 32, 64, or 128 samples).

#### Local gradient computation
$$\frac{\partial J(\mathbf{W})}{\partial \mathbf{W}} = \frac{1}{B} \sum_{k=1}^{B} \frac{\partial J_k(\mathbf{W})}{\partial \mathbf{W}}$$
The network computes the partial derivative of the cost function with respect to the weights ($\frac{\partial J}{\partial \mathbf{W}}$) by averaging the gradients computed exclusively on the $B$ elements of the current mini-batch.
* **Advantage:** This computation is extremely fast to run on GPU/CPU memory compared to the gradient computed on the entire dataset, ensuring dynamic and responsive learning.

#### Weight Update
$$\mathbf{W} \leftarrow \mathbf{W} - \eta \frac{\partial J(\mathbf{W})}{\partial \mathbf{W}}$$
The current weights are modified by subtracting the gradient computed in the previous step, multiplied by the **Learning Rate**.
The minus sign ($-$) is crucial: since the gradient points in the direction of maximum increase in error, moving in the opposite direction guarantees descending along the valley toward the point of minimum error.

#### Return of Optimal Weights
When the algorithm converges and the Loss stops decreasing, the cycle ends and the model returns the final matrix of optimized weights ($\mathbf{W}^*$).

To understand the engineering need for mini-batches, imagine a student who has to study a very heavy **1,000-page** exam textbook (our entire dataset). The student's goal is to understand the concepts and correct their study method (update the weights $\mathbf{W}$) to get a perfect score (minimize the Loss).

There are three ways the student can approach studying:

#### The full Batch approach
The student decides to read **all 1,000 pages in one sitting** without ever stopping, without asking questions, and without doing any self-assessment quizzes. Only after closing the last page do they sit down, take stock of what they understood, and correct their study method.
* **The problem:** This is an extremely slow process. The student takes weeks before making a single adjustment. Also, their memory (the computer's RAM) is overloaded by an enormous amount of information all at once.

#### The Stochastic approach (SGD)
The student reads **a single line from the first page** and immediately stops to take a test. If they get the question about that line wrong, they completely overhaul their study method. Then they read the second line, stop, take the test again, change method again, and so on for 1,000 pages.
* **The problem:** This approach is frantic and chaotic. The study method constantly changes direction at every single line, picking up insignificant details or exceptions, creating a fragmented path that oscillates continuously without ever finding a balanced study strategy.

#### The Mini-Batch approach, the middle ground
The student finds the perfect compromise: they divide the book into **small chapters of 32 or 64 pages** (our *Batch Size*, $B$).
They read the first block of pages, stop, take a quiz on that chapter, and based on the average score, correct and refine their study method. Then they move on to the next block.

# Overfitting and Regularization
One of the most common problems in training neural networks is Overfitting: the network memorizes the training data (noise included) but fails on new data.

In reality there are 2 problems:
- Overfitting: which we just described -- the network memorizes and cannot generalize, meaning it cannot correctly answer questions it has not seen before;
- Underfitting: when the model does not have the capacity to learn well from the data -- the example we saw at the beginning about black and white. The model is too simple, cannot capture the complexity of the data, and makes decisions that are too blunt and coarse, exactly like a perceptron without a bias that sees only black and white.

We therefore need a correct fit, meaning we must find the right balance between simplicity and complexity.

How do we do it? With regularization.
Regularization is a set of techniques that penalize model complexity during training, preventing it from memorizing the training data and forcing it to learn generalizable patterns.

Here are some of the main regularization techniques:
- Dropout;
- Early Stopping.
They can also be used in combination.

***Dropout*** is a technique that works as follows: during training, we randomly turn off a percentage of neurons (e.g. 50%) at each pass.
Why does it work? It forces the network not to rely on any single specific neuron, distributing knowledge more robustly across the entire architecture. It reduces overfitting precisely because it prevents any neuron from blindly trusting what the others tell it -- it has to start understanding things on its own.

**Why does it work?** Imagine a software development team of 5 people where only one person (the "genius neuron") does all the work and the other 4 just watch. If the genius gets sick, the team fails (Overfitting on a single element).
Dropout forces the company to randomly send employees on mandatory leave every day. If the genius is not there today, the other 4 are forced to learn to code and collaborate. This way, knowledge is distributed robustly across the entire architecture: the network learns to survive and perform regardless of whether any single specific neuron is present.

***Early Stopping***: we monitor the error on a dataset the network has never seen (the test set). When the error on training decreases but the error on testing starts to rise, we stop everything: the network is starting to memorize instead of understand.

**Why does it work?** Imagine a student who practices for weeks on the exact same past exam papers.
In the first few days, the student understands the general rules of the subject (the Loss decreases both on past exams and on new questions posed by the professor).
After the third week of intensive studying, the student starts memorizing that *"If the question contains the word X, then the answer is C"*. They have stopped reasoning. If the professor changes a single word on the actual exam, the student fails.
Early Stopping acts like a timer that takes the book out of the student's hands as soon as it notices they are starting to memorize the answers to old exams instead of studying the theory.

**We have therefore covered:**
- the concept of the Perceptron;
- the concept of a neural network: how we go from a perceptron to a network, and optimization through backpropagation;
- neural network training: adaptive training, batching, and regularization.