# feedforward_neuralnetwork
a simple python implementation of a feed forward neural network using adam as a gradient descent optimizer using python .


##Required dependencies :
-Numpy,
-Pandas,&
-Matplotlib

Gentle Introduction to the Adam Optimization Algorithm for Deep Learning
by Jason Brownlee on July 3, 2017 in Deep Learning
The choice of optimization algorithm for your deep learning model can mean the difference between good results in minutes, hours, and days.

The Adam optimization algorithm is an extension to stochastic gradient descent that has recently seen broader adoption for deep learning applications in computer vision and natural language processing.


###What is the Adam optimization algorithm?


Adam is an optimization algorithm that can used instead of the classical stochastic gradient descent procedure to update network weights iterative based in training data.

Adam was presented by Diederik Kingma from OpenAI and Jimmy Ba from the University of Toronto in their 2015 ICLR paper (poster) titled “Adam: A Method for Stochastic Optimization“. I will quote liberally from their paper in this post, unless stated otherwise.

The algorithm is called Adam. It is not an acronym and is not written as “ADAM”.

… the name Adam is derived from adaptive moment estimation.

When introducing the algorithm, the authors list the attractive benefits of using Adam on non-convex optimization problems, as follows:

Straightforward to implement.
Computationally efficient.
Little memory requirements.
Invariant to diagonal rescale of the gradients.
Well suited for problems that are large in terms of data and/or parameters.
Appropriate for non-stationary objectives.
Appropriate for problems with very noisy/or sparse gradients.
Hyper-parameters have intuitive interpretation and typically require little tuning.

####Adam Configuration Parameters

alpha. Also referred to as the learning rate or step size. The proportion that weights are updated (e.g. 0.001). Larger values (e.g. 0.3) results in faster initial learning before the rate is updated. Smaller values (e.g. 1.0E-5) slow learning right down during training
beta1. The exponential decay rate for the first moment estimates (e.g. 0.9).
beta2. The exponential decay rate for the second-moment estimates (e.g. 0.999). This value should be set close to 1.0 on problems with a sparse gradient (e.g. NLP and computer vision problems).
epsilon. Is a very small number to prevent any division by zero in the implementation (e.g. 10E-8).
