import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Set parameters
tf.compat.v1.enable_eager_execution()
learning_rate = 0.01
training_steps = 1000
display_step = 50
optimiser = tf.keras.optimizers.SGD(learning_rate) # Stochastic Gradient Descent optimiser

# Training Data
X = np.random.uniform(0, 10, 20)
Y = np.random.uniform(0, 3.5, 20)

# X = np.array([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167, 7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1])
# Y = np.array([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221, 2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])

# Weight and Bias initialised randomly
weight = tf.Variable(np.random.randn(), name = "weight")
bias = tf.Variable(np.random.randn(), name = "bias")

# Linear regression 
# y = mx + c
# Y = (weight * X) + bias
def linear_regression(x):
    return (weight * x) + bias

# Mean square error
def mean_square_error(y_predicted, y_actual):
    return tf.reduce_mean(tf.square(y_predicted - y_actual))

def run_optimisation():
    # Wrap computation inside a Gradient Tape for automatic differentiation
    with tf.GradientTape() as g:
        predicted = linear_regression(X)
        loss = mean_square_error(predicted, Y)

    # Compute gradients
    gradients = g.gradient(loss, [weight, bias])

    # update weight and bias following gradients
    optimiser.apply_gradients(zip(gradients, [weight, bias]))

# Run training for the given number of steps
def main():
    for each_step in range(1, training_steps + 1):
        # Run the optimisation to update the weight and bias values
        run_optimisation()

        if each_step % display_step == 0:
            predicted = linear_regression(X)
            loss = mean_square_error(predicted, Y)
            print(f"step: {each_step}   loss: {loss}    weight: {weight.numpy()}    bias: {bias.numpy()}")
    
    plot(X, Y, weight, bias)

# Graphic display
def plot(X, Y, weight, bias):
    plt.plot(X, Y, 'ro', label = "Training Data")
    plt.plot(X, np.array((weight * X) + bias), label = "Fitted Line")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
