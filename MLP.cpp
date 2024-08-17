#include <iostream>
#include "MLP.h"
#include <cstdlib>
#include <ctime>
#include <Eigen/Dense>
#include <string>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <thread>
#include <mutex>
using namespace std;

MLP::MLP(int n_layers, int sizes[]) {
    num_layers = n_layers;
    layer_sizes = new int[n_layers];
    for (int i = 0; i < n_layers; ++i) {
        layer_sizes[i] = sizes[i];
    }

    weights = new Eigen::MatrixXf[n_layers - 1];
        std::random_device rd;
    std::mt19937 gen(rd());
    
    for (int i = 0; i < num_layers - 1; i++) {
        weights[i] = Eigen::MatrixXf(layer_sizes[i + 1], layer_sizes[i]);
        
        // Xavier/Glorot initialization
        float limit = std::sqrt(6.0 / (layer_sizes[i] + layer_sizes[i+1]));
        std::uniform_real_distribution<> dis(-limit, limit);
        
        for (int j = 0; j < layer_sizes[i + 1]; j++) {
            for (int k = 0; k < layer_sizes[i]; k++) {
                weights[i](j, k) = dis(gen);
            }
        }
    }

    biases = new Eigen::VectorXf[num_layers];
    for (int i = 1; i < num_layers; i++) {
        biases[i] = Eigen::VectorXf(layer_sizes[i]);
        // cout << "Initialized biases[" << i << "] with size: "
            //  << layer_sizes[i] << endl;
        for (int j = 0; j < layer_sizes[i]; j++) {
            biases[i](j) = static_cast<float>(rand()) / RAND_MAX - 0.5f;
        }
    }
    back_stages = new Eigen::VectorXf[num_layers];
}

Eigen::VectorXf MLP::relu(Eigen::VectorXf val) {
    for (int i = 0; i < val.size(); i++) {
        if (val(i) < 0) {
            val(i) = 0;
        }
    }
    return val;
}
Eigen::VectorXf MLP::leaky_relu(Eigen::VectorXf val, float alpha = 0.01) {
    return val.array().max(alpha * val.array());
}

Eigen::VectorXf MLP::sigmoid(Eigen::VectorXf val) {
    for (int i = 0; i < val.size(); i++) {
        val(i) = 1 / (1 + exp(-val(i)));
    }
    return val;
}

Eigen::VectorXf MLP::softmax(Eigen::VectorXf val) {
    float sum = 0;
    for (int i = 0; i < val.size(); i++) {
        val(i) = exp(val(i));
        sum += val(i);
    }
    for (int i = 0; i < val.size(); i++) {
        val(i) /= sum;
    }
    return val;
}

Eigen::VectorXf MLP::relu_prime(Eigen::VectorXf val) {
    for (int i = 0; i < val.size(); i++) {
        if (val(i) < 0) {
            val(i) = 0;
        } else {
            val(i) = 1;
        }
    }
    return val;
}

Eigen::VectorXf MLP::sigmoid_prime(Eigen::VectorXf val) {
    for (int i = 0; i < val.size(); i++) {
        val(i) = val(i) * (1 - val(i));
    }
    return val;
}

Eigen::VectorXf MLP::softmax_prime(Eigen::VectorXf val) {
    for (int i = 0; i < val.size(); i++) {
        val(i) = val(i) * (1 - val(i));
    }
    return val;
}
Eigen::VectorXf MLP::leaky_relu_prime(Eigen::VectorXf val, float alpha = 0.01) {
    return (val.array() > 0).select(Eigen::VectorXf::Ones(val.size()), alpha * Eigen::VectorXf::Ones(val.size()));
}

float MLP::loss(const vector<Eigen::VectorXf>& inputs, const vector<Eigen::VectorXf>& targets) {
    float total_loss = 0;
    for (size_t i = 0; i < inputs.size(); i++) {
        Eigen::VectorXf output = forward(inputs[i]);
        total_loss += (output - targets[i]).squaredNorm();
    }
    return total_loss / inputs.size();
}
float MLP::accuracy(const vector<Eigen::VectorXf>& inputs, const vector<Eigen::VectorXf>& targets) {
    int correct = 0;
    for (size_t i = 0; i < inputs.size(); i++) {
        Eigen::VectorXf output = forward(inputs[i]);
        int predicted_label = distance(output.data(), max_element(output.data(), output.data() + output.size()));
        int actual_label = distance(targets[i].data(), max_element(targets[i].data(), targets[i].data() + targets[i].size()));
        if (predicted_label == actual_label) {
            correct++;
        }
    }
    return static_cast<float>(correct) / inputs.size();
}

float MLP::binary_cross_entropy(const std::vector<Eigen::VectorXf>& inputs, const std::vector<Eigen::VectorXf>& targets) {
    float total_loss = 0;
    size_t batch_size = inputs.size();
    const float epsilon = 1e-7;  // Small constant to prevent log(0)

    for (size_t i = 0; i < batch_size; i++) {
        Eigen::VectorXf output = forward(inputs[i]);
        Eigen::VectorXf target = targets[i];

        // Clip output to prevent log(0)
        output = output.array().max(epsilon).min(1 - epsilon);

        // Compute the binary cross-entropy loss
        Eigen::VectorXf loss_terms = -target.array() * output.array().log() - (1 - target.array()) * (1 - output.array()).log();
        total_loss += loss_terms.sum();
    }

    return total_loss / batch_size;
}




Eigen::VectorXf MLP::forward(const Eigen::VectorXf& input) {
    Eigen::VectorXf current = leaky_relu( weights[0] * input + biases[1]);
    back_stages[0] = current;
    for (int i = 1; i < num_layers - 2; i++) {
        current = leaky_relu(weights[i] * current + biases[i + 1]);
        back_stages[i] = current;
        // cout<<"successfully pased layer "<<i<<endl;
    }
    current = softmax(weights[num_layers - 2] * current + biases[num_layers - 1]);
    back_stages[num_layers - 2] = current;
    return current;
}

Eigen::MatrixXf* MLP::compute_gradients_w(const Eigen::VectorXf& input, const Eigen::VectorXf& target) {
    Eigen::VectorXf output = forward(input);
    
    Eigen::MatrixXf* gradients = new Eigen::MatrixXf[num_layers - 1];
    Eigen::VectorXf* deltas = new Eigen::VectorXf[num_layers - 1];

    // Compute delta for the output layer
    deltas[num_layers - 2] = output - target;

    gradients[num_layers - 2] = deltas[num_layers - 2] * back_stages[num_layers - 3].transpose();

    for (int i = num_layers - 3; i >= 0; i--) {
        deltas[i] = (weights[i + 1].transpose() * deltas[i + 1]).cwiseProduct(leaky_relu_prime(back_stages[i]));

        if (i == 0) {
            gradients[i] = deltas[i] * input.transpose();
        } else {
            gradients[i] = deltas[i] * back_stages[i - 1].transpose();
        }
    }

    delete[] deltas;  
    return gradients;
}


Eigen::VectorXf* MLP::compute_gradients_b(const Eigen::VectorXf& input, const Eigen::VectorXf& target) {
    // Allocate memory for gradients, which are the same as deltas
    Eigen::VectorXf* gradients = new Eigen::VectorXf[num_layers];

    // Compute delta for the output layer (which is also the gradient for the output layer's biases)
    gradients[num_layers - 1] = back_stages[num_layers - 2] - target;

    // Backpropagate through the hidden layers
    for (int i = num_layers - 2; i >= 1; i--) {
        gradients[i] = (weights[i].transpose() * gradients[i + 1]).cwiseProduct(leaky_relu_prime(back_stages[i - 1]));
    }

    // Handle the input layer bias gradient
    gradients[0] = (weights[0].transpose() * gradients[1]).cwiseProduct(leaky_relu_prime(input));

    return gradients;
}
float compute_gradient_norm(Eigen::MatrixXf* gradients, int num_layers) {
    float norm = 0;
    for (int i = 0; i < num_layers - 1; i++) {
        norm += gradients[i].squaredNorm(); // Sum of squared norms
    }
    return std::sqrt(norm); // Return the square root to get the Euclidean norm
}
void MLP::clip_gradients(Eigen::MatrixXf* gradients_w, Eigen::VectorXf* gradients_b, float max_norm = 1.0) {
    float total_norm = 0;
    for (int i = 0; i < num_layers - 1; i++) {
        total_norm += gradients_w[i].squaredNorm();
        total_norm += gradients_b[i+1].squaredNorm();
    }
    total_norm = std::sqrt(total_norm);

    if (total_norm > max_norm) {
        float scale = max_norm / total_norm;
        for (int i = 0; i < num_layers - 1; i++) {
            gradients_w[i] *= scale;
            gradients_b[i+1] *= scale;
        }
    }
}

void MLP::stochastic_gradient_descent(const vector<Eigen::VectorXf>& inputs, const vector<Eigen::VectorXf>& targets, const vector<Eigen::VectorXf>& test_x, const vector<Eigen::VectorXf>& test_y, int epochs = 100, int batch_size = 32, float initial_learning_rate  = 0.01) {
    // Create index vector for shuffling
    vector<int> indices(inputs.size());
    for (int i = 0; i < inputs.size(); ++i) {
        indices[i] = i;
    }

    // Create a random number generator
    std::random_device rd;
    std::default_random_engine generator(rd());

    float learning_rate = initial_learning_rate;
    for (int i = 0; i < epochs; i++) {
        // Shuffle the indices before each epoch
        std::shuffle(indices.begin(), indices.end(), generator);
        learning_rate = initial_learning_rate / (1 + 0.01 * i);
        for (int j = 0; j < inputs.size(); j += batch_size) {
            if (j % 100 == 0) {
                cout << "Epoch " << i << ", batch " << j / batch_size << endl;
                cout << "Loss: " << binary_cross_entropy(test_x, test_y) << endl;
                cout << "Accuracy: " << accuracy(test_x, test_y) << endl;
                cout << "--------------------------------" << endl;
            }
            Eigen::MatrixXf* accumulated_gradients_w = nullptr;
            Eigen::VectorXf* accumulated_gradients_b = nullptr;

            int actual_batch_size = min(batch_size, static_cast<int>(inputs.size()) - j);

            for (int b = 0; b < actual_batch_size; b++) {
                int idx = indices[j + b]; // Use shuffled index
                Eigen::MatrixXf* gradients_w = compute_gradients_w(inputs[idx], targets[idx]);
                Eigen::VectorXf* gradients_b = compute_gradients_b(inputs[idx], targets[idx]);

                if (b == 0) {
                    accumulated_gradients_w = gradients_w;
                    accumulated_gradients_b = gradients_b;
                } else {
                    for (int k = 0; k < num_layers - 1; k++) {
                        accumulated_gradients_w[k] += gradients_w[k];
                        accumulated_gradients_b[k + 1] += gradients_b[k + 1];
                    }
                    delete[] gradients_w;
                    delete[] gradients_b;
                }
            }

            for (int k = 0; k < num_layers - 1; k++) {
                weights[k] -= (learning_rate / actual_batch_size) * accumulated_gradients_w[k];
                biases[k + 1] -= (learning_rate / actual_batch_size) * accumulated_gradients_b[k + 1];
            }
            clip_gradients(accumulated_gradients_w, accumulated_gradients_b);

            delete[] accumulated_gradients_w;
            delete[] accumulated_gradients_b;
        }
    }
}



MLP::~MLP() {
    delete[] layer_sizes;
    delete[] weights;
    delete[] biases;
}

