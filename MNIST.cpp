#include <fstream>
#include <vector>
#include <iostream>
#include <Eigen/Dense>
#include "MLP.cpp"
// Function to read integers in big-endian format
int readInt(std::ifstream& file) {
    int value = 0;
    file.read(reinterpret_cast<char*>(&value), 4);
    value = __builtin_bswap32(value); // convert from big-endian to little-endian
    return value;
}

// Function to load MNIST images
std::vector<Eigen::VectorXf> loadMNISTImages(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filepath);
    }

    int magicNumber = readInt(file);
    int numImages = readInt(file);
    int numRows = readInt(file);
    int numCols = readInt(file);

    std::vector<Eigen::VectorXf> images(numImages, Eigen::VectorXf(numRows * numCols));

    for (int i = 0; i < numImages; ++i) {
        for (int j = 0; j < numRows * numCols; ++j) {
            unsigned char pixel = 0;
            file.read(reinterpret_cast<char*>(&pixel), 1);
            images[i](j) = pixel / 255.0f; // Normalize pixel values to [0, 1]
        }
    }
    return images;
}

// Function to load MNIST labels
std::vector<int> loadMNISTLabels(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filepath);
    }

    int magicNumber = readInt(file);
    int numLabels = readInt(file);

    std::vector<int> labels(numLabels);
    for (int i = 0; i < numLabels; ++i) {
        unsigned char label = 0;
        file.read(reinterpret_cast<char*>(&label), 1);
        labels[i] = static_cast<int>(label);
    }
    return labels;
}

int main() {
    //set random seed
    srand(time(0));
    // Load MNIST training and testing data
    std::vector<Eigen::VectorXf> train_images = loadMNISTImages("train-images.idx3-ubyte");
    std::vector<int> train_labels = loadMNISTLabels("train-labels.idx1-ubyte");
    
    std::vector<Eigen::VectorXf> test_images = loadMNISTImages("t10k-images.idx3-ubyte");
    std::vector<int> test_labels = loadMNISTLabels("t10k-labels.idx1-ubyte");

    // Convert labels to one-hot encoding for training
    std::vector<Eigen::VectorXf> train_targets(train_labels.size(), Eigen::VectorXf::Zero(10));
    for (size_t i = 0; i < train_labels.size(); ++i) {
        train_targets[i](train_labels[i]) = 1.0f;
    }
    // Convert labels to one-hot encoding for testing
    std::vector<Eigen::VectorXf> test_targets(test_labels.size(), Eigen::VectorXf::Zero(10));
    for (size_t i = 0; i < test_labels.size(); ++i) {
        test_targets[i](test_labels[i]) = 1.0f;
    }

    // Create and train the MLP
    MLP mlp(4, new int[4]{784, 128, 64, 10});
    mlp.stochastic_gradient_descent(train_images, train_targets, test_images, test_targets, 10, 64, 0.01);


    

    return 0;
}
