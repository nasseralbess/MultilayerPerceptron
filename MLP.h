#include <vector>
#include <Eigen/Dense>
using namespace std;
class MLP {
    public:
        MLP(int n_layers, int sizes[]);
        Eigen::VectorXf forward(const Eigen::VectorXf& input);
        float loss(const vector<Eigen::VectorXf>& inputs, const vector<Eigen::VectorXf>& targets);        
        void stochastic_gradient_descent(const vector<Eigen::VectorXf>& inputs, const vector<Eigen::VectorXf>& targets, const vector<Eigen::VectorXf>& test_x, const vector<Eigen::VectorXf>& test_y,  int epochs, int batch_size, float learning_rate);
        float accuracy(const vector<Eigen::VectorXf>& inputs, const vector<Eigen::VectorXf>& targets);
        float binary_cross_entropy(const std::vector<Eigen::VectorXf>& inputs, const std::vector<Eigen::VectorXf>& targets);
    ~MLP();
    private:
        int num_layers;
        int* layer_sizes;
        Eigen::MatrixXf* weights; 
        Eigen::VectorXf* biases;  
        Eigen::VectorXf* back_stages;
        Eigen::VectorXf relu(Eigen::VectorXf val);
        Eigen::VectorXf sigmoid(Eigen::VectorXf val);
        Eigen::VectorXf softmax(Eigen::VectorXf val);
        Eigen::VectorXf relu_prime(Eigen::VectorXf val);
        Eigen::VectorXf sigmoid_prime(Eigen::VectorXf val);
        Eigen::VectorXf softmax_prime(Eigen::VectorXf val);
        Eigen::MatrixXf* compute_gradients_w(const Eigen::VectorXf& input, const Eigen::VectorXf& target);
        Eigen::VectorXf* compute_gradients_b(const Eigen::VectorXf& input, const Eigen::VectorXf& target);
        void clip_gradients(Eigen::MatrixXf* gradients_w, Eigen::VectorXf* gradients_b, float max_norm );
        Eigen::VectorXf leaky_relu_prime(Eigen::VectorXf val, float alpha );
        Eigen::VectorXf leaky_relu(Eigen::VectorXf val, float alpha);
};