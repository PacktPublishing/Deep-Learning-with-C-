#include <Eigen/Dense>
#include <iostream>
#include <cmath>

using namespace Eigen;

class Optimizer {
public:
    virtual void update(MatrixXd& weights, const MatrixXd& grad) = 0;
    virtual ~Optimizer() = default;
};

class GradientDescent : public Optimizer {
    double lr;
public:
    GradientDescent(double learning_rate = 0.01) : lr(learning_rate) {}
    void update(MatrixXd& weights, const MatrixXd& grad) override {
        weights -= lr * grad;
    }
};

class Momentum : public Optimizer {
    double lr, beta;
    MatrixXd velocity;
public:
    Momentum(double learning_rate = 0.01, double momentum = 0.9) : lr(learning_rate), beta(momentum) {}
    void update(MatrixXd& weights, const MatrixXd& grad) override {
        if (velocity.size() == 0) velocity = MatrixXd::Zero(weights.rows(), weights.cols());
        velocity = beta * velocity + (1 - beta) * grad;
        weights -= lr * velocity;
    }
};

class RMSprop : public Optimizer {
    double lr, beta, eps;
    MatrixXd cache;
public:
    RMSprop(double learning_rate = 0.001, double decay = 0.9, double epsilon = 1e-8) 
        : lr(learning_rate), beta(decay), eps(epsilon) {}
    void update(MatrixXd& weights, const MatrixXd& grad) override {
        if (cache.size() == 0) cache = MatrixXd::Zero(weights.rows(), weights.cols());
        cache = beta * cache + (1 - beta) * grad.array().square().matrix();
        weights -= lr * grad.array() / (cache.array().sqrt() + eps).matrix();
    }
};

class Adam : public Optimizer {
    double lr, beta1, beta2, eps;
    MatrixXd m, v;
    int t = 0;
public:
    Adam(double learning_rate = 0.001, double b1 = 0.9, double b2 = 0.999, double epsilon = 1e-8)
        : lr(learning_rate), beta1(b1), beta2(b2), eps(epsilon) {}
    void update(MatrixXd& weights, const MatrixXd& grad) override {
        if (m.size() == 0) {
            m = MatrixXd::Zero(weights.rows(), weights.cols());
            v = MatrixXd::Zero(weights.rows(), weights.cols());
        }
        t++;
        m = beta1 * m + (1 - beta1) * grad;
        v = beta2 * v + (1 - beta2) * grad.array().square().matrix();
        MatrixXd m_hat = m / (1 - std::pow(beta1, t));
        MatrixXd v_hat = v / (1 - std::pow(beta2, t));
        weights -= lr * m_hat.array() / (v_hat.array().sqrt() + eps).matrix();
    }
};

class AdaGrad : public Optimizer {
    double lr, eps;
    MatrixXd cache;
public:
    AdaGrad(double learning_rate = 0.01, double epsilon = 1e-8) : lr(learning_rate), eps(epsilon) {}
    void update(MatrixXd& weights, const MatrixXd& grad) override {
        if (cache.size() == 0) cache = MatrixXd::Zero(weights.rows(), weights.cols());
        cache += grad.array().square().matrix();
        weights -= lr * grad.array() / (cache.array().sqrt() + eps).matrix();
    }
};

class AdaDelta : public Optimizer {
    double rho, eps;
    MatrixXd eg, edx;
public:
    AdaDelta(double decay = 0.95, double epsilon = 1e-6) : rho(decay), eps(epsilon) {}
    void update(MatrixXd& weights, const MatrixXd& grad) override {
        if (eg.size() == 0) {
            eg = MatrixXd::Zero(weights.rows(), weights.cols());
            edx = MatrixXd::Zero(weights.rows(), weights.cols());
        }
        eg = rho * eg + (1 - rho) * grad.array().square().matrix();
        MatrixXd dx = -grad.array() * ((edx.array() + eps).sqrt() / (eg.array() + eps).sqrt()).matrix();
        edx = rho * edx + (1 - rho) * dx.array().square().matrix();
        weights += dx;
    }
};

void test_optimizer(Optimizer* opt, const std::string& name) {
    MatrixXd w = MatrixXd::Random(2, 2);
    std::cout << name << " - Initial: " << w.norm() << " -> ";
    for (int i = 0; i < 100; i++) {
        MatrixXd grad = 2 * w;
        opt->update(w, grad);
    }
    std::cout << "Final: " << w.norm() << "\n";
}

int main() {
    test_optimizer(new GradientDescent(0.01), "GradientDescent");
    test_optimizer(new Momentum(0.01, 0.9), "Momentum");
    test_optimizer(new RMSprop(0.1), "RMSprop");
    test_optimizer(new Adam(0.1), "Adam");
    test_optimizer(new AdaGrad(0.5), "AdaGrad");
    test_optimizer(new AdaDelta(0.95), "AdaDelta");
    return 0;
}
