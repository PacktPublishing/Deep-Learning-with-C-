#include <torch/torch.h>
#include <iostream>

// Larger teacher model
struct Teacher : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};

    Teacher(int input_dim, int hidden_dim, int num_classes) {
        fc1 = register_module("fc1", torch::nn::Linear(input_dim, hidden_dim));
        fc2 = register_module("fc2", torch::nn::Linear(hidden_dim, hidden_dim));
        fc3 = register_module("fc3", torch::nn::Linear(hidden_dim, num_classes));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        return fc3->forward(x);  // raw logits
    }
};

// Smaller student model
struct Student : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};

    Student(int input_dim, int hidden_dim, int num_classes) {
        fc1 = register_module("fc1", torch::nn::Linear(input_dim, hidden_dim));
        fc2 = register_module("fc2", torch::nn::Linear(hidden_dim, num_classes));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x));
        return fc2->forward(x);  // raw logits
    }
};

// Distillation loss: alpha * hard_loss + (1 - alpha) * soft_loss
torch::Tensor distillation_loss(
    const torch::Tensor& student_logits,
    const torch::Tensor& teacher_logits,
    const torch::Tensor& targets,
    float temperature,
    float alpha
) {
    // Hard loss: student vs true labels
    auto hard_loss = torch::nn::functional::cross_entropy(student_logits, targets);

    // Soft loss: KL divergence between softened distributions
    auto student_soft = torch::log_softmax(student_logits / temperature, /*dim=*/1);
    auto teacher_soft = torch::softmax(teacher_logits / temperature, /*dim=*/1);
    auto soft_loss = torch::nn::functional::kl_div(
        student_soft, teacher_soft,
        torch::nn::functional::KLDivFuncOptions().reduction(torch::kBatchMean)
    ) ;

    // L_total = α * L_soft * T² + (1-α) * L_hard
    return alpha * soft_loss* (temperature * temperature) + (1.0f - alpha) * hard_loss;
}

int main() {
    int input_dim = 20, num_classes = 5, num_samples = 200;
    float temperature = 4.0f, alpha = 0.3f;

    // Synthetic dataset
    auto data = torch::randn({num_samples, input_dim});
    auto labels = torch::randint(0, num_classes, {num_samples});

    // --- Train teacher ---
    auto teacher = std::make_shared<Teacher>(input_dim, 128, num_classes);
    torch::optim::Adam teacher_opt(teacher->parameters(), 0.001);

    std::cout << "Training teacher..." << std::endl;
    for (int epoch = 0; epoch < 50; ++epoch) {
        teacher_opt.zero_grad();
        auto logits = teacher->forward(data);
        auto loss = torch::nn::functional::cross_entropy(logits, labels);
        loss.backward();
        teacher_opt.step();

        if (epoch % 10 == 0)
            std::cout << "  Epoch " << epoch << ", Loss: " << loss.item<float>() << std::endl;
    }

    // --- Distill to student ---
    auto student = std::make_shared<Student>(input_dim, 32, num_classes);
    torch::optim::Adam student_opt(student->parameters(), 0.001);

    teacher->eval();
    std::cout << "\nDistilling to student (T=" << temperature << ", alpha=" << alpha << ")..." << std::endl;

    for (int epoch = 0; epoch < 100; ++epoch) {
        student_opt.zero_grad();

        auto student_logits = student->forward(data);
        torch::Tensor teacher_logits;
        {
            torch::NoGradGuard no_grad;
            teacher_logits = teacher->forward(data);
        }

        auto loss = distillation_loss(student_logits, teacher_logits, labels, temperature, alpha);
        loss.backward();
        student_opt.step();

        if (epoch % 20 == 0)
            std::cout << "  Epoch " << epoch << ", Loss: " << loss.item<float>() << std::endl;
    }

    // --- Compare accuracy ---
    torch::NoGradGuard no_grad;
    auto teacher_preds = teacher->forward(data).argmax(1);
    auto student_preds = student->forward(data).argmax(1);

    float teacher_acc = teacher_preds.eq(labels).sum().item<float>() / num_samples * 100;
    float student_acc = student_preds.eq(labels).sum().item<float>() / num_samples * 100;
    float agreement = teacher_preds.eq(student_preds).sum().item<float>() / num_samples * 100;

    std::cout << "\nResults:" << std::endl;
    std::cout << "  Teacher accuracy: " << teacher_acc << "%" << std::endl;
    std::cout << "  Student accuracy: " << student_acc << "%" << std::endl;
    std::cout << "  Teacher-Student agreement: " << agreement << "%" << std::endl;
    std::cout << "  Teacher params: " << teacher->parameters().size() << std::endl;
    std::cout << "  Student params: " << student->parameters().size() << std::endl;

    return 0;
}
