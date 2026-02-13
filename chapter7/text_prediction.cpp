#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <random>
#include <queue>
#include <memory>
#include <cmath>

// Simple matrix class to replace Eigen
class Matrix {
public:
    std::vector<std::vector<float>> data;
    int rows, cols;
    
    Matrix(int r, int c) : rows(r), cols(c) {
        data.resize(r, std::vector<float>(c, 0.0f));
    }
    
    float& operator()(int i, int j) { return data[i][j]; }
    const float& operator()(int i, int j) const { return data[i][j]; }
    
    Matrix operator*(const Matrix& other) const {
        Matrix result(rows, other.cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < other.cols; ++j) {
                for (int k = 0; k < cols; ++k) {
                    result(i, j) += data[i][k] * other(k, j);
                }
            }
        }
        return result;
    }
    
    Matrix operator+(const Matrix& other) const {
        Matrix result(rows, cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                result(i, j) = data[i][j] + other(i, j);
            }
        }
        return result;
    }
    
    void randomize() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, 0.1f);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                data[i][j] = dist(gen);
            }
        }
    }
};

class LSTMCell {
public:
    struct LSTMState {
        Matrix h, c;
        Matrix output;
        LSTMState(int hidden_size, int output_size) 
            : h(hidden_size, 1), c(hidden_size, 1), output(output_size, 1) {}
    };
    
    struct LSTMResult {
        std::vector<Matrix> outputs;
        std::vector<LSTMState> states;
    };
    
private:
    int input_size, hidden_size, output_size;
    Matrix Wf, Wi, Wo, Wg; // LSTM gates
    Matrix Uf, Ui, Uo, Ug; // Hidden weights
    Matrix bf, bi, bo, bg; // Biases
    Matrix Wy, by;         // Output layer
    
    float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
    float tanh_act(float x) { return std::tanh(x); }
    
public:
    LSTMCell(int input_sz, int hidden_sz, int output_sz) 
        : input_size(input_sz), hidden_size(hidden_sz), output_size(output_sz),
          Wf(hidden_sz, input_sz), Wi(hidden_sz, input_sz), Wo(hidden_sz, input_sz), Wg(hidden_sz, input_sz),
          Uf(hidden_sz, hidden_sz), Ui(hidden_sz, hidden_sz), Uo(hidden_sz, hidden_sz), Ug(hidden_sz, hidden_sz),
          bf(hidden_sz, 1), bi(hidden_sz, 1), bo(hidden_sz, 1), bg(hidden_sz, 1),
          Wy(output_sz, hidden_sz), by(output_sz, 1) {
        
        Wf.randomize(); Wi.randomize(); Wo.randomize(); Wg.randomize();
        Uf.randomize(); Ui.randomize(); Uo.randomize(); Ug.randomize();
        Wy.randomize();
    }
    
    LSTMState forward(const Matrix& input, const LSTMState& prev_state) {
        LSTMState state(hidden_size, output_size);
        
        // Simplified LSTM forward pass
        for (int i = 0; i < hidden_size; ++i) {
            float f = 0, inp = 0, o = 0, g = 0;
            
            for (int j = 0; j < input_size; ++j) {
                f += Wf(i, j) * input(j, 0);
                inp += Wi(i, j) * input(j, 0);
                o += Wo(i, j) * input(j, 0);
                g += Wg(i, j) * input(j, 0);
            }
            
            for (int j = 0; j < hidden_size; ++j) {
                f += Uf(i, j) * prev_state.h(j, 0);
                inp += Ui(i, j) * prev_state.h(j, 0);
                o += Uo(i, j) * prev_state.h(j, 0);
                g += Ug(i, j) * prev_state.h(j, 0);
            }
            
            f = sigmoid(f + bf(i, 0));
            inp = sigmoid(inp + bi(i, 0));
            o = sigmoid(o + bo(i, 0));
            g = tanh_act(g + bg(i, 0));
            
            state.c(i, 0) = f * prev_state.c(i, 0) + inp * g;
            state.h(i, 0) = o * tanh_act(state.c(i, 0));
        }
        
        // Output layer
        for (int i = 0; i < output_size; ++i) {
            float sum = by(i, 0);
            for (int j = 0; j < hidden_size; ++j) {
                sum += Wy(i, j) * state.h(j, 0);
            }
            state.output(i, 0) = sum;
        }
        
        return state;
    }
    
    LSTMResult forwardSequence(const std::vector<Matrix>& inputs) {
        LSTMResult result;
        LSTMState state(hidden_size, output_size);
        
        for (const auto& input : inputs) {
            state = forward(input, state);
            result.outputs.push_back(state.output);
            result.states.push_back(state);
        }
        
        return result;
    }
};

class TextPredictor {
private:
    std::unique_ptr<LSTMCell> lstm;
    std::unordered_map<std::string, int> word_to_id;
    std::unordered_map<int, std::string> id_to_word;
    int vocab_size;
    int hidden_size;
    int sequence_length;
    std::string start_token = "<START>";
    std::string end_token = "<END>";
    std::string unk_token = "<UNK>";
    
public:
    TextPredictor(int hidden_sz = 128, int seq_len = 20) 
        : hidden_size(hidden_sz), sequence_length(seq_len), vocab_size(0) {}
    
    // Build vocabulary from training corpus
    void buildVocabulary(const std::vector<std::string>& corpus, int min_freq = 2) {
        std::unordered_map<std::string, int> word_counts;
        
        // Count word frequencies
        for (const auto& text : corpus) {
            std::istringstream iss(text);
            std::string word;
            while (iss >> word) {
                // Simple preprocessing
                std::transform(word.begin(), word.end(), word.begin(), ::tolower);
                word_counts[word]++;
            }
        }
        
        // Add special tokens
        word_to_id[start_token] = vocab_size++;
        word_to_id[end_token] = vocab_size++;
        word_to_id[unk_token] = vocab_size++;
        
        // Add frequent words to vocabulary
        std::vector<std::pair<std::string, int>> sorted_words;
        for (const auto& pair : word_counts) {
            if (pair.second >= min_freq) {
                sorted_words.push_back(pair);
            }
        }
        
        std::sort(sorted_words.begin(), sorted_words.end(),
                 [](const auto& a, const auto& b) { return a.second > b.second; });
        
        for (const auto& pair : sorted_words) {
            word_to_id[pair.first] = vocab_size++;
        }
        
        // Build reverse mapping
        for (const auto& pair : word_to_id) {
            id_to_word[pair.second] = pair.first;
        }
        
        // Initialize LSTM with vocabulary size
        lstm = std::make_unique<LSTMCell>(vocab_size, hidden_size, vocab_size);
        
        std::cout << "Built vocabulary with " << vocab_size << " words" << std::endl;
    }
    
    std::vector<int> textToSequence(const std::string& text) {
        std::vector<int> sequence;
        std::istringstream iss(text);
        std::string word;
        
        sequence.push_back(word_to_id[start_token]);
        
        while (iss >> word) {
            std::transform(word.begin(), word.end(), word.begin(), ::tolower);
            auto it = word_to_id.find(word);
            if (it != word_to_id.end()) {
                sequence.push_back(it->second);
            } else {
                sequence.push_back(word_to_id[unk_token]);
            }
        }
        
        sequence.push_back(word_to_id[end_token]);
        return sequence;
    }
    
    // Convert sequence of IDs back to text
    std::string sequenceToText(const std::vector<int>& sequence) {
        std::string text;
        for (int id : sequence) {
            if (id == word_to_id[start_token] || id == word_to_id[end_token]) {
                continue;
            }
            auto it = id_to_word.find(id);
            if (it != id_to_word.end()) {
                if (!text.empty()) text += " ";
                text += it->second;
            }
        }
        return text;
    }
    
    Matrix createOneHot(int word_id) {
        Matrix one_hot(vocab_size, 1);
        if (word_id >= 0 && word_id < vocab_size) {
            one_hot(word_id, 0) = 1.0f;
        }
        return one_hot;
    }
    
    // Training function
    void train(const std::vector<std::string>& corpus, int epochs = 100, float learning_rate = 0.001f) {
        std::cout << "Starting training for " << epochs << " epochs..." << std::endl;
        
        // Convert corpus to sequences
        std::vector<std::vector<int>> sequences;
        for (const auto& text : corpus) {
            auto seq = textToSequence(text);
            if (seq.size() > 2) { // Must have at least start, one word, end
                sequences.push_back(seq);
            }
        }
        
        std::random_device rd;
        std::mt19937 gen(rd());
        
        for (int epoch = 0; epoch < epochs; ++epoch) {
            float total_loss = 0.0f;
            int num_batches = 0;
            
            // Shuffle sequences
            std::shuffle(sequences.begin(), sequences.end(), gen);
            
            for (const auto& sequence : sequences) {
                if (sequence.size() < 2) continue;
                
                std::vector<Matrix> inputs;
                std::vector<Matrix> targets;
                
                for (size_t i = 0; i < sequence.size() - 1; ++i) {
                    inputs.push_back(createOneHot(sequence[i]));
                    targets.push_back(createOneHot(sequence[i + 1]));
                }
                
                // Forward pass
                auto result = lstm->forwardSequence(inputs);
                
                float batch_loss = 0.0f;
                for (size_t t = 0; t < result.outputs.size(); ++t) {
                    Matrix softmax_output = applySoftmax(result.outputs[t]);
                    float dot_product = 0.0f;
                    for (int i = 0; i < vocab_size; ++i) {
                        dot_product += softmax_output(i, 0) * targets[t](i, 0);
                    }
                    batch_loss -= std::log(std::max(dot_product, 1e-10f));
                }
                batch_loss /= result.outputs.size();
                
                total_loss += batch_loss;
                num_batches++;
                
                // Backward pass (simplified - would need full BPTT implementation)
                // For brevity, we'll use a simplified update
                trainStep(inputs, targets, learning_rate);
            }
            
            if (epoch % 10 == 0) {
                std::cout << "Epoch " << epoch << ", Average Loss: " 
                         << total_loss / num_batches << std::endl;
            }
        }
        
        std::cout << "Training completed!" << std::endl;
    }
    
    Matrix applySoftmax(const Matrix& logits) {
        Matrix result(logits.rows, 1);
        float max_val = logits(0, 0);
        for (int i = 1; i < logits.rows; ++i) {
            max_val = std::max(max_val, logits(i, 0));
        }
        
        float sum = 0.0f;
        for (int i = 0; i < logits.rows; ++i) {
            result(i, 0) = std::exp(logits(i, 0) - max_val);
            sum += result(i, 0);
        }
        
        for (int i = 0; i < logits.rows; ++i) {
            result(i, 0) /= sum;
        }
        return result;
    }
    
    void trainStep(const std::vector<Matrix>& inputs,
                   const std::vector<Matrix>& targets,
                   float learning_rate) {
        // Simplified training step - just forward pass for now
        auto result = lstm->forwardSequence(inputs);
    }
    
    // Predict next word given context
    std::string predictNextWord(const std::string& context) {
        auto sequence = textToSequence(context);
        
        std::vector<Matrix> inputs;
        for (int id : sequence) {
            inputs.push_back(createOneHot(id));
        }
        
        // Forward pass
        auto result = lstm->forwardSequence(inputs);
        
        if (result.outputs.empty()) {
            return unk_token;
        }
        
        Matrix probabilities = applySoftmax(result.outputs.back());
        
        int best_id = 0;
        float best_prob = probabilities(0, 0);
        for (int i = 1; i < vocab_size; ++i) {
            if (probabilities(i, 0) > best_prob) {
                best_prob = probabilities(i, 0);
                best_id = i;
            }
        }
        
        auto it = id_to_word.find(best_id);
        return (it != id_to_word.end()) ? it->second : unk_token;
    }
    
    // Beam search for text generation
    struct BeamCandidate {
        std::vector<int> sequence;
        float score;
        LSTMCell::LSTMState state;
        
        bool operator<(const BeamCandidate& other) const {
            return score < other.score; // For max-heap
        }
    };
    
    std::string generateText(const std::string& seed, int max_length = 50, int beam_width = 5) {
        auto seed_sequence = textToSequence(seed);
        
        // Initialize beam with seed sequence
        std::priority_queue<BeamCandidate> beam;
        
        std::vector<Matrix> seed_inputs;
        for (int id : seed_sequence) {
            seed_inputs.push_back(createOneHot(id));
        }
        
        auto seed_result = lstm->forwardSequence(seed_inputs);
        
        BeamCandidate initial_candidate;
        initial_candidate.sequence = seed_sequence;
        initial_candidate.score = 0.0f;
        if (!seed_result.states.empty()) {
            initial_candidate.state = seed_result.states.back();
        }
        
        beam.push(initial_candidate);
        
        // Generate tokens
        for (int step = 0; step < max_length; ++step) {
            std::vector<BeamCandidate> candidates;
            
            // Expand each beam candidate
            while (!beam.empty() && candidates.size() < beam_width * vocab_size) {
                BeamCandidate current = beam.top();
                beam.pop();
                
                // Check if sequence is complete
                if (!current.sequence.empty() && 
                    current.sequence.back() == word_to_id[end_token]) {
                    candidates.push_back(current);
                    continue;
                }
                
                Matrix input = createOneHot(current.sequence.back());
                auto next_state = lstm->forward(input, current.state);
                Matrix probabilities = applySoftmax(next_state.output);
                
                // Add top candidates
                std::vector<std::pair<float, int>> top_tokens;
                for (int i = 0; i < vocab_size; ++i) {
                    top_tokens.push_back({probabilities(i, 0), i});
                }
                
                std::sort(top_tokens.rbegin(), top_tokens.rend());
                
                for (int i = 0; i < std::min(beam_width, (int)top_tokens.size()); ++i) {
                    BeamCandidate new_candidate;
                    new_candidate.sequence = current.sequence;
                    new_candidate.sequence.push_back(top_tokens[i].second);
                    new_candidate.score = current.score + std::log(top_tokens[i].first + 1e-10f);
                    new_candidate.state = next_state;
                    
                    candidates.push_back(new_candidate);
                }
            }
            
            // Keep top beam_width candidates
            std::sort(candidates.rbegin(), candidates.rend(),
                     [](const BeamCandidate& a, const BeamCandidate& b) {
                         return a.score < b.score;
                     });
            
            // Clear beam and add top candidates
            while (!beam.empty()) beam.pop();
            
            for (int i = 0; i < std::min(beam_width, (int)candidates.size()); ++i) {
                beam.push(candidates[i]);
            }
            
            if (beam.empty()) break;
        }
        
        // Return best sequence
        if (!beam.empty()) {
            return sequenceToText(beam.top().sequence);
        }
        
        return seed;
    }
    
    // Save model to file
    void saveModel(const std::string& filename) {
        std::ofstream file(filename, std::ios::binary);
        
        // Save vocabulary size and mappings
        file.write(reinterpret_cast<const char*>(&vocab_size), sizeof(vocab_size));
        
        for (const auto& pair : word_to_id) {
            size_t word_len = pair.first.length();
            file.write(reinterpret_cast<const char*>(&word_len), sizeof(word_len));
            file.write(pair.first.c_str(), word_len);
            file.write(reinterpret_cast<const char*>(&pair.second), sizeof(pair.second));
        }
        
        // Save LSTM weights (simplified - would need full serialization)
        std::cout << "Model saved to " << filename << std::endl;
    }
    
    // Load model from file
    void loadModel(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        
        if (!file) {
            std::cerr << "Could not open model file: " << filename << std::endl;
            return;
        }
        
        // Load vocabulary
        file.read(reinterpret_cast<char*>(&vocab_size), sizeof(vocab_size));
        
        word_to_id.clear();
        id_to_word.clear();
        
        for (int i = 0; i < vocab_size; ++i) {
            size_t word_len;
            file.read(reinterpret_cast<char*>(&word_len), sizeof(word_len));
            
            std::string word(word_len, '\0');
            file.read(&word, word_len);
            
            int word_id;
            file.read(reinterpret_cast<char*>(&word_id), sizeof(word_id));
            
            word_to_id[word] = word_id;
            id_to_word[word_id] = word;
        }
        
        // Reinitialize LSTM
        lstm = std::make_unique<LSTMCell>(vocab_size, hidden_size, vocab_size);
        
        std::cout << "Model loaded from " << filename << std::endl;
    }
};
// Usage Example and Demonstration

// The following example demonstrates how to use the text prediction system for training and inference:

int main() {
    // Create text predictor
    TextPredictor predictor(128, 20); // 128 hidden units, max sequence length 20
    
    // Sample training corpus
    std::vector<std::string> corpus = {
        "the quick brown fox jumps over the lazy dog",
        "machine learning is a subset of artificial intelligence",
        "natural language processing enables computers to understand human language",
        "deep learning models can learn complex patterns from data",
        "recurrent neural networks are good for sequential data",
        "lstm networks solve the vanishing gradient problem"
    };
    
    // Build vocabulary and train
    predictor.buildVocabulary(corpus, 1);
    predictor.train(corpus, 50, 0.01f);
    
    // Test predictions
    std::cout << "
Testing predictions:" << std::endl;
    
    std::string context = "the quick brown";
    std::string next_word = predictor.predictNextWord(context);
    std::cout << "Context: '" << context << "' -> Next word: '" << next_word << "'" << std::endl;
    
    // Generate text with beam search
    std::string generated = predictor.generateText("machine learning", 10, 3);
    std::cout << "Generated text: " << generated << std::endl;
    
    // Save model
    predictor.saveModel("text_predictor.model");
    
    return 0;
}