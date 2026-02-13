#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cctype>
#include <sstream>

class TextPreprocessor {
private:
    std::unordered_map<std::string, int> word_to_id;
    std::unordered_map<int, std::string> id_to_word;
    int vocab_size = 0;
    
public:
    // Convert text to lowercase
    std::string to_lowercase(const std::string& text) {
        std::string result = text;
        std::transform(result.begin(), result.end(), result.begin(), ::tolower);
        return result;
    }
    
    // Remove punctuation and special characters
    std::string remove_punctuation(const std::string& text) {
        std::string result;
        for (char c : text) {
            if (std::isalnum(c) || std::isspace(c)) {
                result += c;
            }
        }
        return result;
    }
    
    // Split text into sentences
    std::vector<std::string> split_sentences(const std::string& text) {
        std::vector<std::string> sentences;
        std::string current_sentence;
        
        for (char c : text) {
            current_sentence += c;
            if (c == '.' || c == '!' || c == '?') {
                if (!current_sentence.empty()) {
                    sentences.push_back(current_sentence);
                    current_sentence.clear();
                }
            }
        }
        
        if (!current_sentence.empty()) {
            sentences.push_back(current_sentence);
        }
        
        return sentences;
    }
    
    // Tokenize text into words
    std::vector<std::string> tokenize(const std::string& text) {
        std::vector<std::string> tokens;
        std::istringstream iss(text);
        std::string word;
        
        while (iss >> word) {
            if (!word.empty()) {
                tokens.push_back(word);
            }
        }
        return tokens;
    }
    
    // Build vocabulary from tokens
    void build_vocabulary(const std::vector<std::string>& tokens) {
        word_to_id.clear();
        id_to_word.clear();
        vocab_size = 0;
        
        // Add special tokens
        add_word("<UNK>");  // Unknown word
        add_word("<PAD>");  // Padding
        add_word("<SOS>");  // Start of sequence
        add_word("<EOS>");  // End of sequence
        
        // Add words from tokens
        for (const std::string& word : tokens) {
            if (word_to_id.find(word) == word_to_id.end()) {
                add_word(word);
            }
        }
    }
    
    // Convert tokens to integer IDs
    std::vector<int> tokens_to_ids(const std::vector<std::string>& tokens) {
        std::vector<int> ids;
        for (const std::string& token : tokens) {
            auto it = word_to_id.find(token);
            if (it != word_to_id.end()) {
                ids.push_back(it->second);
            } else {
                ids.push_back(word_to_id["<UNK>"]);  // Unknown word
            }
        }
        return ids;
    }
    
    // Convert integer IDs back to tokens
    std::vector<std::string> ids_to_tokens(const std::vector<int>& ids) {
        std::vector<std::string> tokens;
        for (int id : ids) {
            auto it = id_to_word.find(id);
            if (it != id_to_word.end()) {
                tokens.push_back(it->second);
            } else {
                tokens.push_back("<UNK>");
            }
        }
        return tokens;
    }
    
    // Pad sequences to fixed length
    std::vector<int> pad_sequence(const std::vector<int>& sequence, int max_length) {
        std::vector<int> padded = sequence;
        int pad_id = word_to_id["<PAD>"];
        
        if (padded.size() > max_length) {
            padded.resize(max_length);  // Truncate
        } else {
            while (padded.size() < max_length) {
                padded.push_back(pad_id);  // Pad
            }
        }
        return padded;
    }
    
    // Complete preprocessing pipeline
    std::vector<int> preprocess_text(const std::string& text, int max_length = 50) {
        // Step 1: Lowercase
        std::string clean_text = to_lowercase(text);
        
        // Step 2: Remove punctuation
        clean_text = remove_punctuation(clean_text);
        
        // Step 3: Tokenize
        std::vector<std::string> tokens = tokenize(clean_text);
        
        // Step 4: Convert to IDs
        std::vector<int> ids = tokens_to_ids(tokens);
        
        // Step 5: Pad sequence
        return pad_sequence(ids, max_length);
    }
    
    // Create training sequences for RNN
    std::pair<std::vector<std::vector<int>>, std::vector<std::vector<int>>> 
    create_sequences(const std::vector<int>& ids, int seq_length) {
        std::vector<std::vector<int>> inputs, targets;
        
        for (int i = 0; i <= (int)ids.size() - seq_length - 1; i++) {
            std::vector<int> input_seq(ids.begin() + i, ids.begin() + i + seq_length);
            std::vector<int> target_seq(ids.begin() + i + 1, ids.begin() + i + seq_length + 1);
            
            inputs.push_back(input_seq);
            targets.push_back(target_seq);
        }
        
        return {inputs, targets};
    }
    
    // One-hot encode sequences
    std::vector<std::vector<std::vector<int>>> one_hot_encode(const std::vector<std::vector<int>>& sequences) {
        std::vector<std::vector<std::vector<int>>> encoded;
        
        for (const auto& seq : sequences) {
            std::vector<std::vector<int>> encoded_seq;
            for (int id : seq) {
                std::vector<int> one_hot(vocab_size, 0);
                if (id < vocab_size) {
                    one_hot[id] = 1;
                }
                encoded_seq.push_back(one_hot);
            }
            encoded.push_back(encoded_seq);
        }
        
        return encoded;
    }
    
    // Getters
    int get_vocab_size() const { return vocab_size; }
    int get_word_id(const std::string& word) const {
        auto it = word_to_id.find(word);
        return (it != word_to_id.end()) ? it->second : word_to_id.at("<UNK>");
    }
    
    void print_vocabulary() const {
        std::cout << "Vocabulary (size: " << vocab_size << "):\n";
        for (const auto& pair : word_to_id) {
            std::cout << pair.first << " -> " << pair.second << "\n";
        }
    }
    
private:
    void add_word(const std::string& word) {
        word_to_id[word] = vocab_size;
        id_to_word[vocab_size] = word;
        vocab_size++;
    }
};

// Usage example
int main() {
    TextPreprocessor preprocessor;
    
    // Sample text
    std::string text = "Hello World! This is a simple text preprocessing example. "
                      "We will tokenize, build vocabulary, and create sequences.";
    
    std::cout << "Original text: " << text << "\n\n";
    
    // Step 1: Basic preprocessing
    std::string clean = preprocessor.to_lowercase(text);
    clean = preprocessor.remove_punctuation(clean);
    std::cout << "Cleaned text: " << clean << "\n\n";
    
    // Step 2: Split into sentences
    std::vector<std::string> sentences = preprocessor.split_sentences(text);
    std::cout << "Sentences:\n";
    for (size_t i = 0; i < sentences.size(); i++) {
        std::cout << i + 1 << ": " << sentences[i] << "\n";
    }
    std::cout << "\n";
    
    // Step 3: Tokenization
    std::vector<std::string> tokens = preprocessor.tokenize(clean);
    std::cout << "Tokens: ";
    for (const std::string& token : tokens) {
        std::cout << token << " ";
    }
    std::cout << "\n\n";
    
    // Step 4: Build vocabulary
    preprocessor.build_vocabulary(tokens);
    preprocessor.print_vocabulary();
    std::cout << "\n";
    
    // Step 5: Convert to IDs
    std::vector<int> ids = preprocessor.tokens_to_ids(tokens);
    std::cout << "Token IDs: ";
    for (int id : ids) {
        std::cout << id << " ";
    }
    std::cout << "\n\n";
    
    // Step 6: Create training sequences
    auto [inputs, targets] = preprocessor.create_sequences(ids, 3);
    std::cout << "Training sequences (seq_length=3):\n";
    for (size_t i = 0; i < inputs.size(); i++) {
        std::cout << "Input: ";
        for (int id : inputs[i]) std::cout << id << " ";
        std::cout << "-> Target: ";
        for (int id : targets[i]) std::cout << id << " ";
        std::cout << "\n";
    }
    
    return 0;
}