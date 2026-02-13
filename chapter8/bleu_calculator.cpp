#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <sstream>

class BLEUCalculator {
private:
    // Extract n-grams from a sentence
    std::vector<std::string> extract_ngrams(const std::vector<std::string>& tokens, int n) {
        std::vector<std::string> ngrams;
        if (tokens.size() < n) return ngrams;
        
        for (size_t i = 0; i <= tokens.size() - n; ++i) {
            std::string ngram = "";
            for (int j = 0; j < n; ++j) {
                if (j > 0) ngram += " ";
                ngram += tokens[i + j];
            }
            ngrams.push_back(ngram);
        }
        return ngrams;
    }
    
    // Count n-grams in a sentence
    std::unordered_map<std::string, int> count_ngrams(const std::vector<std::string>& tokens, int n) {
        std::unordered_map<std::string, int> counts;
        auto ngrams = extract_ngrams(tokens, n);
        
        for (const auto& ngram : ngrams) {
            counts[ngram]++;
        }
        return counts;
    }
    
    // Calculate modified precision for n-grams
    float calculate_modified_precision(const std::vector<std::string>& candidate,
                                     const std::vector<std::vector<std::string>>& references,
                                     int n) {
        auto candidate_counts = count_ngrams(candidate, n);
        
        // Count maximum occurrences in any reference
        std::unordered_map<std::string, int> max_ref_counts;
        for (const auto& reference : references) {
            auto ref_counts = count_ngrams(reference, n);
            for (const auto& [ngram, count] : ref_counts) {
                max_ref_counts[ngram] = std::max(max_ref_counts[ngram], count);
            }
        }
        
        // Calculate clipped counts
        int clipped_count = 0;
        int total_count = 0;
        
        for (const auto& [ngram, count] : candidate_counts) {
            clipped_count += std::min(count, max_ref_counts[ngram]);
            total_count += count;
        }
        
        return total_count > 0 ? static_cast<float>(clipped_count) / total_count : 0.0f;
    }
    
    // Calculate brevity penalty
    float calculate_brevity_penalty(int candidate_length, const std::vector<std::vector<std::string>>& references) {
        // Find closest reference length
        int closest_ref_length = references[0].size();
        int min_diff = std::abs(candidate_length - closest_ref_length);
        
        for (const auto& reference : references) {
            int diff = std::abs(candidate_length - static_cast<int>(reference.size()));
            if (diff < min_diff) {
                min_diff = diff;
                closest_ref_length = reference.size();
            }
        }
        
        if (candidate_length > closest_ref_length) {
            return 1.0f;
        } else {
            return std::exp(1.0f - static_cast<float>(closest_ref_length) / candidate_length);
        }
    }
    
public:
    // Tokenize string into words
    std::vector<std::string> tokenize(const std::string& text) {
        std::vector<std::string> tokens;
        std::istringstream iss(text);
        std::string token;
        
        while (iss >> token) {
            // Simple tokenization - split by whitespace
            tokens.push_back(token);
        }
        return tokens;
    }
    
    // Calculate BLEU score for single sentence
    float calculate_sentence_bleu(const std::string& candidate_text,
                                const std::vector<std::string>& reference_texts,
                                int max_n = 4) {
        auto candidate = tokenize(candidate_text);
        std::vector<std::vector<std::string>> references;
        
        for (const auto& ref_text : reference_texts) {
            references.push_back(tokenize(ref_text));
        }
        
        return calculate_sentence_bleu(candidate, references, max_n);
    }
    
    // Calculate BLEU score for tokenized sentence
    float calculate_sentence_bleu(const std::vector<std::string>& candidate,
                                const std::vector<std::vector<std::string>>& references,
                                int max_n = 4) {
        if (candidate.empty() || references.empty()) return 0.0f;
        
        // Calculate modified precision for each n-gram order
        std::vector<float> precisions;
        for (int n = 1; n <= max_n; ++n) {
            float precision = calculate_modified_precision(candidate, references, n);
            precisions.push_back(precision);
        }
        
        // Calculate geometric mean of precisions
        float log_sum = 0.0f;
        for (float precision : precisions) {
            if (precision > 0) {
                log_sum += std::log(precision);
            } else {
                return 0.0f; // If any precision is 0, BLEU is 0
            }
        }
        float geometric_mean = std::exp(log_sum / max_n);
        
        // Calculate brevity penalty
        float bp = calculate_brevity_penalty(candidate.size(), references);
        
        return bp * geometric_mean;
    }
    
    // Calculate corpus-level BLEU score
    struct CorpusBLEU {
        float bleu_score;
        std::vector<float> precisions;
        float brevity_penalty;
        int candidate_length;
        int reference_length;
    };
    
    CorpusBLEU calculate_corpus_bleu(const std::vector<std::string>& candidate_texts,
                                   const std::vector<std::vector<std::string>>& reference_texts_list,
                                   int max_n = 4) {
        std::vector<int> total_clipped_counts(max_n, 0);
        std::vector<int> total_counts(max_n, 0);
        int total_candidate_length = 0;
        int total_reference_length = 0;
        
        for (size_t i = 0; i < candidate_texts.size(); ++i) {
            auto candidate = tokenize(candidate_texts[i]);
            const auto& references = reference_texts_list[i];
            
            total_candidate_length += candidate.size();
            
            // Find closest reference length for this sentence
            int closest_ref_length = references[0].size();
            int min_diff = std::abs(static_cast<int>(candidate.size()) - closest_ref_length);
            
            for (const auto& reference : references) {
                int diff = std::abs(static_cast<int>(candidate.size()) - static_cast<int>(reference.size()));
                if (diff < min_diff) {
                    min_diff = diff;
                    closest_ref_length = reference.size();
                }
            }
            total_reference_length += closest_ref_length;
            
            // Calculate n-gram counts for this sentence
            for (int n = 1; n <= max_n; ++n) {
                auto candidate_counts = count_ngrams(candidate, n);
                
                std::unordered_map<std::string, int> max_ref_counts;
                for (const auto& reference : references) {
                    auto ref_counts = count_ngrams(reference, n);
                    for (const auto& [ngram, count] : ref_counts) {
                        max_ref_counts[ngram] = std::max(max_ref_counts[ngram], count);
                    }
                }
                
                for (const auto& [ngram, count] : candidate_counts) {
                    total_clipped_counts[n-1] += std::min(count, max_ref_counts[ngram]);
                    total_counts[n-1] += count;
                }
            }
        }
        
        // Calculate corpus-level precisions
        std::vector<float> precisions;
        for (int n = 0; n < max_n; ++n) {
            float precision = total_counts[n] > 0 ? 
                static_cast<float>(total_clipped_counts[n]) / total_counts[n] : 0.0f;
            precisions.push_back(precision);
        }
        
        // Calculate geometric mean
        float log_sum = 0.0f;
        for (float precision : precisions) {
            if (precision > 0) {
                log_sum += std::log(precision);
            } else {
                return {0.0f, precisions, 0.0f, total_candidate_length, total_reference_length};
            }
        }
        float geometric_mean = std::exp(log_sum / max_n);
        
        // Calculate brevity penalty
        float bp = total_candidate_length > total_reference_length ? 1.0f :
                   std::exp(1.0f - static_cast<float>(total_reference_length) / total_candidate_length);
        
        float bleu_score = bp * geometric_mean;
        
        return {bleu_score, precisions, bp, total_candidate_length, total_reference_length};
    }
    
    // Calculate individual n-gram precisions
    std::vector<float> calculate_ngram_precisions(const std::string& candidate_text,
                                                const std::vector<std::string>& reference_texts,
                                                int max_n = 4) {
        auto candidate = tokenize(candidate_text);
        std::vector<std::vector<std::string>> references;
        
        for (const auto& ref_text : reference_texts) {
            references.push_back(tokenize(ref_text));
        }
        
        std::vector<float> precisions;
        for (int n = 1; n <= max_n; ++n) {
            float precision = calculate_modified_precision(candidate, references, n);
            precisions.push_back(precision);
        }
        
        return precisions;
    }
};

int main() {
    BLEUCalculator bleu_calc;
    
    std::cout << "=== BLEU Score Calculator ===" << std::endl;
    
    // Test data
    std::string candidate1 = "the cat is on the mat";
    std::vector<std::string> references1 = {
        "the cat is sitting on the mat",
        "a cat is on the mat",
        "the cat sits on a mat"
    };
    
    std::string candidate2 = "it is a good day";
    std::vector<std::string> references2 = {
        "it is a beautiful day",
        "today is a good day",
        "it is a nice day"
    };
    
    std::cout << "\n=== Single Sentence BLEU ===" << std::endl;
    
    // Calculate sentence-level BLEU
    float bleu1 = bleu_calc.calculate_sentence_bleu(candidate1, references1);
    float bleu2 = bleu_calc.calculate_sentence_bleu(candidate2, references2);
    
    std::cout << "Candidate 1: \"" << candidate1 << "\"" << std::endl;
    std::cout << "BLEU Score: " << bleu1 << std::endl;
    
    std::cout << "\nCandidate 2: \"" << candidate2 << "\"" << std::endl;
    std::cout << "BLEU Score: " << bleu2 << std::endl;
    
    // N-gram precision analysis
    std::cout << "\n=== N-gram Precision Analysis ===" << std::endl;
    auto precisions1 = bleu_calc.calculate_ngram_precisions(candidate1, references1);
    auto precisions2 = bleu_calc.calculate_ngram_precisions(candidate2, references2);
    
    std::cout << "Candidate 1 precisions:" << std::endl;
    for (size_t i = 0; i < precisions1.size(); ++i) {
        std::cout << "  " << (i+1) << "-gram: " << precisions1[i] << std::endl;
    }
    
    std::cout << "Candidate 2 precisions:" << std::endl;
    for (size_t i = 0; i < precisions2.size(); ++i) {
        std::cout << "  " << (i+1) << "-gram: " << precisions2[i] << std::endl;
    }
    
    // Corpus-level BLEU
    std::cout << "\n=== Corpus-level BLEU ===" << std::endl;
    std::vector<std::string> candidates = {candidate1, candidate2};
    std::vector<std::vector<std::string>> all_references = {references1, references2};
    
    auto corpus_result = bleu_calc.calculate_corpus_bleu(candidates, all_references);
    
    std::cout << "Corpus BLEU Score: " << corpus_result.bleu_score << std::endl;
    std::cout << "Brevity Penalty: " << corpus_result.brevity_penalty << std::endl;
    std::cout << "Candidate Length: " << corpus_result.candidate_length << std::endl;
    std::cout << "Reference Length: " << corpus_result.reference_length << std::endl;
    
    std::cout << "Corpus N-gram Precisions:" << std::endl;
    for (size_t i = 0; i < corpus_result.precisions.size(); ++i) {
        std::cout << "  " << (i+1) << "-gram: " << corpus_result.precisions[i] << std::endl;
    }
    
    // Edge cases
    std::cout << "\n=== Edge Cases ===" << std::endl;
    
    // Perfect match
    std::string perfect_candidate = "the cat is on the mat";
    std::vector<std::string> perfect_references = {"the cat is on the mat"};
    float perfect_bleu = bleu_calc.calculate_sentence_bleu(perfect_candidate, perfect_references);
    std::cout << "Perfect match BLEU: " << perfect_bleu << std::endl;
    
    // No match
    std::string no_match_candidate = "hello world";
    std::vector<std::string> no_match_references = {"goodbye universe"};
    float no_match_bleu = bleu_calc.calculate_sentence_bleu(no_match_candidate, no_match_references);
    std::cout << "No match BLEU: " << no_match_bleu << std::endl;
    
    // Short candidate (brevity penalty test)
    std::string short_candidate = "cat";
    std::vector<std::string> long_references = {"the cat is sitting on the mat"};
    float short_bleu = bleu_calc.calculate_sentence_bleu(short_candidate, long_references);
    std::cout << "Short candidate BLEU: " << short_bleu << std::endl;
    
    std::cout << "\n=== BLEU Score Interpretation ===" << std::endl;
    std::cout << "BLEU Score Range: 0.0 to 1.0" << std::endl;
    std::cout << "Higher scores indicate better translation quality" << std::endl;
    std::cout << "Typical thresholds:" << std::endl;
    std::cout << "  > 0.4: Good translation" << std::endl;
    std::cout << "  > 0.3: Understandable translation" << std::endl;
    std::cout << "  > 0.2: Hard to understand" << std::endl;
    std::cout << "  < 0.2: Poor translation" << std::endl;
    
    return 0;
}