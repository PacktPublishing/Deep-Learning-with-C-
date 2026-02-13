#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <cmath>
#include <sstream>

class ROUGECalculator {
private:
    // Tokenize string into words
    std::vector<std::string> tokenize(const std::string& text) {
        std::vector<std::string> tokens;
        std::istringstream iss(text);
        std::string token;
        
        while (iss >> token) {
            tokens.push_back(token);
        }
        return tokens;
    }
    
    // Extract n-grams from tokens
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
    
    // Count n-grams
    std::unordered_map<std::string, int> count_ngrams(const std::vector<std::string>& tokens, int n) {
        std::unordered_map<std::string, int> counts;
        auto ngrams = extract_ngrams(tokens, n);
        
        for (const auto& ngram : ngrams) {
            counts[ngram]++;
        }
        return counts;
    }
    
    // Longest Common Subsequence length
    int lcs_length(const std::vector<std::string>& seq1, const std::vector<std::string>& seq2) {
        int m = seq1.size();
        int n = seq2.size();
        
        std::vector<std::vector<int>> dp(m + 1, std::vector<int>(n + 1, 0));
        
        for (int i = 1; i <= m; ++i) {
            for (int j = 1; j <= n; ++j) {
                if (seq1[i-1] == seq2[j-1]) {
                    dp[i][j] = dp[i-1][j-1] + 1;
                } else {
                    dp[i][j] = std::max(dp[i-1][j], dp[i][j-1]);
                }
            }
        }
        
        return dp[m][n];
    }
    
    // Extract skip-bigrams with maximum skip distance
    std::vector<std::string> extract_skip_bigrams(const std::vector<std::string>& tokens, int max_skip = 4) {
        std::vector<std::string> skip_bigrams;
        
        for (size_t i = 0; i < tokens.size(); ++i) {
            for (size_t j = i + 1; j < tokens.size() && j <= i + max_skip + 1; ++j) {
                skip_bigrams.push_back(tokens[i] + " " + tokens[j]);
            }
        }
        
        return skip_bigrams;
    }
    
    // Calculate weighted n-gram score (for ROUGE-W)
    float calculate_weighted_score(const std::vector<std::string>& candidate, 
                                 const std::vector<std::string>& reference, 
                                 float weight_factor = 1.2) {
        std::unordered_map<std::string, int> ref_counts;
        std::unordered_map<std::string, int> cand_counts;
        
        // Count unigrams
        for (const auto& token : reference) {
            ref_counts[token]++;
        }
        for (const auto& token : candidate) {
            cand_counts[token]++;
        }
        
        float weighted_matches = 0.0f;
        float total_ref_weight = 0.0f;
        
        // Calculate weighted matches
        for (const auto& [token, ref_count] : ref_counts) {
            float weight = std::pow(weight_factor, ref_count);
            total_ref_weight += weight;
            
            if (cand_counts.count(token)) {
                int matches = std::min(ref_count, cand_counts[token]);
                weighted_matches += matches * weight;
            }
        }
        
        return total_ref_weight > 0 ? weighted_matches / total_ref_weight : 0.0f;
    }
    
public:
    struct ROUGEScores {
        float precision;
        float recall;
        float f1_score;
    };
    
    // ROUGE-N: N-gram overlap
    ROUGEScores calculate_rouge_n(const std::string& candidate_text,
                                const std::string& reference_text,
                                int n = 1) {
        auto candidate = tokenize(candidate_text);
        auto reference = tokenize(reference_text);
        
        auto cand_ngrams = count_ngrams(candidate, n);
        auto ref_ngrams = count_ngrams(reference, n);
        
        int overlap = 0;
        int total_ref_ngrams = 0;
        int total_cand_ngrams = 0;
        
        // Count overlapping n-grams
        for (const auto& [ngram, ref_count] : ref_ngrams) {
            total_ref_ngrams += ref_count;
            if (cand_ngrams.count(ngram)) {
                overlap += std::min(ref_count, cand_ngrams[ngram]);
            }
        }
        
        for (const auto& [ngram, cand_count] : cand_ngrams) {
            total_cand_ngrams += cand_count;
        }
        
        float recall = total_ref_ngrams > 0 ? static_cast<float>(overlap) / total_ref_ngrams : 0.0f;
        float precision = total_cand_ngrams > 0 ? static_cast<float>(overlap) / total_cand_ngrams : 0.0f;
        float f1 = (precision + recall) > 0 ? 2 * precision * recall / (precision + recall) : 0.0f;
        
        return {precision, recall, f1};
    }
    
    // ROUGE-L: Longest Common Subsequence
    ROUGEScores calculate_rouge_l(const std::string& candidate_text,
                                const std::string& reference_text) {
        auto candidate = tokenize(candidate_text);
        auto reference = tokenize(reference_text);
        
        int lcs_len = lcs_length(candidate, reference);
        
        float recall = reference.size() > 0 ? static_cast<float>(lcs_len) / reference.size() : 0.0f;
        float precision = candidate.size() > 0 ? static_cast<float>(lcs_len) / candidate.size() : 0.0f;
        float f1 = (precision + recall) > 0 ? 2 * precision * recall / (precision + recall) : 0.0f;
        
        return {precision, recall, f1};
    }
    
    // ROUGE-W: Weighted Longest Common Subsequence
    ROUGEScores calculate_rouge_w(const std::string& candidate_text,
                                const std::string& reference_text,
                                float weight_factor = 1.2) {
        auto candidate = tokenize(candidate_text);
        auto reference = tokenize(reference_text);
        
        // Simplified weighted LCS (using weighted unigram overlap as approximation)
        float weighted_score = calculate_weighted_score(candidate, reference, weight_factor);
        
        // For ROUGE-W, we use the weighted score as both precision and recall base
        float recall = weighted_score;
        float precision = weighted_score;
        float f1 = weighted_score;
        
        return {precision, recall, f1};
    }
    
    // ROUGE-S: Skip-bigram overlap
    ROUGEScores calculate_rouge_s(const std::string& candidate_text,
                                const std::string& reference_text,
                                int max_skip = 4) {
        auto candidate = tokenize(candidate_text);
        auto reference = tokenize(reference_text);
        
        auto cand_skip_bigrams = extract_skip_bigrams(candidate, max_skip);
        auto ref_skip_bigrams = extract_skip_bigrams(reference, max_skip);
        
        // Count skip-bigrams
        std::unordered_map<std::string, int> cand_counts;
        std::unordered_map<std::string, int> ref_counts;
        
        for (const auto& bigram : cand_skip_bigrams) {
            cand_counts[bigram]++;
        }
        for (const auto& bigram : ref_skip_bigrams) {
            ref_counts[bigram]++;
        }
        
        int overlap = 0;
        for (const auto& [bigram, ref_count] : ref_counts) {
            if (cand_counts.count(bigram)) {
                overlap += std::min(ref_count, cand_counts[bigram]);
            }
        }
        
        float recall = ref_skip_bigrams.size() > 0 ? static_cast<float>(overlap) / ref_skip_bigrams.size() : 0.0f;
        float precision = cand_skip_bigrams.size() > 0 ? static_cast<float>(overlap) / cand_skip_bigrams.size() : 0.0f;
        float f1 = (precision + recall) > 0 ? 2 * precision * recall / (precision + recall) : 0.0f;
        
        return {precision, recall, f1};
    }
    
    // Calculate all ROUGE metrics
    struct AllROUGEScores {
        ROUGEScores rouge_1;
        ROUGEScores rouge_2;
        ROUGEScores rouge_l;
        ROUGEScores rouge_w;
        ROUGEScores rouge_s;
    };
    
    AllROUGEScores calculate_all_rouge(const std::string& candidate_text,
                                     const std::string& reference_text) {
        AllROUGEScores scores;
        
        scores.rouge_1 = calculate_rouge_n(candidate_text, reference_text, 1);
        scores.rouge_2 = calculate_rouge_n(candidate_text, reference_text, 2);
        scores.rouge_l = calculate_rouge_l(candidate_text, reference_text);
        scores.rouge_w = calculate_rouge_w(candidate_text, reference_text);
        scores.rouge_s = calculate_rouge_s(candidate_text, reference_text);
        
        return scores;
    }
    
    // Multi-reference ROUGE (average over multiple references)
    AllROUGEScores calculate_multi_reference_rouge(const std::string& candidate_text,
                                                 const std::vector<std::string>& reference_texts) {
        AllROUGEScores avg_scores = {{0,0,0}, {0,0,0}, {0,0,0}, {0,0,0}, {0,0,0}};
        
        for (const auto& reference : reference_texts) {
            auto scores = calculate_all_rouge(candidate_text, reference);
            
            avg_scores.rouge_1.precision += scores.rouge_1.precision;
            avg_scores.rouge_1.recall += scores.rouge_1.recall;
            avg_scores.rouge_1.f1_score += scores.rouge_1.f1_score;
            
            avg_scores.rouge_2.precision += scores.rouge_2.precision;
            avg_scores.rouge_2.recall += scores.rouge_2.recall;
            avg_scores.rouge_2.f1_score += scores.rouge_2.f1_score;
            
            avg_scores.rouge_l.precision += scores.rouge_l.precision;
            avg_scores.rouge_l.recall += scores.rouge_l.recall;
            avg_scores.rouge_l.f1_score += scores.rouge_l.f1_score;
            
            avg_scores.rouge_w.precision += scores.rouge_w.precision;
            avg_scores.rouge_w.recall += scores.rouge_w.recall;
            avg_scores.rouge_w.f1_score += scores.rouge_w.f1_score;
            
            avg_scores.rouge_s.precision += scores.rouge_s.precision;
            avg_scores.rouge_s.recall += scores.rouge_s.recall;
            avg_scores.rouge_s.f1_score += scores.rouge_s.f1_score;
        }
        
        int num_refs = reference_texts.size();
        if (num_refs > 0) {
            avg_scores.rouge_1.precision /= num_refs;
            avg_scores.rouge_1.recall /= num_refs;
            avg_scores.rouge_1.f1_score /= num_refs;
            
            avg_scores.rouge_2.precision /= num_refs;
            avg_scores.rouge_2.recall /= num_refs;
            avg_scores.rouge_2.f1_score /= num_refs;
            
            avg_scores.rouge_l.precision /= num_refs;
            avg_scores.rouge_l.recall /= num_refs;
            avg_scores.rouge_l.f1_score /= num_refs;
            
            avg_scores.rouge_w.precision /= num_refs;
            avg_scores.rouge_w.recall /= num_refs;
            avg_scores.rouge_w.f1_score /= num_refs;
            
            avg_scores.rouge_s.precision /= num_refs;
            avg_scores.rouge_s.recall /= num_refs;
            avg_scores.rouge_s.f1_score /= num_refs;
        }
        
        return avg_scores;
    }
};

void print_rouge_scores(const std::string& metric_name, const ROUGECalculator::ROUGEScores& scores) {
    std::cout << metric_name << ":" << std::endl;
    std::cout << "  Precision: " << scores.precision << std::endl;
    std::cout << "  Recall: " << scores.recall << std::endl;
    std::cout << "  F1-Score: " << scores.f1_score << std::endl;
}

int main() {
    ROUGECalculator rouge_calc;
    
    std::cout << "=== ROUGE Score Calculator ===" << std::endl;
    
    // Test data
    std::string candidate = "the cat sat on the mat and looked around";
    std::string reference1 = "a cat was sitting on the mat";
    std::string reference2 = "the cat sat on a mat and was looking around";
    std::string reference3 = "the cat was on the mat";
    
    std::cout << "\nCandidate: \"" << candidate << "\"" << std::endl;
    std::cout << "Reference 1: \"" << reference1 << "\"" << std::endl;
    std::cout << "Reference 2: \"" << reference2 << "\"" << std::endl;
    
    // Single reference ROUGE scores
    std::cout << "\n=== Single Reference ROUGE Scores ===" << std::endl;
    auto all_scores = rouge_calc.calculate_all_rouge(candidate, reference1);
    
    print_rouge_scores("ROUGE-1", all_scores.rouge_1);
    print_rouge_scores("ROUGE-2", all_scores.rouge_2);
    print_rouge_scores("ROUGE-L", all_scores.rouge_l);
    print_rouge_scores("ROUGE-W", all_scores.rouge_w);
    print_rouge_scores("ROUGE-S", all_scores.rouge_s);
    
    // Multi-reference ROUGE scores
    std::cout << "\n=== Multi-Reference ROUGE Scores ===" << std::endl;
    std::vector<std::string> references = {reference1, reference2, reference3};
    auto multi_scores = rouge_calc.calculate_multi_reference_rouge(candidate, references);
    
    print_rouge_scores("ROUGE-1 (Multi-ref)", multi_scores.rouge_1);
    print_rouge_scores("ROUGE-2 (Multi-ref)", multi_scores.rouge_2);
    print_rouge_scores("ROUGE-L (Multi-ref)", multi_scores.rouge_l);
    print_rouge_scores("ROUGE-W (Multi-ref)", multi_scores.rouge_w);
    print_rouge_scores("ROUGE-S (Multi-ref)", multi_scores.rouge_s);
    
    // Individual metric demonstrations
    std::cout << "\n=== Individual Metric Analysis ===" << std::endl;
    
    // ROUGE-N with different n values
    std::cout << "\nROUGE-N Analysis:" << std::endl;
    for (int n = 1; n <= 4; ++n) {
        auto rouge_n = rouge_calc.calculate_rouge_n(candidate, reference1, n);
        std::cout << "ROUGE-" << n << " F1: " << rouge_n.f1_score << std::endl;
    }
    
    // ROUGE-S with different skip distances
    std::cout << "\nROUGE-S Analysis (different skip distances):" << std::endl;
    for (int skip = 2; skip <= 6; skip += 2) {
        auto rouge_s = rouge_calc.calculate_rouge_s(candidate, reference1, skip);
        std::cout << "ROUGE-S (skip=" << skip << ") F1: " << rouge_s.f1_score << std::endl;
    }
    
    // Edge cases
    std::cout << "\n=== Edge Cases ===" << std::endl;
    
    // Perfect match
    auto perfect_scores = rouge_calc.calculate_all_rouge(reference1, reference1);
    std::cout << "Perfect match ROUGE-1 F1: " << perfect_scores.rouge_1.f1_score << std::endl;
    std::cout << "Perfect match ROUGE-L F1: " << perfect_scores.rouge_l.f1_score << std::endl;
    
    // No overlap
    std::string no_overlap_candidate = "hello world";
    std::string no_overlap_reference = "goodbye universe";
    auto no_overlap_scores = rouge_calc.calculate_all_rouge(no_overlap_candidate, no_overlap_reference);
    std::cout << "No overlap ROUGE-1 F1: " << no_overlap_scores.rouge_1.f1_score << std::endl;
    
    std::cout << "\n=== ROUGE Metrics Explanation ===" << std::endl;
    std::cout << "ROUGE-N: N-gram overlap between candidate and reference" << std::endl;
    std::cout << "ROUGE-L: Longest Common Subsequence based similarity" << std::endl;
    std::cout << "ROUGE-W: Weighted LCS favoring consecutive matches" << std::endl;
    std::cout << "ROUGE-S: Skip-bigram overlap allowing gaps" << std::endl;
    std::cout << "\nHigher scores indicate better summarization quality" << std::endl;
    std::cout << "F1-score balances precision and recall" << std::endl;
    
    return 0;
}