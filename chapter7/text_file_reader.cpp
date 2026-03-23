#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <functional>

class TextFileProcessor {
private:
    std::string filename;
    
public:
    TextFileProcessor(const std::string& file) : filename(file) {}
    
    // Read entire file into memory - suitable for smaller files
    std::string readEntireFile() {
        std::ifstream file(filename, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Cannot open file: " + filename);
        }
        
        // Get file size for efficient memory allocation
        file.seekg(0, std::ios::end);
        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);
        
        std::string content(size, '\0');
        file.read(&content[0], size);
        return content;
    }
    
    // Stream-based processing for large files
    void processLineByLine(std::function<void(const std::string&)> processor) {
        std::ifstream file(filename);
        std::string line;
        
        while (std::getline(file, line)) {
            processor(line);
        }
    }
    
    // Write processed text to file
    void writeToFile(const std::vector<std::string>& lines, 
                     const std::string& outputFile) {
        std::ofstream out(outputFile);
        for (const auto& line : lines) {
            out << line << '\n';
        }
    }
};

int main() {
    // Create a test file
    std::string testFile = "test_input.txt";
    {
        std::ofstream out(testFile);
        out << "Hello World" << '\n';
        out << "Deep Learning with C++" << '\n';
        out << "Testing file reader" << '\n';
    }

    TextFileProcessor processor(testFile);

    // Test readEntireFile
    std::cout << "=== Read Entire File ===" << std::endl;
    std::string content = processor.readEntireFile();
    std::cout << content << std::endl;

    // Test processLineByLine
    std::cout << "=== Process Line By Line ===" << std::endl;
    int lineNum = 0;
    processor.processLineByLine([&lineNum](const std::string& line) {
        std::cout << "Line " << ++lineNum << ": " << line << std::endl;
    });

    // Test writeToFile
    std::cout << "\n=== Write To File ===" << std::endl;
    std::vector<std::string> lines = {"Output line 1", "Output line 2", "Output line 3"};
    std::string outputFile = "test_output.txt";
    processor.writeToFile(lines, outputFile);

    // Verify written file
    TextFileProcessor reader(outputFile);
    std::cout << reader.readEntireFile() << std::endl;

    // Cleanup
    std::remove(testFile.c_str());
    std::remove(outputFile.c_str());

    std::cout << "TextFileProcessor tests completed successfully!" << std::endl;
    return 0;
}
