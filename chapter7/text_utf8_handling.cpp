#include <codecvt>
#include <locale>

class UTF8Handler {
public:
    // Convert UTF-8 to wide string for processing
    static std::wstring utf8ToWide(const std::string& utf8_str) {
        std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
        return converter.from_bytes(utf8_str);
    }
    
    // Convert wide string back to UTF-8
    static std::string wideToUtf8(const std::wstring& wide_str) {
        std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
        return converter.to_bytes(wide_str);
    }
    
    // Count actual characters (not bytes) in UTF-8 string
    static size_t utf8Length(const std::string& utf8_str) {
        size_t length = 0;
        for (size_t i = 0; i < utf8_str.length(); ) {
            unsigned char c = utf8_str[i];
            if (c < 0x80) i += 1;
            else if (c < 0xE0) i += 2;
            else if (c < 0xF0) i += 3;
            else i += 4;
            length++;
        }
        return length;
    }
};