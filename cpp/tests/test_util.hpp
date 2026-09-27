#pragma once

#include <exception>
#include <stdexcept>
#include <string>


// 条件不成立时让测试失败，并说明原因
inline void expect(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}


// 用于检查非法配置是否按预期抛出异常
template <typename Function>
void expect_throws(Function action, const std::string& message) {
    bool threw = false;

    try {
        action();
    } catch (const std::exception&) {
        threw = true;
    }

    expect(threw, message);
}