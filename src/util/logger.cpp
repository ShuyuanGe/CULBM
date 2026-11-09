#include <chrono>
#include <format>
#include <print>
#include <syncstream>
#include "logger.hpp"

namespace culbm::util
{
    void Logger::log(Level lvl, std::string_view msg, const std::source_location& loc)
    {
        const auto time = std::chrono::zoned_time{std::chrono::current_zone(),
            std::chrono::floor<std::chrono::milliseconds>(std::chrono::system_clock::now())};

        auto get_log_level_repr = [lvl]() -> std::string_view
        {
            switch (lvl)
            {
                case Level::info:       return "\033[32mINFO\033[0m";
                case Level::debug:      return "\033[34mDEBUG\033[0m";
                case Level::warning:    return "\033[33mWARNING\033[0m";
                case Level::error:      return "\033[31mERROR\033[0m";
            }
            return "UNKNOWN";
        };

        std::osyncstream out(output);
        std::println(out,
            "[{:%H:%M:%S}] [{}] [{}] [{}:{}] message: {}",
            time,
            name,
            get_log_level_repr(),
            loc.file_name(), 
            loc.line(), 
            msg
        );
    }
}
