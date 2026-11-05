#include <chrono>
#include <format>
#include <print>
#include "logger.hpp"

namespace culbm::basic
{
    void Logger::log(Level lvl, std::string msg, const std::source_location& loc)
    {
        auto getTimeRepr = []() -> std::string
        {
            using namespace std::chrono;

            auto now    = system_clock::now();
            auto t      = system_clock::to_time_t(now);
            auto ms     = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;

            std::tm local_tm;
            localtime_r(&t, &local_tm);

            return
                std::format(
                    "{:02}:{:02}:{:02}.{:03}", 
                    local_tm.tm_hour, 
                    local_tm.tm_min, 
                    local_tm.tm_sec, 
                    ms.count()
                );
        };

        auto getLogLevelRepr = [lvl]() -> std::string_view
        {
            switch (lvl)
            {
                case Level::Info:       return "\033[32mINFO\033[0m";
                case Level::Debug:      return "\033[34mDEBUG\033[0m";
                case Level::Warning:    return "\033[33mWARNING\033[0m";
                case Level::Error:      return "\033[31mERROR\033[0m";
            }
            return "UNKNOWN";
        };

        std::println(output,
            "[{}] [{}] [{}] [{}:{}] message: {}",
            getTimeRepr(),
            name,
            getLogLevelRepr(),
            loc.file_name(), 
            loc.line(), 
            msg
        );
    }
}
