#pragma once

#include <string>
#include <cstdio>
#include <source_location>

namespace culbm::basic
{
    class Logger
    {
        private:
            std::string     name;
            std::FILE*      output;

            enum struct Level
            {
                Info,
                Debug,
                Warning,
                Error
            };

            void log(Level lvl, std::string msg, const std::source_location& loc);

        public:
            Logger(std::string name, std::FILE* output = stdout) : name(std::move(name)), output(output) {}

            inline void info(std::string msg, std::source_location loc = std::source_location::current()) { log(Level::Info, std::move(msg), loc); }
            inline void debug(std::string msg, std::source_location loc = std::source_location::current()) { log(Level::Debug, std::move(msg), loc); }
            inline void warning(std::string msg, std::source_location loc = std::source_location::current()) { log(Level::Warning, std::move(msg), loc); }
            inline void error(std::string msg, std::source_location loc = std::source_location::current()) { log(Level::Error, std::move(msg), loc); }
    };
}
