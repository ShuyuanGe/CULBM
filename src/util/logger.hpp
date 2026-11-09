#pragma once

#include <string>
#include <string_view>
#include <iostream>
#include <source_location>
#include <utility>

namespace culbm::util
{
    class Logger
    {
        private:
            std::string     name;
            std::ostream&   output;

            enum struct Level
            {
                info,
                debug,
                warning,
                error
            };

            void log(Level lvl, std::string_view msg, const std::source_location& loc);

        public:
            Logger(std::string name, std::ostream& output = std::cout) : name(std::move(name)), output(output) {}

            void info(std::string_view msg, std::source_location loc = std::source_location::current()) { log(Level::info, msg, loc); }
            void debug(std::string_view msg, std::source_location loc = std::source_location::current()) { log(Level::debug, msg, loc); }
            void warning(std::string_view msg, std::source_location loc = std::source_location::current()) { log(Level::warning, msg, loc); }
            void error(std::string_view msg, std::source_location loc = std::source_location::current()) { log(Level::error, msg, loc); }
    };
}
