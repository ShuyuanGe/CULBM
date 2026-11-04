#pragma once

#include <string>
#include <iostream>
#include <source_location>

namespace culbm::basic
{
    class Logger
    {
        private:
            std::string     _name;
            std::ostream&   _os;

            enum struct Level
            {
                Info,
                Debug,
                Warning,
                Error
            };

            void _log(Level lvl, std::string msg, const std::source_location& loc);

        public:
            Logger(std::string name, std::ostream& os = std::cout) : _name(std::move(name)), _os(os) {}

            inline void info(std::string msg, std::source_location loc = std::source_location::current()) { _log(Level::Info, std::move(msg), loc); }
            inline void debug(std::string msg, std::source_location loc = std::source_location::current()) { _log(Level::Debug, std::move(msg), loc); }
            inline void warning(std::string msg, std::source_location loc = std::source_location::current()) { _log(Level::Warning, std::move(msg), loc); }
            inline void error(std::string msg, std::source_location loc = std::source_location::current()) { _log(Level::Error, std::move(msg), loc); }
    };
}
