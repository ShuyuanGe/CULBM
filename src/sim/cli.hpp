#pragma once

#include <array>
#include <vector_types.h>
#include "CLI11.hpp"
#include "exit.hpp"

namespace culbm::sim::cli
{
    using DimValues = std::array<decltype(dim3::x), 3>;

    inline CLI::Option* addDimOption(CLI::App& app, const char* name, dim3& dim,
                                     DimValues defaultValue, const char* description)
    {
        return app.add_option_function<DimValues>(name, [&dim](const auto& values) {
            dim = {values[0], values[1], values[2]};
        }, description)->check(CLI::PositiveNumber)->run_callback_for_default()->default_val(defaultValue);
    }

    inline void parse(CLI::App& app, int argc, char** argv) try
    {
        app.parse(argc, argv);
    }
    catch (const CLI::ParseError& e)
    {
        throw Exit{app.exit(e)};
    }
}
