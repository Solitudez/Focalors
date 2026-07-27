#pragma once

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

namespace CrossChannelQuarterMirror
{
    struct Dimensions
    {
        int nx2_full = 0;
        int ny2_full = 0;
        int nx2_q    = 0;
        int ny2_q    = 0;
        int nx3      = 0;
        int ny5      = 0;
    };

    struct Options
    {
        std::filesystem::path    quarter_root;
        std::filesystem::path    output_root;
        int                      quarter_step         = 0;
        double                   assignment_tolerance = 1.0e-12;
        std::vector<std::string> variables;
    };

    struct Result
    {
        std::filesystem::path    source_final_dir;
        std::filesystem::path    output_final_dir;
        int                      quarter_step = 0;
        Dimensions               dimensions;
        std::vector<std::string> reconstructed_variables;
        std::size_t              written_file_count      = 0;
        double                   max_assignment_conflict = 0.0;
    };

    std::vector<std::string> default_variables();
    int                      detect_latest_step(const std::filesystem::path& quarter_root);
    Dimensions               infer_dimensions(const std::filesystem::path& quarter_root, int quarter_step);
    Result                   reconstruct(const Options& options);
} // namespace CrossChannelQuarterMirror
