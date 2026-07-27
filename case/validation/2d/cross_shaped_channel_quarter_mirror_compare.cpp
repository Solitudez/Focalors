#include "io/common.h"
#include "io/para_reader.h"
#include "ns/cross_channel_quarter_mirror.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    namespace fs = std::filesystem;
    using Matrix = std::vector<std::vector<double>>;

    constexpr double NaN = std::numeric_limits<double>::quiet_NaN();

    struct CompareStats
    {
        std::size_t count                = 0;
        std::size_t skipped              = 0;
        double      max_abs              = 0.0;
        double      squared_error        = 0.0;
        double      reference_squared    = 0.0;
        int         max_i                = -1;
        int         max_j                = -1;
        double      reconstructed_at_max = NaN;
        double      full_at_max          = NaN;

        double l2_abs() const { return std::sqrt(squared_error); }
        double relative_l2() const { return std::sqrt(squared_error / std::max(reference_squared, 1.0e-30)); }
    };

    std::vector<std::string> split_csv_line(const std::string& line)
    {
        std::vector<std::string> values;
        std::stringstream        stream(line);
        std::string              value;
        while (std::getline(stream, value, ','))
            values.push_back(value);
        return values;
    }

    fs::path resolve_final_dir(const fs::path& root)
    {
        if (fs::is_directory(root / "final"))
            return root / "final";
        if (root.filename() == "final" && fs::is_directory(root))
            return root;
        throw std::runtime_error("Comparison source must contain final/ or be a final/ directory: " + root.string());
    }

    fs::path field_path(const fs::path& final_dir, const std::string& variable, int step, const std::string& domain)
    {
        return final_dir / (variable + "_" + std::to_string(step) + "_" + domain + ".csv");
    }

    Matrix read_matrix(const fs::path& path)
    {
        std::ifstream input(path);
        if (!input.is_open())
            throw std::runtime_error("Failed to read comparison field: " + path.string());

        Matrix      matrix;
        std::string line;
        while (std::getline(input, line))
        {
            const auto          tokens = split_csv_line(line);
            std::vector<double> row;
            row.reserve(tokens.size());
            for (const auto& token : tokens)
                row.push_back(std::stod(token));
            if (row.empty())
                throw std::runtime_error("Empty CSV row in " + path.string());
            if (!matrix.empty() && row.size() != matrix.front().size())
                throw std::runtime_error("Non-rectangular CSV matrix: " + path.string());
            matrix.push_back(std::move(row));
        }
        if (matrix.empty())
            throw std::runtime_error("Empty CSV matrix: " + path.string());
        return matrix;
    }

    void write_matrix(const fs::path& path, const Matrix& matrix)
    {
        std::ofstream output(path);
        if (!output.is_open())
            throw std::runtime_error("Failed to write difference field: " + path.string());
        output << std::setprecision(16);
        for (const auto& row : matrix)
        {
            for (std::size_t j = 0; j < row.size(); ++j)
            {
                if (j != 0)
                    output << ',';
                output << row[j];
            }
            output << '\n';
        }
    }

    void compare_matrix(const Matrix& reconstructed, const Matrix& full, Matrix& difference, CompareStats& stats)
    {
        if (reconstructed.size() != full.size() || reconstructed.front().size() != full.front().size())
            throw std::runtime_error("Reconstructed/full matrix shape mismatch.");

        difference = Matrix(full.size(), std::vector<double>(full.front().size(), NaN));
        for (std::size_t i = 0; i < full.size(); ++i)
        {
            for (std::size_t j = 0; j < full[i].size(); ++j)
            {
                const double value = reconstructed[i][j];
                if (!std::isfinite(value))
                {
                    ++stats.skipped;
                    continue;
                }
                const double error = value - full[i][j];
                difference[i][j]   = error;
                if (std::abs(error) > stats.max_abs)
                {
                    stats.max_abs              = std::abs(error);
                    stats.max_i                = static_cast<int>(i);
                    stats.max_j                = static_cast<int>(j);
                    stats.reconstructed_at_max = value;
                    stats.full_at_max          = full[i][j];
                }
                stats.squared_error += error * error;
                stats.reference_squared += full[i][j] * full[i][j];
                ++stats.count;
            }
        }
    }

    double read_named_last_value(const fs::path& path, const std::string& column_name)
    {
        std::ifstream input(path);
        if (!input.is_open())
            return NaN;

        std::string header_line;
        if (!std::getline(input, header_line))
            return NaN;
        const auto header = split_csv_line(header_line);
        const auto it     = std::find(header.begin(), header.end(), column_name);
        if (it == header.end())
            return NaN;
        const std::size_t index = static_cast<std::size_t>(std::distance(header.begin(), it));

        std::string              line;
        std::vector<std::string> last;
        while (std::getline(input, line))
            if (!line.empty())
                last = split_csv_line(line);
        return index < last.size() ? std::stod(last[index]) : NaN;
    }
} // namespace

int main(int argc, char* argv[])
{
    const auto para_map = IO::paras_to_map(argc, argv);

    std::string quarter_root;
    std::string full_root;
    std::string reconstructed_residual_root;
    std::string full_residual_root;
    std::string root_dir             = "result/cross_shaped_channel_quarter_mirror_compare";
    int         mirror_only          = 0;
    int         quarter_step         = 0;
    int         full_step            = 0;
    double      assignment_tolerance = 1.0e-12;

    if (!IO::read_string(para_map, "quarter_root", quarter_root))
        throw std::runtime_error("quarter_root is required.");
    IO::read_number(para_map, "mirror_only", mirror_only);
    IO::read_string(para_map, "full_root", full_root);
    IO::read_string(para_map, "root_dir", root_dir);
    IO::read_string(para_map, "reconstructed_residual_root", reconstructed_residual_root);
    IO::read_string(para_map, "full_residual_root", full_residual_root);
    IO::read_number(para_map, "quarter_step", quarter_step);
    IO::read_number(para_map, "full_step", full_step);
    IO::read_number(para_map, "assignment_tolerance", assignment_tolerance);
    if (mirror_only != 0 && mirror_only != 1)
        throw std::runtime_error("mirror_only must be 0 or 1.");
    if (mirror_only == 0 && full_root.empty())
        throw std::runtime_error("full_root is required unless mirror_only=1.");

    CrossChannelQuarterMirror::Options mirror_options;
    mirror_options.quarter_root         = quarter_root;
    mirror_options.output_root          = fs::path(root_dir) / "reconstructed_full";
    mirror_options.quarter_step         = quarter_step;
    mirror_options.assignment_tolerance = assignment_tolerance;
    const auto mirror_result            = CrossChannelQuarterMirror::reconstruct(mirror_options);

    if (mirror_only != 0)
    {
        std::cout << "Quarter-domain mirror reconstruction completed.\n"
                  << "  quarter_root: " << quarter_root << " (step " << mirror_result.quarter_step << ")\n"
                  << "  reconstructed: " << mirror_result.output_final_dir << std::endl;
        return 0;
    }

    if (full_step <= 0)
        full_step = CrossChannelQuarterMirror::detect_latest_step(full_root);
    const fs::path full_final_dir = resolve_final_dir(full_root);
    const fs::path difference_dir = fs::path(root_dir) / "difference";
    IO::create_directory(difference_dir);

    std::ofstream comparison(fs::path(root_dir) / "field_comparison.csv");
    if (!comparison.is_open())
        throw std::runtime_error("Failed to write field_comparison.csv.");
    comparison << std::setprecision(16)
               << "variable,domain,compared_count,skipped_count,max_abs,l2_abs,relative_l2,assignment_conflict,"
                  "max_i,max_j,reconstructed_at_max,full_at_max\n";

    const std::string domains[5] = {"A1", "A2", "A3", "A4", "A5"};
    for (const auto& variable : mirror_result.reconstructed_variables)
    {
        if (!fs::exists(field_path(full_final_dir, variable, full_step, "A2")))
            continue;
        for (const auto& domain : domains)
        {
            const Matrix reconstructed =
                read_matrix(field_path(mirror_result.output_final_dir, variable, mirror_result.quarter_step, domain));
            const Matrix full = read_matrix(field_path(full_final_dir, variable, full_step, domain));
            Matrix       difference;
            CompareStats stats;
            compare_matrix(reconstructed, full, difference, stats);
            write_matrix(difference_dir / (variable + "_quarter_minus_full_" + domain + ".csv"), difference);
            comparison << variable << ',' << domain << ',' << stats.count << ',' << stats.skipped << ','
                       << stats.max_abs << ',' << stats.l2_abs() << ',' << stats.relative_l2() << ','
                       << mirror_result.max_assignment_conflict << ',' << stats.max_i << ',' << stats.max_j << ','
                       << stats.reconstructed_at_max << ',' << stats.full_at_max << '\n';
        }
    }

    std::ofstream residuals(fs::path(root_dir) / "residual_comparison.csv");
    if (!residuals.is_open())
        throw std::runtime_error("Failed to write residual_comparison.csv.");
    residuals << std::setprecision(16) << "source,operator_geometry,step,max_residual\n";
    residuals << "quarter,quarter," << mirror_result.quarter_step << ','
              << read_named_last_value(fs::path(quarter_root) / "quarter_symmetry.csv", "steady_residual") << '\n';
    residuals << "full,full," << full_step << ','
              << read_named_last_value(fs::path(full_root) / "steady_history.csv", "max_residual") << '\n';
    if (!reconstructed_residual_root.empty())
        residuals << "reconstructed_full_one_step,full,1,"
                  << read_named_last_value(fs::path(reconstructed_residual_root) / "steady_history.csv", "max_residual")
                  << '\n';
    if (!full_residual_root.empty())
        residuals << "original_full_one_step,full,1,"
                  << read_named_last_value(fs::path(full_residual_root) / "steady_history.csv", "max_residual") << '\n';

    std::cout << "Quarter/full mirror comparison completed.\n"
              << "  quarter_root: " << quarter_root << " (step " << mirror_result.quarter_step << ")\n"
              << "  reconstructed: " << mirror_result.output_final_dir << "\n"
              << "  full_root: " << full_root << " (step " << full_step << ")\n"
              << "  field report: " << fs::path(root_dir) / "field_comparison.csv" << "\n"
              << "  residual report: " << fs::path(root_dir) / "residual_comparison.csv" << std::endl;
    return 0;
}
