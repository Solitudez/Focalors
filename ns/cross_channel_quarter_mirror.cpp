#include "cross_channel_quarter_mirror.h"

#include "io/common.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace CrossChannelQuarterMirror
{
    namespace
    {
        namespace fs = std::filesystem;
        using Matrix = std::vector<std::vector<double>>;

        constexpr double NaN = std::numeric_limits<double>::quiet_NaN();

        enum class FieldKind
        {
            U,
            V,
            Center,
            Corner,
        };

        struct Shape
        {
            int nx = 0;
            int ny = 0;
        };

        struct AssignmentStats
        {
            std::size_t skipped_count = 0;
            double      max_conflict  = 0.0;
        };

        fs::path resolve_final_dir(const fs::path& root)
        {
            if (fs::is_directory(root / "final"))
                return root / "final";
            if (root.filename() == "final" && fs::is_directory(root))
                return root;
            throw std::runtime_error("Quarter mirror source must contain final/ or be a final/ directory: " +
                                     root.string());
        }

        fs::path field_path(const fs::path& final_dir, const std::string& variable, int step, const std::string& domain)
        {
            return final_dir / (variable + "_" + std::to_string(step) + "_" + domain + ".csv");
        }

        Matrix read_matrix(const fs::path& path)
        {
            std::ifstream input(path);
            if (!input.is_open())
                throw std::runtime_error("Failed to read quarter field: " + path.string());

            Matrix      matrix;
            std::string line;
            while (std::getline(input, line))
            {
                std::stringstream   stream(line);
                std::string         token;
                std::vector<double> row;
                while (std::getline(stream, token, ','))
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

        void require_shape(const Matrix& matrix, const Shape& shape, const fs::path& path)
        {
            if (static_cast<int>(matrix.size()) != shape.nx || static_cast<int>(matrix.front().size()) != shape.ny)
            {
                std::ostringstream message;
                message << "Unexpected shape for " << path << ": got " << matrix.size() << "x" << matrix.front().size()
                        << ", expected " << shape.nx << "x" << shape.ny;
                throw std::runtime_error(message.str());
            }
        }

        void write_matrix(const fs::path& path, const Matrix& matrix)
        {
            std::ofstream output(path);
            if (!output.is_open())
                throw std::runtime_error("Failed to write reconstructed field: " + path.string());
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

        Matrix make_matrix(const Shape& shape)
        {
            return Matrix(static_cast<std::size_t>(shape.nx),
                          std::vector<double>(static_cast<std::size_t>(shape.ny), NaN));
        }

        FieldKind field_kind(const std::string& variable)
        {
            if (variable == "u")
                return FieldKind::U;
            if (variable == "v")
                return FieldKind::V;
            if (variable == "mu" || variable == "tau_xy")
                return FieldKind::Corner;
            return FieldKind::Center;
        }

        double x_parity(const std::string& variable)
        {
            return variable == "u" || variable == "vorticity" || variable == "tau_xy" ? -1.0 : 1.0;
        }

        double y_parity(const std::string& variable)
        {
            return variable == "v" || variable == "vorticity" || variable == "tau_xy" ? -1.0 : 1.0;
        }

        Shape full_shape(FieldKind kind, int domain, const Dimensions& dimensions)
        {
            const int base_nx[5] = {
                dimensions.nx3, dimensions.nx2_full, dimensions.nx3, dimensions.nx2_full, dimensions.nx2_full};
            const int base_ny[5] = {
                dimensions.ny2_full, dimensions.ny2_full, dimensions.ny2_full, dimensions.ny5, dimensions.ny5};
            Shape shape {base_nx[domain], base_ny[domain]};
            if (kind == FieldKind::Corner)
            {
                ++shape.nx;
                ++shape.ny;
            }
            else if (kind == FieldKind::U && (domain == 2 || domain == 3 || domain == 4))
            {
                ++shape.nx;
            }
            else if (kind == FieldKind::V && (domain == 0 || domain == 2 || domain == 4))
            {
                ++shape.ny;
            }
            return shape;
        }

        Shape quarter_shape(FieldKind kind, int domain, const Dimensions& dimensions)
        {
            Shape shape;
            if (domain == 1)
                shape = {dimensions.nx2_q, dimensions.ny2_q};
            else if (domain == 2)
                shape = {dimensions.nx3, dimensions.ny2_q};
            else if (domain == 4)
                shape = {dimensions.nx2_q, dimensions.ny5};
            else
                throw std::runtime_error("Quarter geometry only contains A2, A3 and A5.");

            if (kind == FieldKind::Corner)
            {
                ++shape.nx;
                ++shape.ny;
            }
            else if (kind == FieldKind::U && (domain == 2 || domain == 4))
            {
                ++shape.nx;
            }
            else if (kind == FieldKind::V && (domain == 2 || domain == 4))
            {
                ++shape.ny;
            }
            return shape;
        }

        Matrix load_quarter_field(const fs::path&    final_dir,
                                  const std::string& variable,
                                  int                step,
                                  const std::string& domain,
                                  const Shape&       shape)
        {
            const fs::path path   = field_path(final_dir, variable, step, domain);
            Matrix         matrix = read_matrix(path);
            require_shape(matrix, shape, path);
            return matrix;
        }

        void assign(Matrix& target, int i, int j, double value, double sign, AssignmentStats& stats)
        {
            if (i < 0 || i >= static_cast<int>(target.size()) || j < 0 || j >= static_cast<int>(target.front().size()))
            {
                ++stats.skipped_count;
                return;
            }

            const double signed_value = sign * value;
            double&      destination  = target[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
            if (std::isfinite(destination))
            {
                stats.max_conflict = std::max(stats.max_conflict, std::abs(destination - signed_value));
                return;
            }
            destination = signed_value;
        }

        std::vector<Matrix> reconstruct_variable(const fs::path&    final_dir,
                                                 const std::string& variable,
                                                 int                step,
                                                 const Dimensions&  dimensions,
                                                 AssignmentStats&   stats)
        {
            const FieldKind kind = field_kind(variable);
            const double    sx   = x_parity(variable);
            const double    sy   = y_parity(variable);

            std::vector<Matrix> result;
            result.reserve(5);
            for (int domain = 0; domain < 5; ++domain)
                result.push_back(make_matrix(full_shape(kind, domain, dimensions)));

            const Shape  q_A2_shape = quarter_shape(kind, 1, dimensions);
            const Shape  q_A3_shape = quarter_shape(kind, 2, dimensions);
            const Shape  q_A5_shape = quarter_shape(kind, 4, dimensions);
            const Matrix q_A2       = load_quarter_field(final_dir, variable, step, "A2", q_A2_shape);
            const Matrix q_A3       = load_quarter_field(final_dir, variable, step, "A3", q_A3_shape);
            const Matrix q_A5       = load_quarter_field(final_dir, variable, step, "A5", q_A5_shape);

            const bool x_aligned = kind == FieldKind::U || kind == FieldKind::Corner;
            const bool y_aligned = kind == FieldKind::V || kind == FieldKind::Corner;

            for (int i = 0; i < q_A2_shape.nx; ++i)
            {
                for (int j = 0; j < q_A2_shape.ny; ++j)
                {
                    const double value = q_A2[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
                    const int    xr    = dimensions.nx2_q + i;
                    const int    xl    = x_aligned ? dimensions.nx2_q - i : dimensions.nx2_q - 1 - i;
                    const int    yu    = dimensions.ny2_q + j;
                    const int    yl    = y_aligned ? dimensions.ny2_q - j : dimensions.ny2_q - 1 - j;
                    assign(result[1], xr, yu, value, 1.0, stats);
                    assign(result[1], xl, yu, value, sx, stats);
                    assign(result[1], xr, yl, value, sy, stats);
                    assign(result[1], xl, yl, value, sx * sy, stats);
                }
            }

            for (int i = 0; i < q_A3_shape.nx; ++i)
            {
                for (int j = 0; j < q_A3_shape.ny; ++j)
                {
                    const double value = q_A3[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
                    const int    yu    = dimensions.ny2_q + j;
                    const int    yl    = y_aligned ? dimensions.ny2_q - j : dimensions.ny2_q - 1 - j;
                    assign(result[2], i, yu, value, 1.0, stats);
                    assign(result[2], i, yl, value, sy, stats);

                    const int reflected_i = x_aligned ? dimensions.nx3 - i : dimensions.nx3 - 1 - i;
                    assign(result[0], reflected_i, yu, value, sx, stats);
                    assign(result[0], reflected_i, yl, value, sx * sy, stats);
                }
            }

            for (int i = 0; i < q_A5_shape.nx; ++i)
            {
                for (int j = 0; j < q_A5_shape.ny; ++j)
                {
                    const double value = q_A5[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
                    const int    xr    = dimensions.nx2_q + i;
                    const int    xl    = x_aligned ? dimensions.nx2_q - i : dimensions.nx2_q - 1 - i;
                    const int    yb    = y_aligned ? dimensions.ny5 - j : dimensions.ny5 - 1 - j;
                    assign(result[4], xr, j, value, 1.0, stats);
                    assign(result[4], xl, j, value, sx, stats);
                    assign(result[3], xr, yb, value, sy, stats);
                    assign(result[3], xl, yb, value, sx * sy, stats);
                }
            }

            if (kind == FieldKind::U)
            {
                for (int j = 0; j < q_A3_shape.ny; ++j)
                {
                    const double value = q_A3[0][static_cast<std::size_t>(j)];
                    const int    yu    = dimensions.ny2_q + j;
                    const int    yl    = dimensions.ny2_q - 1 - j;
                    assign(result[1], 0, yu, value, sx, stats);
                    assign(result[1], 0, yl, value, sx * sy, stats);
                }
            }
            else if (kind == FieldKind::V)
            {
                for (int i = 0; i < q_A5_shape.nx; ++i)
                {
                    const double value = q_A5[static_cast<std::size_t>(i)][0];
                    const int    xr    = dimensions.nx2_q + i;
                    const int    xl    = dimensions.nx2_q - 1 - i;
                    assign(result[1], xr, 0, value, sy, stats);
                    assign(result[1], xl, 0, value, sx * sy, stats);
                }
            }

            return result;
        }

        void validate_reconstruction(const std::vector<Matrix>& matrices,
                                     const std::string&         variable,
                                     const AssignmentStats&     stats,
                                     double                     tolerance)
        {
            if (stats.max_conflict > tolerance)
            {
                std::ostringstream message;
                message << "Mirror reconstruction conflict for " << variable << ": " << stats.max_conflict
                        << " exceeds tolerance " << tolerance;
                throw std::runtime_error(message.str());
            }
            for (const auto& matrix : matrices)
                for (const auto& row : matrix)
                    for (double value : row)
                        if (!std::isfinite(value))
                            throw std::runtime_error("Mirror reconstruction left an unset value for " + variable);
        }

        void write_manifest(const Result& result)
        {
            const fs::path path = result.output_final_dir.parent_path() / "mirror_manifest.csv";
            std::ofstream  output(path);
            if (!output.is_open())
                throw std::runtime_error("Failed to write mirror manifest: " + path.string());

            output << "source_final_dir,source_step,nx2_full,ny2_full,nx3,ny5,variable_count,file_count,"
                      "max_assignment_conflict\n";
            output << result.source_final_dir.string() << ',' << result.quarter_step << ','
                   << result.dimensions.nx2_full << ',' << result.dimensions.ny2_full << ',' << result.dimensions.nx3
                   << ',' << result.dimensions.ny5 << ',' << result.reconstructed_variables.size() << ','
                   << result.written_file_count << ',' << std::setprecision(16) << result.max_assignment_conflict
                   << '\n';
        }
    } // namespace

    std::vector<std::string> default_variables()
    {
        return {"u", "v", "p", "mu", "tau_xx", "tau_yy", "tau_xy", "vorticity", "phi", "c"};
    }

    int detect_latest_step(const std::filesystem::path& quarter_root)
    {
        const fs::path   final_dir = resolve_final_dir(quarter_root);
        const std::regex pattern("^u_([0-9]+)_A2\\.csv$");
        int              latest_step = -1;
        for (const auto& entry : fs::directory_iterator(final_dir))
        {
            if (!entry.is_regular_file())
                continue;
            std::smatch       match;
            const std::string name = entry.path().filename().string();
            if (std::regex_match(name, match, pattern))
                latest_step = std::max(latest_step, std::stoi(match[1].str()));
        }
        if (latest_step < 0)
            throw std::runtime_error("Failed to detect a quarter-domain final step in " + final_dir.string());
        return latest_step;
    }

    Dimensions infer_dimensions(const std::filesystem::path& quarter_root, int quarter_step)
    {
        if (quarter_step <= 0)
            throw std::runtime_error("quarter_step must be positive when inferring dimensions.");
        const fs::path final_dir = resolve_final_dir(quarter_root);
        const Matrix   p_A2      = read_matrix(field_path(final_dir, "p", quarter_step, "A2"));
        const Matrix   p_A3      = read_matrix(field_path(final_dir, "p", quarter_step, "A3"));
        const Matrix   p_A5      = read_matrix(field_path(final_dir, "p", quarter_step, "A5"));

        Dimensions dimensions;
        dimensions.nx2_q = static_cast<int>(p_A2.size());
        dimensions.ny2_q = static_cast<int>(p_A2.front().size());
        dimensions.nx3   = static_cast<int>(p_A3.size());
        dimensions.ny5   = static_cast<int>(p_A5.front().size());
        if (static_cast<int>(p_A3.front().size()) != dimensions.ny2_q ||
            static_cast<int>(p_A5.size()) != dimensions.nx2_q)
            throw std::runtime_error("A2/A3/A5 quarter-domain dimensions are inconsistent.");
        dimensions.nx2_full = 2 * dimensions.nx2_q;
        dimensions.ny2_full = 2 * dimensions.ny2_q;
        return dimensions;
    }

    Result reconstruct(const Options& options)
    {
        if (options.quarter_root.empty())
            throw std::runtime_error("quarter_root is required.");
        if (options.output_root.empty())
            throw std::runtime_error("output_root is required.");
        if (!std::isfinite(options.assignment_tolerance) || options.assignment_tolerance < 0.0)
            throw std::runtime_error("assignment_tolerance must be finite and non-negative.");

        Result result;
        result.source_final_dir = fs::absolute(resolve_final_dir(options.quarter_root)).lexically_normal();
        result.output_final_dir = fs::absolute(options.output_root / "final").lexically_normal();
        if (result.source_final_dir == result.output_final_dir)
            throw std::runtime_error("Mirror output final/ must differ from the quarter-domain source final/.");
        if (fs::exists(result.output_final_dir) && !fs::is_empty(result.output_final_dir))
            throw std::runtime_error("Mirror output final/ already exists and is not empty: " +
                                     result.output_final_dir.string());

        result.quarter_step =
            options.quarter_step > 0 ? options.quarter_step : detect_latest_step(options.quarter_root);
        result.dimensions = infer_dimensions(options.quarter_root, result.quarter_step);
        IO::create_directory(options.output_root);
        IO::create_directory(result.output_final_dir);

        const std::vector<std::string> variables  = options.variables.empty() ? default_variables() : options.variables;
        const std::string              domains[5] = {"A1", "A2", "A3", "A4", "A5"};
        for (const auto& variable : variables)
        {
            if (!fs::exists(field_path(result.source_final_dir, variable, result.quarter_step, "A2")))
                continue;

            AssignmentStats stats;
            const auto      matrices =
                reconstruct_variable(result.source_final_dir, variable, result.quarter_step, result.dimensions, stats);
            validate_reconstruction(matrices, variable, stats, options.assignment_tolerance);
            result.max_assignment_conflict = std::max(result.max_assignment_conflict, stats.max_conflict);
            for (int domain = 0; domain < 5; ++domain)
            {
                write_matrix(field_path(result.output_final_dir, variable, result.quarter_step, domains[domain]),
                             matrices[static_cast<std::size_t>(domain)]);
                ++result.written_file_count;
            }
            result.reconstructed_variables.push_back(variable);
        }
        if (result.reconstructed_variables.empty())
            throw std::runtime_error("No quarter-domain fields were reconstructed.");

        write_manifest(result);
        return result;
    }
} // namespace CrossChannelQuarterMirror
