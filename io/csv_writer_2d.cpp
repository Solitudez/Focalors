#include "csv_writer_2d.h"
#include "common.h"

#include <fstream>
#include <iomanip>
#include <sstream>
#include <vector>

namespace IO
{
    namespace
    {
        const char* position_type_to_string(VariablePositionType type)
        {
            switch (type)
            {
                case VariablePositionType::Center:
                    return "Center";
                case VariablePositionType::XFace:
                    return "XFace";
                case VariablePositionType::YFace:
                    return "YFace";
                case VariablePositionType::Corner:
                    return "Corner";
                default:
                    return "Null";
            }
        }

        void position_shift(VariablePositionType type, double& shift_x, double& shift_y)
        {
            switch (type)
            {
                case VariablePositionType::Center:
                    shift_x = 0.5;
                    shift_y = 0.5;
                    break;
                case VariablePositionType::XFace:
                    shift_x = 0.0;
                    shift_y = 0.5;
                    break;
                case VariablePositionType::YFace:
                    shift_x = 0.5;
                    shift_y = 0.0;
                    break;
                case VariablePositionType::Corner:
                    shift_x = 0.0;
                    shift_y = 0.0;
                    break;
                default:
                    shift_x = 0.0;
                    shift_y = 0.0;
                    break;
            }
        }
    } // namespace

    bool write_csv(double** value, int nx, int ny, const std::string& filename)
    {
        fs::path path(filename);
        fs::path dir = path.parent_path();
        IO::create_directory(dir);

        std::ofstream outfile(filename + ".csv");

        if (!outfile.is_open())
        {
            std::cerr << "Failed to open file: " << filename + ".csv" << std::endl;
            return false;
        }
        outfile << std::setprecision(17);

        for (int i = 0; i < nx; i++)
        {
            for (int j = 0; j < ny; j++)
            {
                if (j == (ny - 1))
                {
                    outfile << value[i][j] << std::endl;
                }
                else
                {
                    outfile << value[i][j] << ",";
                }
            }
        }
        outfile.close();

        if (outfile.fail())
        {
            std::cout << "Writing to file failed: " << std::endl;
            return false;
        }

        return true;
    }

    bool write_csv(field2& field, const std::string& filename)
    {
        fs::path path(filename);
        fs::path dir = path.parent_path();
        IO::create_directory(dir);

        std::ofstream outfile(filename + ".csv");

        if (!outfile.is_open())
        {
            std::cerr << "Failed to open file: " << filename + ".csv" << std::endl;
            return false;
        }
        outfile << std::setprecision(17);

        for (int i = 0; i < field.get_nx(); i++)
        {
            for (int j = 0; j < field.get_ny(); j++)
            {
                if (j == (field.get_ny() - 1))
                {
                    outfile << field(i, j) << std::endl;
                }
                else
                {
                    outfile << field(i, j) << ",";
                }
            }
        }
        outfile.close();

        if (outfile.fail())
        {
            std::cout << "Writing to file failed: " << std::endl;
            return false;
        }

        return true;
    }

    bool read_csv(field2& field, const std::string& filename)
    {
        std::ifstream infile(filename + ".csv");

        if (!infile.is_open())
        {
            std::cerr << "Failed to open file: " << filename + ".csv" << std::endl;
            return false;
        }

        std::string line;
        int         i = 0;

        while (std::getline(infile, line))
        {
            std::stringstream ss(line);
            std::string       value;
            int               j = 0;

            while (std::getline(ss, value, ','))
            {
                try
                {
                    double numeric_value = std::stod(value);
                    field(i, j)          = numeric_value;
                    j++;
                }
                catch (const std::invalid_argument&)
                {
                    std::cerr << "Invalid number at i " << i << ", j " << j << ": " << value << std::endl;
                }
            }
            i++;
        }

        infile.close();
        return true;
    }

    bool read_csv(Variable2D& var, const std::string& filename)
    {
        if (!var.geometry)
        {
            std::cerr << "[read_csv] Error: variable has no geometry" << std::endl;
            return false;
        }

        auto read_matrix = [](const std::string& file_path, std::vector<std::vector<double>>& rows) -> bool {
            std::ifstream infile(file_path);
            if (!infile.is_open())
            {
                std::cerr << "Failed to open file: " << file_path << std::endl;
                return false;
            }

            rows.clear();
            std::string line;
            while (std::getline(infile, line))
            {
                std::stringstream   ss(line);
                std::string         value;
                std::vector<double> row;
                while (std::getline(ss, value, ','))
                {
                    try
                    {
                        row.push_back(std::stod(value));
                    }
                    catch (const std::exception&)
                    {
                        std::cerr << "[read_csv] Invalid number in " << file_path << ": " << value << std::endl;
                        return false;
                    }
                }
                rows.push_back(std::move(row));
            }

            if (!infile.good() && !infile.eof())
            {
                std::cerr << "[read_csv] Failed while reading file: " << file_path << std::endl;
                return false;
            }
            return true;
        };

        auto zero_buffers = [](Variable2D& in_var, Domain2DUniform* domain) {
            auto buffer_it = in_var.buffer_map.find(domain);
            if (buffer_it == in_var.buffer_map.end())
                return;

            const int nx = domain->get_nx();
            const int ny = domain->get_ny();

            auto zero_if_present = [&](LocationType loc, int count) {
                const auto it = buffer_it->second.find(loc);
                if (it == buffer_it->second.end() || it->second == nullptr)
                    return;
                std::fill(it->second, it->second + count, 0.0);
            };

            switch (in_var.position_type)
            {
                case VariablePositionType::Center:
                case VariablePositionType::XFace:
                case VariablePositionType::YFace:
                    zero_if_present(LocationType::XNegative, ny);
                    zero_if_present(LocationType::XPositive, ny);
                    zero_if_present(LocationType::YNegative, nx);
                    zero_if_present(LocationType::YPositive, nx);
                    break;
                case VariablePositionType::Corner:
                    zero_if_present(LocationType::XNegative, ny + 1);
                    zero_if_present(LocationType::XPositive, ny + 1);
                    zero_if_present(LocationType::YNegative, nx + 1);
                    zero_if_present(LocationType::YPositive, nx + 1);
                    break;
                default:
                    break;
            }
        };

        for (auto* domain : var.geometry->domains)
        {
            try
            {
                field2& field = *var.field_map.at(domain);
                zero_buffers(var, domain);

                const std::string                file_path = filename + "_" + domain->name + ".csv";
                std::vector<std::vector<double>> rows;
                if (!read_matrix(file_path, rows))
                    return false;

                const int field_nx = field.get_nx();
                const int field_ny = field.get_ny();

                auto require_uniform_cols = [&](int expected_cols) -> bool {
                    for (std::size_t i = 0; i < rows.size(); ++i)
                    {
                        if (static_cast<int>(rows[i].size()) != expected_cols)
                        {
                            std::cerr << "[read_csv] Column-count mismatch in " << file_path << " at row " << i
                                      << ": expected " << expected_cols << ", got " << rows[i].size() << std::endl;
                            return false;
                        }
                    }
                    return true;
                };

                if (var.position_type == VariablePositionType::Center ||
                    var.position_type == VariablePositionType::Corner)
                {
                    if (static_cast<int>(rows.size()) != field_nx)
                    {
                        std::cerr << "[read_csv] Row-count mismatch in " << file_path << ": expected " << field_nx
                                  << ", got " << rows.size() << std::endl;
                        return false;
                    }
                    if (!require_uniform_cols(field_ny))
                        return false;

                    for (int i = 0; i < field_nx; ++i)
                        for (int j = 0; j < field_ny; ++j)
                            field(i, j) = rows[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];

                    continue;
                }

                if (var.position_type == VariablePositionType::XFace)
                {
                    const auto& boundary_type = var.boundary_type_map.at(domain);
                    const bool  has_xpos_buffer =
                        boundary_type.at(LocationType::XPositive) != PDEBoundaryType::Adjacented;
                    const int expected_rows = field_nx + (has_xpos_buffer ? 1 : 0);
                    if (static_cast<int>(rows.size()) != expected_rows)
                    {
                        std::cerr << "[read_csv] Row-count mismatch in " << file_path << ": expected " << expected_rows
                                  << ", got " << rows.size() << std::endl;
                        return false;
                    }
                    if (!require_uniform_cols(field_ny))
                        return false;

                    for (int i = 0; i < field_nx; ++i)
                        for (int j = 0; j < field_ny; ++j)
                            field(i, j) = rows[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];

                    if (has_xpos_buffer)
                    {
                        double* xpos_buffer = var.buffer_map.at(domain).at(LocationType::XPositive);
                        for (int j = 0; j < field_ny; ++j)
                            xpos_buffer[j] = rows[static_cast<std::size_t>(field_nx)][static_cast<std::size_t>(j)];
                    }

                    continue;
                }

                if (var.position_type == VariablePositionType::YFace)
                {
                    const auto& boundary_type = var.boundary_type_map.at(domain);
                    const bool  has_ypos_buffer =
                        boundary_type.at(LocationType::YPositive) != PDEBoundaryType::Adjacented;
                    const int expected_cols = field_ny + (has_ypos_buffer ? 1 : 0);
                    if (static_cast<int>(rows.size()) != field_nx)
                    {
                        std::cerr << "[read_csv] Row-count mismatch in " << file_path << ": expected " << field_nx
                                  << ", got " << rows.size() << std::endl;
                        return false;
                    }
                    if (!require_uniform_cols(expected_cols))
                        return false;

                    for (int i = 0; i < field_nx; ++i)
                    {
                        for (int j = 0; j < field_ny; ++j)
                            field(i, j) = rows[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];

                        if (has_ypos_buffer)
                        {
                            double* ypos_buffer = var.buffer_map.at(domain).at(LocationType::YPositive);
                            ypos_buffer[i]      = rows[static_cast<std::size_t>(i)][static_cast<std::size_t>(field_ny)];
                        }
                    }

                    continue;
                }

                std::cerr << "[read_csv] Unsupported VariablePositionType for variable " << var.name << std::endl;
                return false;
            }
            catch (const std::exception& e)
            {
                std::cerr << "[read_csv] Error: " << e.what() << std::endl;
                return false;
            }
        }

        return true;
    }

    bool write_csv(field2& field, double* buffer, const std::string& filename, VariablePositionType pos_type)
    {
        fs::path path(filename);
        fs::path dir = path.parent_path();
        IO::create_directory(dir);

        std::ofstream outfile(filename + ".csv");

        if (!outfile.is_open())
        {
            std::cerr << "Failed to open file: " << filename + ".csv" << std::endl;
            return false;
        }
        outfile << std::setprecision(17);

        int nx = field.get_nx();
        int ny = field.get_ny();

        if (pos_type == VariablePositionType::XFace)
        {
            for (int i = 0; i < nx + 1; i++)
            {
                for (int j = 0; j < ny; j++)
                {
                    double val = 0;
                    if (i == nx)
                    {
                        val = buffer[j];
                    }
                    else
                    {
                        val = field(i, j);
                    }

                    if (j == (ny - 1))
                    {
                        outfile << val << std::endl;
                    }
                    else
                    {
                        outfile << val << ",";
                    }
                }
            }
        }
        else if (pos_type == VariablePositionType::YFace)
        {
            for (int i = 0; i < nx; i++)
            {
                for (int j = 0; j < ny + 1; j++)
                {
                    double val = 0;
                    if (j == ny)
                    {
                        val = buffer[i];
                    }
                    else
                    {
                        val = field(i, j);
                    }

                    if (j == ny)
                    {
                        outfile << val << std::endl;
                    }
                    else
                    {
                        outfile << val << ",";
                    }
                }
            }
        }
        outfile.close();

        if (outfile.fail())
        {
            std::cout << "Writing to file failed: " << std::endl;
            return false;
        }

        return true;
    }

    bool write_csv(const Variable2D& var, const std::string& filename)
    {
        auto& domains        = var.geometry->domains;
        auto& field_map      = var.field_map;
        auto& buffer_map     = var.buffer_map;
        auto& boundary_types = var.boundary_type_map;

        for (auto& domain : domains)
        {
            try
            {
                auto& field         = field_map.at(domain);
                auto& buffers       = buffer_map.at(domain);
                auto& boundary_type = boundary_types.at(domain);

                int nx = field->get_nx();
                int ny = field->get_ny();
                if (var.position_type == VariablePositionType::XFace)
                {
                    if (boundary_type.at(LocationType::XPositive) == PDEBoundaryType::Adjacented)
                        write_csv(*field, filename + "_" + domain->name);
                    else
                        write_csv(*field,
                                  buffers.at(LocationType::XPositive),
                                  filename + "_" + domain->name,
                                  VariablePositionType::XFace);
                }
                else if (var.position_type == VariablePositionType::YFace)
                {
                    if (boundary_type.at(LocationType::YPositive) == PDEBoundaryType::Adjacented)
                        write_csv(*field, filename + "_" + domain->name);
                    else
                        write_csv(*field,
                                  buffers.at(LocationType::YPositive),
                                  filename + "_" + domain->name,
                                  VariablePositionType::YFace);
                }
                else
                {
                    write_csv(*field, filename + "_" + domain->name);
                }
            }
            catch (const std::exception& e)
            {
                std::cerr << "[write_csv] Error: " << e.what() << std::endl;
                return false;
            }
        }
        return true;
    }

    bool matlab_read_var(const Variable2D& var, const std::string& filename)
    {
        if (!var.geometry)
        {
            std::cerr << "[matlab_read_var] Error: variable has no geometry" << std::endl;
            return false;
        }

        fs::path    path(filename);
        fs::path    dir         = path.parent_path();
        std::string stem        = path.stem().string();
        std::string base        = stem;
        std::string ext         = path.extension().string();
        std::string read_suffix = "_read";

        if (ext == ".m")
        {
            if (base.size() > read_suffix.size() &&
                base.compare(base.size() - read_suffix.size(), read_suffix.size(), read_suffix) == 0)
            {
                base = base.substr(0, base.size() - read_suffix.size());
            }
        }

        fs::path csv_base_path = dir / base;
        fs::path script_path   = dir / (base + "_read.m");

        if (!dir.empty())
            IO::create_directory(dir.string());

        std::ofstream outfile(script_path.string());

        if (!outfile.is_open())
        {
            std::cerr << "Failed to open file: " << script_path.string() << std::endl;
            return false;
        }

        double shift_x = 0.0;
        double shift_y = 0.0;
        position_shift(var.position_type, shift_x, shift_y);

        outfile << "% Auto-generated by IO::matlab_read_var\n";
        outfile << "base = '" << csv_base_path.generic_string() << "';\n";
        outfile << "pos_type = '" << position_type_to_string(var.position_type) << "';\n";
        outfile << "shift_x = " << shift_x << ";\n";
        outfile << "shift_y = " << shift_y << ";\n";
        outfile << "domains = struct([]);\n\n";

        int idx = 1;
        for (auto& domain : var.geometry->domains)
        {
            outfile << "domains(" << idx << ").name = '" << domain->name << "';\n";
            outfile << "domains(" << idx << ").offset = [" << domain->get_offset_x() << ", " << domain->get_offset_y()
                    << "];\n";
            outfile << "domains(" << idx << ").hx = " << domain->get_hx() << ";\n";
            outfile << "domains(" << idx << ").hy = " << domain->get_hy() << ";\n";
            outfile << "domains(" << idx << ").nx = " << domain->get_nx() << ";\n";
            outfile << "domains(" << idx << ").ny = " << domain->get_ny() << ";\n";
            outfile << "domains(" << idx << ").lx = " << domain->get_lx() << ";\n";
            outfile << "domains(" << idx << ").ly = " << domain->get_ly() << ";\n";
            outfile << "domains(" << idx << ").field = readmatrix([base '_" << domain->name << ".csv']);\n";
            outfile << "domains(" << idx << ").size = size(domains(" << idx << ").field);\n";
            outfile << "domains(" << idx << ").x = domains(" << idx << ").offset(1) + (shift_x + (0:domains(" << idx
                    << ").size(1)-1)) * domains(" << idx << ").hx;\n";
            outfile << "domains(" << idx << ").y = domains(" << idx << ").offset(2) + (shift_y + (0:domains(" << idx
                    << ").size(2)-1)) * domains(" << idx << ").hy;\n\n";
            idx++;
        }

        outfile.close();

        if (outfile.fail())
        {
            std::cout << "Writing to file failed: " << std::endl;
            return false;
        }

        return true;
    }
} // namespace IO
