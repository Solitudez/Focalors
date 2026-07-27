#pragma once

#include "io/case_base.hpp"
#include "io/common.h"
#include "io/csv_writer_2d.h"

#include <algorithm>
#include <regex>
#include <stdexcept>
#include <string>

namespace CrossSlotRestart
{
    struct FinalFieldRestartInfo
    {
        bool        enabled = false;
        std::string source_root;
        std::string final_dir;
        int         step = 0;
    };

    inline bool has_restart_request(const CaseBase& case_param)
    {
        return !case_param.savepoint_root_to_read.empty() && case_param.savepoint_root_to_read != "invalid";
    }

    inline fs::path resolve_final_dir(const std::string& source_root)
    {
        fs::path source_path(source_root);
        if (fs::is_directory(source_path / "final"))
            return source_path / "final";
        if (source_path.filename() == "final" && fs::is_directory(source_path))
            return source_path;

        throw std::runtime_error(
            "Restart source must be a run root containing final/ or the final/ directory itself: " + source_root);
    }

    inline int detect_latest_final_step(const fs::path& final_dir, const std::string& primary_var = "u")
    {
        const std::regex pattern("^" + primary_var + "_([0-9]+)_A1\\.csv$");
        int              latest_step = -1;

        for (const auto& entry : fs::directory_iterator(final_dir))
        {
            if (!entry.is_regular_file())
                continue;

            std::smatch       match;
            const std::string name = entry.path().filename().string();
            if (!std::regex_match(name, match, pattern))
                continue;

            latest_step = std::max(latest_step, std::stoi(match[1].str()));
        }

        if (latest_step < 0)
            throw std::runtime_error("Failed to detect final-field step from " + final_dir.string());

        return latest_step;
    }

    inline FinalFieldRestartInfo resolve_final_field_restart(const CaseBase&    case_param,
                                                             const std::string& primary_var = "u")
    {
        FinalFieldRestartInfo info;
        if (!has_restart_request(case_param))
            return info;

        const fs::path final_dir = resolve_final_dir(case_param.savepoint_root_to_read);
        const int      step =
            case_param.step_to_read > 0 ? case_param.step_to_read : detect_latest_final_step(final_dir, primary_var);

        const fs::path probe = final_dir / (primary_var + "_" + std::to_string(step) + "_A1.csv");
        if (!fs::exists(probe))
            throw std::runtime_error("Restart probe file does not exist: " + probe.string());

        info.enabled     = true;
        info.source_root = fs::absolute(final_dir.parent_path()).string();
        info.final_dir   = fs::absolute(final_dir).string();
        info.step        = step;
        return info;
    }

    inline void record_restart_metadata(IO::ParasRecord& paras_record, const FinalFieldRestartInfo& info)
    {
        paras_record.record("restart_from_final_field_enabled", info.enabled ? 1 : 0)
            .record("restart_from_final_field_source_root", info.enabled ? info.source_root : std::string(""))
            .record("restart_from_final_field_final_dir", info.enabled ? info.final_dir : std::string(""))
            .record("restart_from_final_field_step", info.enabled ? info.step : 0);
    }

    inline void warm_start_from_final_field(const FinalFieldRestartInfo& info,
                                            Variable2D&                  u,
                                            Variable2D&                  v,
                                            Variable2D&                  p,
                                            Variable2D*                  phi    = nullptr,
                                            Variable2D*                  c      = nullptr,
                                            Variable2D*                  mu     = nullptr,
                                            Variable2D*                  tau_xx = nullptr,
                                            Variable2D*                  tau_yy = nullptr,
                                            Variable2D*                  tau_xy = nullptr)
    {
        if (!info.enabled)
            return;

        const std::string step_tag = std::to_string(info.step);
        if (!IO::read_csv(u, info.final_dir + "/u_" + step_tag))
            throw std::runtime_error("Failed to load restart velocity u from " + info.final_dir);
        if (!IO::read_csv(v, info.final_dir + "/v_" + step_tag))
            throw std::runtime_error("Failed to load restart velocity v from " + info.final_dir);
        if (!IO::read_csv(p, info.final_dir + "/p_" + step_tag))
            throw std::runtime_error("Failed to load restart pressure p from " + info.final_dir);

        if (phi != nullptr)
        {
            const fs::path phi_probe = fs::path(info.final_dir) / ("phi_" + step_tag + "_A1.csv");
            if (fs::exists(phi_probe) && !IO::read_csv(*phi, info.final_dir + "/phi_" + step_tag))
                throw std::runtime_error("Failed to load restart electric potential phi from " + info.final_dir);
        }

        if (c != nullptr)
        {
            const fs::path c_probe = fs::path(info.final_dir) / ("c_" + step_tag + "_A1.csv");
            if (fs::exists(c_probe) && !IO::read_csv(*c, info.final_dir + "/c_" + step_tag))
                throw std::runtime_error("Failed to load restart scalar c from " + info.final_dir);
        }

        if (mu != nullptr)
        {
            const fs::path mu_probe = fs::path(info.final_dir) / ("mu_" + step_tag + "_A1.csv");
            if (fs::exists(mu_probe) && !IO::read_csv(*mu, info.final_dir + "/mu_" + step_tag))
                throw std::runtime_error("Failed to load restart viscosity mu from " + info.final_dir);
        }

        auto load_optional_field = [&](Variable2D* variable, const std::string& name) {
            if (variable == nullptr)
                return;
            const fs::path probe = fs::path(info.final_dir) / (name + "_" + step_tag + "_A1.csv");
            if (fs::exists(probe) && !IO::read_csv(*variable, info.final_dir + "/" + name + "_" + step_tag))
                throw std::runtime_error("Failed to load restart field " + name + " from " + info.final_dir);
        };
        load_optional_field(tau_xx, "tau_xx");
        load_optional_field(tau_yy, "tau_yy");
        load_optional_field(tau_xy, "tau_xy");
    }
} // namespace CrossSlotRestart
