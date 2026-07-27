#include "base/config.h"
#include "base/domain/domain2d.h"
#include "base/domain/geometry2d.h"
#include "base/domain/variable2d.h"
#include "io/csv_writer_2d.h"
#include "ns/ns_solver2d.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    struct ErrorStats
    {
        double max_abs_error = 0.0;
        double max_ref_abs   = 0.0;
        double min_ref       = std::numeric_limits<double>::max();
        double max_ref       = -std::numeric_limits<double>::max();
    };

    void set_dirichlet_zero(Variable2D& var, Domain2DUniform* domain, LocationType loc)
    {
        var.set_boundary_type(domain, loc, PDEBoundaryType::Dirichlet);
        var.set_boundary_value(domain, loc, 0.0);
    }

    void set_neumann_zero(Variable2D& var, Domain2DUniform* domain, LocationType loc)
    {
        var.set_boundary_type(domain, loc, PDEBoundaryType::Neumann);
        var.set_boundary_value(domain, loc, 0.0);
    }

    void set_cross_slot_boundaries(Geometry2D&                           geo,
                                   Variable2D&                           u,
                                   Variable2D&                           v,
                                   Variable2D&                           p,
                                   const std::vector<Domain2DUniform*>& domains,
                                   Domain2DUniform*                      A1,
                                   Domain2DUniform*                      A3,
                                   Domain2DUniform*                      A4,
                                   Domain2DUniform*                      A5)
    {
        const std::vector<LocationType> dirs = {
            LocationType::XNegative, LocationType::XPositive, LocationType::YNegative, LocationType::YPositive};

        auto is_adjacented = [&](Domain2DUniform* domain, LocationType loc) {
            return geo.adjacency.count(domain) != 0 && geo.adjacency[domain].count(loc) != 0;
        };
        auto is_pressure_outlet = [&](Domain2DUniform* domain, LocationType loc) {
            return (domain == A4 && loc == LocationType::YNegative) ||
                   (domain == A5 && loc == LocationType::YPositive);
        };

        for (auto* domain : domains)
        {
            for (auto loc : dirs)
            {
                if (is_adjacented(domain, loc))
                    continue;

                set_dirichlet_zero(u, domain, loc);
                set_dirichlet_zero(v, domain, loc);

                if (is_pressure_outlet(domain, loc))
                    set_dirichlet_zero(p, domain, loc);
                else
                    set_neumann_zero(p, domain, loc);
            }
        }

        const double inlet_profile[5] = {0.54, 1.26, 1.50, 1.26, 0.54};

        double* left_inlet = u.boundary_value_map.at(A1).at(LocationType::XNegative);
        for (int j = 0; j < A1->get_ny(); ++j)
            left_inlet[j] = inlet_profile[j];

        double* right_inlet = u.boundary_value_map.at(A3).at(LocationType::XPositive);
        for (int j = 0; j < A3->get_ny(); ++j)
            right_inlet[j] = -inlet_profile[j];

        set_neumann_zero(u, A4, LocationType::YNegative);
        set_neumann_zero(v, A4, LocationType::YNegative);
        set_neumann_zero(u, A5, LocationType::YPositive);
        set_neumann_zero(v, A5, LocationType::YPositive);
    }

    void reference_raw_vorticity(const Variable2D& u_var, const Variable2D& v_var, Variable2D& omega_ref_var)
    {
        for (auto* domain : u_var.geometry->domains)
        {
            const field2& u         = *u_var.field_map.at(domain);
            const field2& v         = *v_var.field_map.at(domain);
            field2&       omega_ref = *omega_ref_var.field_map.at(domain);

            const int    nx = domain->get_nx();
            const int    ny = domain->get_ny();
            const double hx = domain->get_hx();
            const double hy = domain->get_hy();

            const double* v_xneg_buffer = v_var.buffer_map.at(domain).at(LocationType::XNegative);
            const double* u_yneg_buffer = u_var.buffer_map.at(domain).at(LocationType::YNegative);

            for (int i = 0; i < nx; ++i)
            {
                for (int j = 0; j < ny; ++j)
                {
                    const double v_right = v(i, j);
                    const double v_left  = (i == 0) ? v_xneg_buffer[j] : v(i - 1, j);
                    const double u_up    = u(i, j);
                    const double u_down  = (j == 0) ? u_yneg_buffer[i] : u(i, j - 1);

                    const double dv_dx = (v_right - v_left) / hx;
                    const double du_dy = (u_up - u_down) / hy;
                    omega_ref(i, j)    = dv_dx - du_dy;
                }
            }
        }
    }

    ErrorStats compare_fields(const Variable2D& omega, const Variable2D& omega_ref)
    {
        ErrorStats stats;
        for (auto* domain : omega.geometry->domains)
        {
            const field2& field     = *omega.field_map.at(domain);
            const field2& ref_field = *omega_ref.field_map.at(domain);

            for (int i = 0; i < field.get_nx(); ++i)
            {
                for (int j = 0; j < field.get_ny(); ++j)
                {
                    const double ref_val = ref_field(i, j);
                    stats.max_abs_error  = std::max(stats.max_abs_error, std::abs(field(i, j) - ref_val));
                    stats.max_ref_abs    = std::max(stats.max_ref_abs, std::abs(ref_val));
                    stats.min_ref        = std::min(stats.min_ref, ref_val);
                    stats.max_ref        = std::max(stats.max_ref, ref_val);
                }
            }
        }
        return stats;
    }
} // namespace

int main()
{
    constexpr int    n      = 5;
    constexpr double length = 1.0;

    EnvironmentConfig::Get().showGmresRes    = false;
    EnvironmentConfig::Get().showCurrentStep = false;

    TimeAdvancingConfig& time_cfg = TimeAdvancingConfig::Get();
    time_cfg.dt                   = 0.005;
    time_cfg.num_iterations       = 5;
    time_cfg.t_max                = time_cfg.dt * static_cast<double>(time_cfg.num_iterations);
    time_cfg.corr_iter            = 1;

    PhysicsConfig& physics_cfg = PhysicsConfig::Get();
    physics_cfg.set_Re(25.0);
    physics_cfg.set_enable_mhd(false);

    Geometry2D geo;
    Domain2DUniform A2(n, n, length, length, "A2");
    Domain2DUniform A1(n, n, length, length, "A1");
    Domain2DUniform A3(n, n, length, length, "A3");
    Domain2DUniform A4(n, n, length, length, "A4");
    Domain2DUniform A5(n, n, length, length, "A5");

    geo.connect(&A2, LocationType::XNegative, &A1);
    geo.connect(&A2, LocationType::XPositive, &A3);
    geo.connect(&A2, LocationType::YNegative, &A4);
    geo.connect(&A2, LocationType::YPositive, &A5);
    geo.axis(&A2, LocationType::XNegative);
    geo.axis(&A2, LocationType::YNegative);
    geo.check();
    geo.solve_prepare();

    std::vector<Domain2DUniform*> domains = {&A1, &A2, &A3, &A4, &A5};

    Variable2D u("u"), v("v"), p("p"), omega("vorticity_raw"), omega_ref("vorticity_reference");
    u.set_geometry(geo);
    v.set_geometry(geo);
    p.set_geometry(geo);
    omega.set_geometry(geo);
    omega_ref.set_geometry(geo);

    field2 u_A1, u_A2, u_A3, u_A4, u_A5;
    field2 v_A1, v_A2, v_A3, v_A4, v_A5;
    field2 p_A1, p_A2, p_A3, p_A4, p_A5;
    field2 omega_A1, omega_A2, omega_A3, omega_A4, omega_A5;
    field2 omega_ref_A1, omega_ref_A2, omega_ref_A3, omega_ref_A4, omega_ref_A5;

    u.set_x_edge_field(&A1, u_A1);
    u.set_x_edge_field(&A2, u_A2);
    u.set_x_edge_field(&A3, u_A3);
    u.set_x_edge_field(&A4, u_A4);
    u.set_x_edge_field(&A5, u_A5);
    v.set_y_edge_field(&A1, v_A1);
    v.set_y_edge_field(&A2, v_A2);
    v.set_y_edge_field(&A3, v_A3);
    v.set_y_edge_field(&A4, v_A4);
    v.set_y_edge_field(&A5, v_A5);
    p.set_center_field(&A1, p_A1);
    p.set_center_field(&A2, p_A2);
    p.set_center_field(&A3, p_A3);
    p.set_center_field(&A4, p_A4);
    p.set_center_field(&A5, p_A5);
    omega.set_center_field(&A1, omega_A1);
    omega.set_center_field(&A2, omega_A2);
    omega.set_center_field(&A3, omega_A3);
    omega.set_center_field(&A4, omega_A4);
    omega.set_center_field(&A5, omega_A5);
    omega_ref.set_center_field(&A1, omega_ref_A1);
    omega_ref.set_center_field(&A2, omega_ref_A2);
    omega_ref.set_center_field(&A3, omega_ref_A3);
    omega_ref.set_center_field(&A4, omega_ref_A4);
    omega_ref.set_center_field(&A5, omega_ref_A5);

    set_cross_slot_boundaries(geo, u, v, p, domains, &A1, &A3, &A4, &A5);

    ConcatPoissonSolver2D p_solver(&p);
    p_solver.set_parameter(30, 1.0e-12, 300);

    ConcatNSSolver2D ns_solver(&u, &v, &p, &p_solver);
    for (int step = 0; step < time_cfg.num_iterations; ++step)
        ns_solver.solve();
    ns_solver.phys_boundary_update();
    ns_solver.nondiag_shared_boundary_update();
    ns_solver.diag_shared_boundary_update();

    ns_solver.raw_vorticity_update(omega);
    reference_raw_vorticity(u, v, omega_ref);

    const ErrorStats stats     = compare_fields(omega, omega_ref);
    const std::string out_root = "result/raw_vorticity_cross_5x5";
    IO::write_csv(u, out_root + "/u/u_5");
    IO::write_csv(v, out_root + "/v/v_5");
    IO::write_csv(omega, out_root + "/vorticity_raw/vorticity_raw_5");
    IO::write_csv(omega_ref, out_root + "/vorticity_reference/vorticity_reference_5");

    std::cout << "raw_vorticity_cross_5x5 real-case max_error = " << stats.max_abs_error
              << ", ref_range = [" << stats.min_ref << ", " << stats.max_ref << "]" << std::endl;

    if (stats.max_ref_abs < 1.0e-10)
        throw std::runtime_error("raw_vorticity_cross_5x5 failed: reference vorticity is nearly zero");
    if (stats.max_abs_error > 1.0e-12)
        throw std::runtime_error("raw_vorticity_cross_5x5 failed: raw vorticity mismatch");

    return 0;
}
