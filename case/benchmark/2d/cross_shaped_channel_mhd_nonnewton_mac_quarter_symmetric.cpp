#include "base/config.h"
#include "base/domain/domain2d.h"
#include "base/domain/geometry2d.h"
#include "base/domain/variable2d.h"
#include "base/field/field2.h"
#include "base/location_boundary.h"
#include "cross_shaped_channel.h"
#include "io/common.h"
#include "io/csv_writer_2d.h"
#include "ns/ns_solver2d.h"
#include "ns/scalar_solver2d.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace
{
    constexpr double EXPLICIT_DIFFUSION_DT_FACTOR = 0.20;
    constexpr double MAGNETIC_DT_FACTOR           = 0.50;
    constexpr double SMALL_NUMBER                 = 1.0e-12;

    double scale_viscosity_to_solver_units(double viscosity_value, const PhysicsConfig& physics_cfg)
    {
        if (physics_cfg.model_type == 0 || !physics_cfg.use_dimensionless_viscosity)
            return viscosity_value;

        const double scale = physics_cfg.mu_ref * physics_cfg.Re;
        return scale > 0.0 ? viscosity_value / scale : viscosity_value;
    }

    double select_dt(double h, double dt_factor, const PhysicsConfig& physics_cfg)
    {
        const double convective_dt = dt_factor * h;
        const double viscosity_raw = physics_cfg.model_type == 0 ? physics_cfg.nu : physics_cfg.mu_max;
        const double viscosity     = scale_viscosity_to_solver_units(viscosity_raw, physics_cfg);
        const double diffusion_dt  = viscosity > 0.0 ? EXPLICIT_DIFFUSION_DT_FACTOR * h * h / viscosity :
                                                       std::numeric_limits<double>::infinity();
        const double b_sq =
            physics_cfg.Bx * physics_cfg.Bx + physics_cfg.By * physics_cfg.By + physics_cfg.Bz * physics_cfg.Bz;
        const double magnetic_dt = std::abs(physics_cfg.Ha) > 0.0 && b_sq > 0.0 && physics_cfg.Re > 0.0 ?
                                       MAGNETIC_DT_FACTOR * physics_cfg.Re / (physics_cfg.Ha * physics_cfg.Ha * b_sq) :
                                       std::numeric_limits<double>::infinity();
        return std::min(convective_dt, std::min(diffusion_dt, magnetic_dt));
    }

    DifferenceSchemeType parse_scalar_scheme(const std::string& value)
    {
        if (value == "quick" || value == "QUICK")
            return DifferenceSchemeType::Conv_QUICK_Diff_Center2nd;
        if (value == "upwind" || value == "uw1")
            return DifferenceSchemeType::Conv_Upwind1st_Diff_Center2nd;
        if (value == "center" || value == "cd2")
            return DifferenceSchemeType::Conv_Center2nd_Diff_Center2nd;
        if (value == "tvd" || value == "vanleer")
            return DifferenceSchemeType::Conv_TVD_VanLeer_Diff_Center2nd;
        throw std::runtime_error("Unknown scalar_scheme: " + value);
    }

    void configure_physics(const CrossShapedChannel2DCase& case_param)
    {
        PhysicsConfig& physics_cfg = PhysicsConfig::Get();
        physics_cfg.set_Re(case_param.Re);
        physics_cfg.set_enable_mhd(std::abs(case_param.Ha) > 0.0);
        physics_cfg.Ha = case_param.Ha;
        physics_cfg.set_magnetic_field(case_param.Bx, case_param.By, case_param.Bz);
        physics_cfg.set_model_type(case_param.model_type);
        physics_cfg.set_gamma_ref(case_param.gamma_ref);

        if (case_param.model_type == 1)
        {
            physics_cfg.set_power_law_dimensionless(case_param.k_pl,
                                                    case_param.n_index,
                                                    case_param.Re,
                                                    case_param.mu_ref,
                                                    case_param.use_dimensionless_viscosity,
                                                    case_param.mu_min_pl,
                                                    case_param.mu_max_pl);
        }
        else if (case_param.model_type == 2)
        {
            physics_cfg.set_carreau_dimensionless(case_param.mu_0,
                                                  case_param.mu_inf,
                                                  case_param.a,
                                                  case_param.lambda,
                                                  case_param.n_index,
                                                  case_param.Re,
                                                  case_param.mu_ref,
                                                  case_param.use_dimensionless_viscosity,
                                                  case_param.mu_min_pl,
                                                  case_param.mu_max_pl);
        }
        else if (case_param.model_type == 3)
        {
            physics_cfg.set_casson_dimensionless(case_param.casson_mu,
                                                 case_param.casson_tau0,
                                                 case_param.Re,
                                                 case_param.mu_ref,
                                                 case_param.use_dimensionless_viscosity,
                                                 case_param.mu_min_pl,
                                                 case_param.mu_max_pl);
        }
    }

    void set_dirichlet_zero(Variable2D& variable, Domain2DUniform* domain, LocationType location)
    {
        variable.set_boundary_type(domain, location, PDEBoundaryType::Dirichlet);
        variable.set_boundary_value(domain, location, 0.0);
    }

    void set_neumann_zero(Variable2D& variable, Domain2DUniform* domain, LocationType location)
    {
        variable.set_boundary_type(domain, location, PDEBoundaryType::Neumann);
        variable.set_boundary_value(domain, location, 0.0);
    }

    double max_abs_xneg(const field2& field)
    {
        double result = 0.0;
        for (int j = 0; j < field.get_ny(); ++j)
            result = std::max(result, std::abs(field(0, j)));
        return result;
    }

    double max_abs_yneg(const field2& field)
    {
        double result = 0.0;
        for (int i = 0; i < field.get_nx(); ++i)
            result = std::max(result, std::abs(field(i, 0)));
        return result;
    }

    double max_xneg_ghost_mismatch(const Variable2D& variable, Domain2DUniform* domain, const field2& field)
    {
        const auto domain_it = variable.buffer_map.find(domain);
        if (domain_it == variable.buffer_map.end() || domain_it->second.at(LocationType::XNegative) == nullptr)
            return std::numeric_limits<double>::infinity();

        const double* buffer = domain_it->second.at(LocationType::XNegative);
        double        result = 0.0;
        const int     count  = std::min(field.get_ny(), domain->get_ny() + 1);
        for (int j = 0; j < count; ++j)
            result = std::max(result, std::abs(buffer[j] - field(0, j)));
        return result;
    }

    double max_yneg_ghost_mismatch(const Variable2D& variable, Domain2DUniform* domain, const field2& field)
    {
        const auto domain_it = variable.buffer_map.find(domain);
        if (domain_it == variable.buffer_map.end() || domain_it->second.at(LocationType::YNegative) == nullptr)
            return std::numeric_limits<double>::infinity();

        const double* buffer = domain_it->second.at(LocationType::YNegative);
        double        result = 0.0;
        const int     count  = std::min(field.get_nx(), domain->get_nx() + 1);
        for (int i = 0; i < count; ++i)
            result = std::max(result, std::abs(buffer[i] - field(i, 0)));
        return result;
    }

    void initialize_previous_field_map(Variable2D&                                   variable,
                                       std::unordered_map<Domain2DUniform*, field2>& previous,
                                       const std::string&                            prefix)
    {
        for (auto* domain : variable.geometry->domains)
        {
            field2& current = *variable.field_map.at(domain);
            field2& prior   = previous[domain];
            prior.init(current.get_nx(), current.get_ny(), prefix + "_" + domain->name);
            prior = current;
        }
    }

    double compute_relative_field_update(Variable2D& variable, std::unordered_map<Domain2DUniform*, field2>& previous)
    {
        double diff_sq = 0.0;
        double norm_sq = 0.0;
        for (auto* domain : variable.geometry->domains)
        {
            field2& current = *variable.field_map.at(domain);
            field2& prior   = previous.at(domain);
            field2  diff    = current - prior;
            diff_sq += diff.squared_sum();
            norm_sq += current.squared_sum();
        }
        return std::sqrt(diff_sq / std::max(norm_sq, SMALL_NUMBER));
    }

    void write_symmetry_row(std::ofstream&   out,
                            int              step,
                            double           time,
                            Variable2D&      u,
                            Variable2D&      v,
                            Variable2D&      tau_xy,
                            Domain2DUniform* A2,
                            Domain2DUniform* A3,
                            Domain2DUniform* A5,
                            double           steady_residual,
                            int              steady_hits)
    {
        const field2& u_A2   = *u.field_map.at(A2);
        const field2& v_A2   = *v.field_map.at(A2);
        const field2& u_A3   = *u.field_map.at(A3);
        const field2& v_A3   = *v.field_map.at(A3);
        const field2& u_A5   = *u.field_map.at(A5);
        const field2& v_A5   = *v.field_map.at(A5);
        const field2& txy_A2 = *tau_xy.field_map.at(A2);
        const field2& txy_A3 = *tau_xy.field_map.at(A3);
        const field2& txy_A5 = *tau_xy.field_map.at(A5);

        const double ux_axis          = std::max(max_abs_xneg(u_A2), max_abs_xneg(u_A5));
        const double vy_axis          = std::max(max_abs_yneg(v_A2), max_abs_yneg(v_A3));
        const double txy_x_axis       = std::max(max_abs_xneg(txy_A2), max_abs_xneg(txy_A5));
        const double txy_y_axis       = std::max(max_abs_yneg(txy_A2), max_abs_yneg(txy_A3));
        const double txy_axis         = std::max(txy_x_axis, txy_y_axis);
        const double tangential_ghost = std::max({max_yneg_ghost_mismatch(u, A2, u_A2),
                                                  max_yneg_ghost_mismatch(u, A3, u_A3),
                                                  max_xneg_ghost_mismatch(v, A2, v_A2),
                                                  max_xneg_ghost_mismatch(v, A5, v_A5)});

        out << step << "," << time << "," << ux_axis << "," << vy_axis << "," << tangential_ghost << "," << txy_x_axis
            << "," << txy_y_axis << "," << txy_axis << "," << std::max({ux_axis, vy_axis, tangential_ghost, txy_axis})
            << "," << steady_residual << "," << steady_hits << "\n";
        std::cout << "quarter_symmetry step=" << step << ", max_normal_axis=" << std::max(ux_axis, vy_axis)
                  << ", max_tangent_ghost_mismatch=" << tangential_ghost << ", max_tau_xy_axis=" << txy_axis
                  << std::endl;
    }
} // namespace

/**
 * Upper-right quarter of the cross channel.
 *
 * A2 is the upper-right quarter of the crossing, A3 is the upper half of the
 * right inlet, and A5 is the right half of the upper outlet.  XNegative faces
 * on A2/A5 impose x=0 symmetry; YNegative faces on A2/A3 impose y=0 symmetry.
 */
int main(int argc, char* argv[])
{
    CrossShapedChannel2DCase case_param(argc, argv);
    case_param.read_paras();

    double      Sc                 = 0.0;
    std::string scalar_scheme_name = "quick";
    IO::read_number(case_param.para_map, "Sc", Sc);
    IO::read_string(case_param.para_map, "scalar_scheme", scalar_scheme_name);
    const bool                 enable_scalar_transport = Sc > 0.0;
    const double               Pe                      = enable_scalar_transport ? case_param.Re * Sc : 0.0;
    const double               nr                      = enable_scalar_transport ? 1.0 / Pe : 0.0;
    const DifferenceSchemeType scalar_scheme           = parse_scalar_scheme(scalar_scheme_name);
    if (enable_scalar_transport && Pe <= 0.0)
        throw std::runtime_error("Scalar transport requires Re * Sc > 0.");

    configure_physics(case_param);
    PhysicsConfig&       physics_cfg = PhysicsConfig::Get();
    EnvironmentConfig&   env_cfg     = EnvironmentConfig::Get();
    TimeAdvancingConfig& time_cfg    = TimeAdvancingConfig::Get();
    env_cfg.showGmresRes             = true;
    env_cfg.showCurrentStep          = false;

    const double h        = case_param.h;
    const int    full_nx2 = static_cast<int>(std::lround(case_param.lx_2 / h));
    const int    full_ny2 = static_cast<int>(std::lround(case_param.ly_2 / h));
    if (full_nx2 <= 1 || full_ny2 <= 1 || full_nx2 % 2 != 0 || full_ny2 % 2 != 0)
        throw std::runtime_error("Quarter case requires even lx_2/h and ly_2/h, both greater than 1.");

    const int    nx2 = full_nx2 / 2;
    const int    ny2 = full_ny2 / 2;
    const int    nx3 = std::max(1, static_cast<int>(std::lround(case_param.lx_3 / h)));
    const int    ny5 = std::max(1, static_cast<int>(std::lround(case_param.ly_5 / h)));
    const double lx2 = 0.5 * case_param.lx_2;
    const double ly2 = 0.5 * case_param.ly_2;

    int    steady_stop_enabled   = 1;
    double steady_residual_tol   = 1.0e-8;
    int    steady_min_steps      = 10;
    int    steady_converged_hits = 5;
    IO::read_number(case_param.para_map, "steady_stop_enabled", steady_stop_enabled);
    IO::read_number(case_param.para_map, "steady_residual_tol", steady_residual_tol);
    IO::read_number(case_param.para_map, "steady_min_steps", steady_min_steps);
    IO::read_number(case_param.para_map, "steady_converged_hits", steady_converged_hits);
    if (steady_stop_enabled != 0 && steady_stop_enabled != 1)
        throw std::runtime_error("steady_stop_enabled must be 0 or 1.");
    if (!std::isfinite(steady_residual_tol) || steady_residual_tol <= 0.0 || steady_min_steps <= 0 ||
        steady_converged_hits <= 0)
        throw std::runtime_error("Invalid quarter-domain steady convergence parameters.");

    Geometry2D      geometry;
    Domain2DUniform A2(nx2, ny2, lx2, ly2, "A2");
    Domain2DUniform A3(nx3, ny2, case_param.lx_3, ly2, "A3");
    Domain2DUniform A5(nx2, ny5, lx2, case_param.ly_5, "A5");
    geometry.connect(&A2, LocationType::XPositive, &A3);
    geometry.connect(&A2, LocationType::YPositive, &A5);
    geometry.axis(&A2, LocationType::XNegative);
    geometry.axis(&A2, LocationType::YNegative);
    geometry.check();
    geometry.solve_prepare();

    Variable2D u("u"), v("v"), p("p"), vorticity("vorticity");
    u.set_geometry(geometry);
    v.set_geometry(geometry);
    p.set_geometry(geometry);
    vorticity.set_geometry(geometry);

    Variable2D phi("phi");
    const bool enable_mhd = std::abs(case_param.Ha) > 0.0;
    if (enable_mhd)
        phi.set_geometry(geometry);
    Variable2D c("c");
    if (enable_scalar_transport)
        c.set_geometry(geometry);

    Variable2D mu("mu"), tau_xx("tau_xx"), tau_yy("tau_yy"), tau_xy("tau_xy");
    mu.set_geometry(geometry);
    tau_xx.set_geometry(geometry);
    tau_yy.set_geometry(geometry);
    tau_xy.set_geometry(geometry);

    field2 u_A2, u_A3, u_A5, v_A2, v_A3, v_A5, p_A2, p_A3, p_A5, vort_A2, vort_A3, vort_A5;
    u.set_x_edge_field(&A2, u_A2);
    u.set_x_edge_field(&A3, u_A3);
    u.set_x_edge_field(&A5, u_A5);
    v.set_y_edge_field(&A2, v_A2);
    v.set_y_edge_field(&A3, v_A3);
    v.set_y_edge_field(&A5, v_A5);
    p.set_center_field(&A2, p_A2);
    p.set_center_field(&A3, p_A3);
    p.set_center_field(&A5, p_A5);
    vorticity.set_center_field(&A2, vort_A2);
    vorticity.set_center_field(&A3, vort_A3);
    vorticity.set_center_field(&A5, vort_A5);

    field2 phi_A2("phi_A2"), phi_A3("phi_A3"), phi_A5("phi_A5");
    if (enable_mhd)
    {
        phi.set_center_field(&A2, phi_A2);
        phi.set_center_field(&A3, phi_A3);
        phi.set_center_field(&A5, phi_A5);
    }
    field2 c_A2("c_A2"), c_A3("c_A3"), c_A5("c_A5");
    if (enable_scalar_transport)
    {
        c.set_center_field(&A2, c_A2);
        c.set_center_field(&A3, c_A3);
        c.set_center_field(&A5, c_A5);
    }

    field2 mu_A2("mu_A2"), mu_A3("mu_A3"), mu_A5("mu_A5");
    field2 txx_A2("txx_A2"), txx_A3("txx_A3"), txx_A5("txx_A5");
    field2 tyy_A2("tyy_A2"), tyy_A3("tyy_A3"), tyy_A5("tyy_A5");
    field2 txy_A2("txy_A2"), txy_A3("txy_A3"), txy_A5("txy_A5");
    mu.set_corner_field(&A2, mu_A2);
    mu.set_corner_field(&A3, mu_A3);
    mu.set_corner_field(&A5, mu_A5);
    tau_xx.set_center_field(&A2, txx_A2);
    tau_xx.set_center_field(&A3, txx_A3);
    tau_xx.set_center_field(&A5, txx_A5);
    tau_yy.set_center_field(&A2, tyy_A2);
    tau_yy.set_center_field(&A3, tyy_A3);
    tau_yy.set_center_field(&A5, tyy_A5);
    tau_xy.set_corner_field(&A2, txy_A2);
    tau_xy.set_corner_field(&A3, txy_A3);
    tau_xy.set_corner_field(&A5, txy_A5);

    const std::vector<Domain2DUniform*> domains   = {&A2, &A3, &A5};
    const std::vector<LocationType>     locations = {
        LocationType::XNegative, LocationType::XPositive, LocationType::YNegative, LocationType::YPositive};
    auto is_adjacent = [&](Domain2DUniform* domain, LocationType location) {
        return geometry.adjacency.count(domain) && geometry.adjacency.at(domain).count(location);
    };

    for (auto* domain : domains)
    {
        for (auto location : locations)
        {
            if (is_adjacent(domain, location))
                continue;
            set_dirichlet_zero(u, domain, location);
            set_dirichlet_zero(v, domain, location);
            set_neumann_zero(p, domain, location);
            if (enable_scalar_transport)
                set_neumann_zero(c, domain, location);
            if (enable_mhd)
                set_neumann_zero(phi, domain, location);
        }
    }

    // x=0 symmetry: normal u=0, tangential dv/dx=0.
    set_dirichlet_zero(u, &A2, LocationType::XNegative);
    set_neumann_zero(v, &A2, LocationType::XNegative);
    set_neumann_zero(p, &A2, LocationType::XNegative);
    set_dirichlet_zero(u, &A5, LocationType::XNegative);
    set_neumann_zero(v, &A5, LocationType::XNegative);
    set_neumann_zero(p, &A5, LocationType::XNegative);

    // y=0 symmetry: normal v=0, tangential du/dy=0.
    set_neumann_zero(u, &A2, LocationType::YNegative);
    set_dirichlet_zero(v, &A2, LocationType::YNegative);
    set_neumann_zero(p, &A2, LocationType::YNegative);
    set_neumann_zero(u, &A3, LocationType::YNegative);
    set_dirichlet_zero(v, &A3, LocationType::YNegative);
    set_neumann_zero(p, &A3, LocationType::YNegative);

    // Right inlet and upper outlet.
    u.set_boundary_type(&A3, LocationType::XPositive, PDEBoundaryType::Dirichlet);
    u.set_boundary_value(&A3, LocationType::XPositive, -1.0);
    set_dirichlet_zero(v, &A3, LocationType::XPositive);
    set_dirichlet_zero(p, &A5, LocationType::YPositive);
    set_neumann_zero(u, &A5, LocationType::YPositive);
    set_neumann_zero(v, &A5, LocationType::YPositive);

    if (enable_mhd)
    {
        phi.set_boundary_type(&A3, LocationType::XPositive, PDEBoundaryType::Dirichlet);
        phi.set_boundary_value(&A3, LocationType::XPositive, 0.0);
        phi.has_boundary_value_map[&A3][LocationType::XPositive] = true;
        phi.set_boundary_type(&A5, LocationType::YPositive, PDEBoundaryType::Dirichlet);
        phi.set_boundary_value(&A5, LocationType::YPositive, 0.0);
        phi.has_boundary_value_map[&A5][LocationType::YPositive] = true;
    }
    if (enable_scalar_transport)
    {
        c.set_boundary_type(&A3, LocationType::XPositive, PDEBoundaryType::Dirichlet);
        c.set_boundary_value(&A3, LocationType::XPositive, 1.0);
        c.has_boundary_value_map[&A3][LocationType::XPositive] = true;
        set_neumann_zero(c, &A5, LocationType::YPositive);
    }

    ConcatPoissonSolver2D p_solver(&p);
    ConcatNSSolver2D      ns_solver(&u, &v, &p, &p_solver);
    ns_solver.init_nonnewton(&mu, &tau_xx, &tau_yy, &tau_xy, enable_mhd ? &phi : nullptr);
    ns_solver.p_solver->set_parameter(case_param.gmres_m, case_param.gmres_tol, case_param.gmres_max_iter);
    std::unique_ptr<ScalarSolver2D> scalar_solver;
    if (enable_scalar_transport)
        scalar_solver = std::make_unique<ScalarSolver2D>(&u, &v, &c, nr, scalar_scheme);

    const double dt = select_dt(h, case_param.dt_factor, physics_cfg);
    if (!(dt > 0.0) || !std::isfinite(dt))
        throw std::runtime_error("Quarter case selected an invalid time step.");
    const int estimated_steps = std::max(1, static_cast<int>(std::ceil(case_param.T_total / dt)));
    time_cfg.dt               = dt;
    time_cfg.t_max            = case_param.T_total;
    time_cfg.num_iterations   = estimated_steps;
    case_param.max_step       = estimated_steps;

    const bool should_record_paras = case_param.record_paras();
    if (should_record_paras)
    {
        case_param.paras_record.record("quarter_domain", 1)
            .record("symmetry_x", std::string("XNegative:A2,A5"))
            .record("symmetry_y", std::string("YNegative:A2,A3"))
            .record("dt_selected", dt)
            .record("estimated_total_steps", estimated_steps)
            .record("steady_stop_enabled", steady_stop_enabled)
            .record("steady_residual_tol", steady_residual_tol)
            .record("steady_min_steps", steady_min_steps)
            .record("steady_converged_hits", steady_converged_hits)
            .record("scalar_transport_enabled", enable_scalar_transport ? 1 : 0)
            .record("Sc", Sc)
            .record("Pe", Pe)
            .record("nr", nr);
    }

    IO::create_directory(case_param.root_dir + "/final");
    std::ofstream symmetry_out(case_param.root_dir + "/quarter_symmetry.csv");
    if (!symmetry_out.is_open())
        throw std::runtime_error("Failed to open quarter_symmetry.csv.");
    symmetry_out << std::setprecision(16)
                 << "step,time,max_u_xneg_axis,max_v_yneg_axis,max_tangential_ghost_mismatch,"
                    "max_tau_xy_xneg_axis,max_tau_xy_yneg_axis,max_tau_xy_axis,max_symmetry_residual,steady_residual,"
                    "steady_hits\n";

    ns_solver.phys_boundary_update();
    ns_solver.nondiag_shared_boundary_update();
    ns_solver.diag_shared_boundary_update();
    if (scalar_solver)
    {
        scalar_solver->phys_boundary_update();
        scalar_solver->nondiag_shared_boundary_update();
    }

    std::cout << "Construct quarter-domain symmetric cross: A2=" << nx2 << "x" << ny2 << ", A3=" << nx3 << "x" << ny2
              << ", A5=" << nx2 << "x" << ny5 << ", total_cells=" << nx2 * ny2 + nx3 * ny2 + nx2 * ny5 << std::endl;
    std::unordered_map<Domain2DUniform*, field2> previous_u_fields;
    std::unordered_map<Domain2DUniform*, field2> previous_v_fields;
    if (steady_stop_enabled != 0)
    {
        initialize_previous_field_map(u, previous_u_fields, "quarter_previous_u");
        initialize_previous_field_map(v, previous_v_fields, "quarter_previous_v");
        std::cout << "Quarter steady-stop enabled: tol=" << steady_residual_tol << ", min_steps=" << steady_min_steps
                  << ", converged_hits=" << steady_converged_hits << std::endl;
    }
    double steady_residual = std::numeric_limits<double>::infinity();
    int    steady_hits     = 0;
    bool   steady_reached  = false;
    write_symmetry_row(symmetry_out, 0, 0.0, u, v, tau_xy, &A2, &A3, &A5, steady_residual, steady_hits);

    double current_time = 0.0;
    int    final_step   = 0;
    for (int step = 1; step <= estimated_steps; ++step)
    {
        const double remaining = case_param.T_total - current_time;
        if (remaining <= 128.0 * std::numeric_limits<double>::epsilon() * std::max(1.0, case_param.T_total))
            break;
        const double dt_step = std::min(dt, remaining);
        time_cfg.dt          = dt_step;
        ns_solver.setTimeStep(dt_step);
        if (scalar_solver)
            scalar_solver->setTimeStep(dt_step);
        ns_solver.solve_nonnewton();
        if (scalar_solver)
            scalar_solver->solve();
        current_time += dt_step;
        final_step = step;

        const bool finite = std::isfinite(u_A2(0, 0)) && std::isfinite(v_A2(0, 0)) && std::isfinite(p_A2(0, 0));
        if (!finite)
        {
            std::cerr << "Quarter-domain run diverged at step=" << step << std::endl;
            return -1;
        }
        ns_solver.phys_boundary_update();
        ns_solver.nondiag_shared_boundary_update();
        ns_solver.diag_shared_boundary_update();

        if (steady_stop_enabled != 0)
        {
            const double u_residual = compute_relative_field_update(u, previous_u_fields);
            const double v_residual = compute_relative_field_update(v, previous_v_fields);
            steady_residual         = std::max(u_residual, v_residual);
            if (step >= steady_min_steps && steady_residual < steady_residual_tol)
                ++steady_hits;
            else
                steady_hits = 0;

            for (auto* domain : u.geometry->domains)
            {
                previous_u_fields.at(domain) = *u.field_map.at(domain);
                previous_v_fields.at(domain) = *v.field_map.at(domain);
            }
            if (steady_hits >= steady_converged_hits)
            {
                steady_reached = true;
                std::cout << "Quarter steady state reached at step=" << step << ", residual=" << steady_residual
                          << std::endl;
            }
        }
        write_symmetry_row(symmetry_out, step, current_time, u, v, tau_xy, &A2, &A3, &A5, steady_residual, steady_hits);
        if (steady_reached)
            break;
    }

    ns_solver.phys_boundary_update();
    ns_solver.nondiag_shared_boundary_update();
    ns_solver.diag_shared_boundary_update();
    ns_solver.raw_vorticity_update(vorticity);
    IO::write_csv(u, case_param.root_dir + "/final/u_" + std::to_string(final_step));
    IO::write_csv(v, case_param.root_dir + "/final/v_" + std::to_string(final_step));
    IO::write_csv(p, case_param.root_dir + "/final/p_" + std::to_string(final_step));
    IO::write_csv(vorticity, case_param.root_dir + "/final/vorticity_" + std::to_string(final_step));
    IO::write_csv(mu, case_param.root_dir + "/final/mu_" + std::to_string(final_step));
    IO::write_csv(tau_xx, case_param.root_dir + "/final/tau_xx_" + std::to_string(final_step));
    IO::write_csv(tau_yy, case_param.root_dir + "/final/tau_yy_" + std::to_string(final_step));
    IO::write_csv(tau_xy, case_param.root_dir + "/final/tau_xy_" + std::to_string(final_step));
    if (enable_mhd)
        IO::write_csv(phi, case_param.root_dir + "/final/phi_" + std::to_string(final_step));
    if (enable_scalar_transport)
        IO::write_csv(c, case_param.root_dir + "/final/c_" + std::to_string(final_step));

    std::cout << "Quarter-domain symmetric run finished: step=" << final_step << ", time=" << current_time
              << ", steady_reached=" << (steady_reached ? 1 : 0) << ", steady_residual=" << steady_residual
              << std::endl;
    return 0;
}
