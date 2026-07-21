#include "boundary_2d_utils.h"
#include "mhd_module_2d_mac.h"
#include "ns_solver2d.h"

#include <algorithm>
#include <cmath>
#include <vector>

/** @brief Minimum shear rate threshold to prevent singularity in power-law model */
constexpr double GAMMA_DOT_MIN = 1.0e-4;
/** @brief Under-relaxation factor for viscosity update (0, 1], smaller means stronger damping */
constexpr double VISCOSITY_RELAX_ALPHA = 0.7;
/** @brief Spatial smoothing weights for viscosity field */
constexpr double VISCOSITY_SMOOTH_CENTER_WEIGHT   = 0.8;
constexpr double VISCOSITY_SMOOTH_NEIGHBOR_WEIGHT = 0.05;
/** @brief Disable spatial smoothing temporarily to isolate the remaining oscillation source. */
constexpr bool VISCOSITY_ENABLE_SPATIAL_SMOOTHING = false;

namespace
{
    using SharedNodeKey  = std::pair<long long, long long>;
    using CornerFieldMap = std::unordered_map<Domain2DUniform*, field2*>;

    struct SharedCornerFieldEntry
    {
        double*          value_ptr = nullptr;
        Domain2DUniform* domain    = nullptr;
        int              i         = 0;
        int              j         = 0;
    };

    struct SharedNodeKeyHash
    {
        std::size_t operator()(const SharedNodeKey& key) const
        {
            const std::size_t x_hash = std::hash<long long> {}(key.first);
            const std::size_t y_hash = std::hash<long long> {}(key.second);
            return x_hash ^ (y_hash << 1);
        }
    };


    /**
     * @brief Average duplicated corner-field values on all shared interfaces.
     */
    void shared_corner_field_average_update(
        const std::vector<Domain2DUniform*>& domains,
        const CornerFieldMap&                corner_field_map,
        const std::unordered_map<Domain2DUniform*, std::unordered_map<LocationType, PDEBoundaryType>>&
            boundary_type_map)
    {
        std::unordered_map<SharedNodeKey, std::vector<SharedCornerFieldEntry>, SharedNodeKeyHash> shared_node_map;

        for (auto& domain : domains)
        {
            field2& field = *corner_field_map.at(domain);

            int nx = domain->get_nx();
            int ny = domain->get_ny();

            const long long offset_x_idx =
                static_cast<long long>(std::llround(domain->get_offset_x() / domain->get_hx()));
            const long long offset_y_idx =
                static_cast<long long>(std::llround(domain->get_offset_y() / domain->get_hy()));

            std::vector<char> local_node_mark((nx + 1) * (ny + 1), 0);

            const auto& bound_type_map = boundary_type_map.at(domain);

            auto is_adjacented_boundary = [&](LocationType loc) {
                const auto type_it = bound_type_map.find(loc);
                return type_it != bound_type_map.end() && type_it->second == PDEBoundaryType::Adjacented;
            };

            auto add_shared_node = [&](int i, int j) {
                const int flat_idx = i * (ny + 1) + j;
                if (local_node_mark[flat_idx] != 0)
                    return;

                local_node_mark[flat_idx] = 1;
                shared_node_map[{offset_x_idx + i, offset_y_idx + j}].push_back({field.get_ptr(i, j), domain, i, j});
            };

            if (is_adjacented_boundary(LocationType::XNegative))
                for (int j = 0; j <= ny; j++)
                    add_shared_node(0, j);

            if (is_adjacented_boundary(LocationType::XPositive))
                for (int j = 0; j <= ny; j++)
                    add_shared_node(nx, j);

            if (is_adjacented_boundary(LocationType::YNegative))
                for (int i = 0; i <= nx; i++)
                    add_shared_node(i, 0);

            if (is_adjacented_boundary(LocationType::YPositive))
                for (int i = 0; i <= nx; i++)
                    add_shared_node(i, ny);
        }

        for (auto& entry : shared_node_map)
        {
            auto& node_values = entry.second;

            if (node_values.size() < 2)
                continue;

            double avg_val = 0.0;
            for (const auto& node_value : node_values)
                avg_val += *node_value.value_ptr;

            avg_val /= static_cast<double>(node_values.size());

            for (const auto& node_value : node_values)
                *node_value.value_ptr = avg_val;
        }
    }

    double calc_viscosity_by_model(double gamma_dot, const PhysicsConfig& physics_cfg)
    {
        double mu_val    = physics_cfg.nu;
        double gamma_use = gamma_dot;

        // Recover physical shear rate only for dimensionless viscosity mode.
        if (physics_cfg.use_dimensionless_viscosity)
            gamma_use *= physics_cfg.gamma_ref;

        if (physics_cfg.model_type == 1) // Power Law
        {
            if (physics_cfg.n < 1.0)
                gamma_use = std::max(gamma_use, GAMMA_DOT_MIN); // protect against zero when exponent negative

            mu_val = physics_cfg.k * std::pow(gamma_use, physics_cfg.n - 1.0);

            // Enforce limits
            mu_val = std::max(physics_cfg.mu_min, std::min(mu_val, physics_cfg.mu_max));
        }
        else if (physics_cfg.model_type == 2) // Carreau
        {
            mu_val = physics_cfg.mu_inf + (physics_cfg.mu_0 - physics_cfg.mu_inf) *
                                              std::pow(1.0 + std::pow(physics_cfg.lambda * gamma_use, physics_cfg.a),
                                                       (physics_cfg.n - 1.0) / physics_cfg.a);

            // Enforce limits
            mu_val = std::max(physics_cfg.mu_min, std::min(mu_val, physics_cfg.mu_max));
        }
        else if (physics_cfg.model_type == 3) // Casson
        {
            const double gamma_safe = std::max(gamma_use, GAMMA_DOT_MIN);
            const double sqrt_mu    = std::sqrt(std::max(physics_cfg.casson_mu, 0.0));
            const double sqrt_term  = std::sqrt(std::max(physics_cfg.casson_tau0, 0.0) / gamma_safe);
            mu_val                  = (sqrt_mu + sqrt_term) * (sqrt_mu + sqrt_term);

            // Enforce limits
            mu_val = std::max(physics_cfg.mu_min, std::min(mu_val, physics_cfg.mu_max));
        }

        if (physics_cfg.use_dimensionless_viscosity)
        {
            mu_val *= 1.0 / (physics_cfg.mu_ref * physics_cfg.Re);
        }

        return mu_val;
    }
} // namespace

void ConcatNSSolver2D::init_nonnewton(Variable2D* in_mu_var,
                                      Variable2D* in_tau_xx_var,
                                      Variable2D* in_tau_yy_var,
                                      Variable2D* in_tau_xy_var,
                                      Variable2D* in_phi_var)
{
    mu_var     = in_mu_var;
    tau_xx_var = in_tau_xx_var;
    tau_yy_var = in_tau_yy_var;
    tau_xy_var = in_tau_xy_var;

    mu_field_map     = mu_var->field_map;
    tau_xx_field_map = tau_xx_var->field_map;
    tau_yy_field_map = tau_yy_var->field_map;
    tau_xy_field_map = tau_xy_var->field_map;

    tau_xx_buffer_map = tau_xx_var->buffer_map;
    tau_yy_buffer_map = tau_yy_var->buffer_map;

    PhysicsConfig& physics_cfg = PhysicsConfig::Get();
    if (physics_cfg.enable_mhd)
        init_mhd(in_phi_var);
}

void ConcatNSSolver2D::solve_nonnewton()
{
    PhysicsConfig& physics_cfg = PhysicsConfig::Get();

    // 2. Calculate Viscosity (mu) based on current velocity field
    viscosity_update();

    // The corner-based viscosity field is duplicated on shared interfaces.
    // Synchronize those duplicated nodes before reconstructing the stress tensor.
    mu_shared_boundary_field_update();

    // 3. Calculate Stress Tensor (tau) based on velocity and viscosity
    stress_update();

    // tau_xy is also stored as a duplicated corner field on shared interfaces.
    // Synchronize those nodes before using tau in the momentum predictor.
    tau_xy_shared_boundary_field_update();

    // 3.1 YPositivedate Stress Buffers (tau_xx, tau_yy)
    stress_buffer_update();

    // 4. Solve Momentum Equation (Predictor Step)
    // Replaces euler_conv_diff_inner/outer with Non-Newtonian versions
    euler_conv_diff_inner_nonnewton();
    euler_conv_diff_outer_nonnewton();

    // MHD: predictor step finished, before div(u) boundary update
    if (physics_cfg.enable_mhd && mhd_module)
    {
        // The predictor swaps u/v field storage. Rebuild every physical,
        // shared, and diagonal velocity boundary before reconstructing current.
        phys_boundary_update();
        nondiag_shared_boundary_update();
        diag_shared_boundary_update();

        mhd_module->solveElectricPotential();
        mhd_module->updateCurrentDensity();
        mhd_module->applyLorentzForce();
    }

    // 5. YPositivedate boundary for divu (Prepare for Pressure Projection)
    phys_boundary_update();
    nondiag_shared_boundary_update();

    // 6. Pressure Correction Loop (Projection Method)
    for (int it = 0; it < corr_it; it++)
    {
        // Calculate Velocity Divergence
        velocity_div_inner();
        velocity_div_outer();

        // Solve Poisson Equation for Pressure
        normalize_pressure();
        p_solver->solve();

        // YPositivedate Pressure Boundaries
        pressure_buffer_update();

        // Correct Velocity with Pressure Gradient
        add_pressure_gradient();
    }

    // update boundary at last to ensure other solver get xpos value at boundary
    phys_boundary_update();
    nondiag_shared_boundary_update();
    diag_shared_boundary_update();
}

void ConcatNSSolver2D::viscosity_update()
{
    PhysicsConfig& physics_cfg = PhysicsConfig::Get();

    for (auto& domain : domains)
    {
        field2& u  = *u_field_map[domain];
        field2& v  = *v_field_map[domain];
        field2& mu = *mu_field_map[domain];

        int    nx = u.get_nx();
        int    ny = u.get_ny();
        double hx = domain->hx;
        double hy = domain->hy;

        const bool enable_spatial_smoothing = VISCOSITY_ENABLE_SPATIAL_SMOOTHING;

        const int mu_ny  = ny + 1;
        auto      mu_idx = [mu_ny](int i_idx, int j_idx) { return i_idx * mu_ny + j_idx; };

        std::vector<double> mu_relaxed((nx + 1) * (ny + 1), 0.0);
        std::vector<int>    mu_valid((nx + 1) * (ny + 1), 0);

        double* u_xneg_buffer = u_buffer_map[domain][LocationType::XNegative];
        double* u_xpos_buffer = u_buffer_map[domain][LocationType::XPositive];
        double* u_yneg_buffer = u_buffer_map[domain][LocationType::YNegative];
        double* u_ypos_buffer = u_buffer_map[domain][LocationType::YPositive];

        double* v_xneg_buffer = v_buffer_map[domain][LocationType::XNegative];
        double* v_xpos_buffer = v_buffer_map[domain][LocationType::XPositive];
        double* v_yneg_buffer = v_buffer_map[domain][LocationType::YNegative];
        double* v_ypos_buffer = v_buffer_map[domain][LocationType::YPositive];

        // Helper to get u at (i, j) handling boundaries
        // u is defined at (i, j+0.5) for i in [0, nx], j in [-1, ny]
        auto get_u = [&](int i, int j) -> double {
            return get_u_with_boundary(i,
                                       j,
                                       nx,
                                       ny,
                                       u,
                                       u_xneg_buffer,
                                       u_xpos_buffer,
                                       u_yneg_buffer,
                                       u_ypos_buffer,
                                       xpos_yneg_corner_map[domain]);
        };

        // Helper to get v at (i, j) handling boundaries
        // v is defined at (i+0.5, j) for i in [-1, nx], j in [0, ny]
        auto get_v = [&](int i, int j) -> double {
            return get_v_with_boundary(i,
                                       j,
                                       nx,
                                       ny,
                                       v,
                                       v_xneg_buffer,
                                       v_xpos_buffer,
                                       v_yneg_buffer,
                                       v_ypos_buffer,
                                       xneg_ypos_corner_map[domain]);
        };

        // Helper lambda for du/dx at (i, j_row) where u is defined
        auto calc_du_dx_row = [&](int i_idx, int j_idx) -> double {
            if (i_idx > 0 && i_idx < nx)
                return (get_u(i_idx + 1, j_idx) - get_u(i_idx - 1, j_idx)) / (2.0 * hx);
            else if (i_idx == 0)
                return (-3 * get_u(0, j_idx) + 4 * get_u(1, j_idx) - get_u(2, j_idx)) /
                       (2.0 * hx); // Forward difference at 2 order accuaracy
            else                   // i_idx == nx
                return (3 * get_u(nx, j_idx) - 4 * get_u(nx - 1, j_idx) + get_u(nx - 2, j_idx)) /
                       (2.0 * hx); // Backward difference at 2 order accuaracy
        };

        // Helper lambda for dv/dy at (i_col, j) where v is defined
        auto calc_dv_dy_col = [&](int i_idx, int j_idx) -> double {
            if (j_idx > 0 && j_idx < ny)
                return (get_v(i_idx, j_idx + 1) - get_v(i_idx, j_idx - 1)) / (2.0 * hy);
            else if (j_idx == 0)
                return (-3 * get_v(i_idx, 0) + 4 * get_v(i_idx, 1) - get_v(i_idx, 2)) /
                       (2.0 * hy); // Forward difference at 2 order accuaracy
            else                   // j_idx == ny
                return (3 * get_v(i_idx, ny) - 4 * get_v(i_idx, ny - 1) + get_v(i_idx, ny - 2)) /
                       (2.0 * hy); // Backward difference at 2 order accuaracy
        };

        // First pass: calculate viscosity at Nodes (nx+1, ny+1), exclude (nx, ny)
        // Apply temporal under-relaxation and cache to temporary array.
        OPENMP_PARALLEL_FOR()
        for (int i = 0; i <= nx; i++)
        {
            int j_end = (i == nx) ? ny : ny + 1;
            for (int j = 0; j < j_end; j++)
            {
                // 1. Calculate du_dy and dv_dx at Node (i, j)
                // Node (i, j) is at x=i*hx, y=j*hy

                // du_dy approx (u(i, j) - u(i, j-1)) / hy
                // u(i, j) is at y=j+0.5, u(i, j-1) is at y=j-0.5. Midpoint is y=j. Correct.
                double val_u_j   = get_u(i, j);     // u(i, j)
                double val_u_jm1 = get_u(i, j - 1); // u(i, j-1)
                double du_dy     = (val_u_j - val_u_jm1) / hy;

                // dv_dx approx (v(i, j) - v(i-1, j)) / hx
                // v(i, j) is at x=i+0.5, v(i-1, j) is at x=i-0.5. Midpoint is x=i. Correct.
                double val_v_i   = get_v(i, j);     // v(i, j)
                double val_v_im1 = get_v(i - 1, j); // v(i-1, j)
                double dv_dx     = (val_v_i - val_v_im1) / hx;

                // // TODO 可以在不改变buffer的情况下，主动根据边界类型及其值计算更精确的边界导数，暂时先不考虑
                // 2. Calculate du_dx and dv_dy at Node (i, j)
                // Using central differencing for interior nodes
                // Using one-sided differencing for boundary nodes
                double du_dx = 0.5 * (calc_du_dx_row(i, j) + calc_du_dx_row(i, j - 1));
                double dv_dy = 0.5 * (calc_dv_dy_col(i, j) + calc_dv_dy_col(i - 1, j));

                // 3. Calculate Shear Rate GammaDot
                // gamma_dot = sqrt( 2*(du_dx^2 + dv_dy^2) + (du_dy + dv_dx)^2 )
                double gamma_dot = std::sqrt(2.0 * (du_dx * du_dx + dv_dy * dv_dy) + (du_dy + dv_dx) * (du_dy + dv_dx));

                // 4. Update Viscosity with under-relaxation to suppress oscillations
                double mu_new = calc_viscosity_by_model(gamma_dot, physics_cfg);

                // Skip relaxation for first initialization-like states to avoid delaying startup.
                double mu_old         = mu(i, j);
                double mu_relaxed_val = (mu_old <= 0.0) ?
                                            mu_new :
                                            (VISCOSITY_RELAX_ALPHA * mu_new + (1.0 - VISCOSITY_RELAX_ALPHA) * mu_old);

                int idx         = mu_idx(i, j);
                mu_relaxed[idx] = mu_relaxed_val;
                mu_valid[idx]   = 1;
            }
        }

        // Second pass: spatial smoothing (interior nodes only)
        // mu_smooth = 0.8 * mu_center + 0.05 * (mu_left + mu_right + mu_down + mu_up)
        OPENMP_PARALLEL_FOR()
        for (int i = 0; i <= nx; i++)
        {
            int j_end = (i == nx) ? ny : ny + 1;
            for (int j = 0; j < j_end; j++)
            {
                int    center_idx = mu_idx(i, j);
                double mu_center  = mu_relaxed[center_idx];

                if (!enable_spatial_smoothing || i == 0 || i == nx || j == 0 || j == ny)
                {
                    // Keep boundary nodes unsmoothed.
                    mu(i, j) = mu_center;
                    continue;
                }

                // Interior nodes: all 4 neighbors are within [0,nx]x[0,ny] and are valid.
                const double mu_left  = mu_relaxed[mu_idx(i - 1, j)];
                const double mu_right = mu_relaxed[mu_idx(i + 1, j)];
                const double mu_down  = mu_relaxed[mu_idx(i, j - 1)];
                const double mu_up    = mu_relaxed[mu_idx(i, j + 1)];

                const double mu_smooth = VISCOSITY_SMOOTH_CENTER_WEIGHT * mu_center +
                                         VISCOSITY_SMOOTH_NEIGHBOR_WEIGHT * (mu_left + mu_right + mu_down + mu_up);

                mu(i, j) = mu_smooth;
            }
        }
    }
}

void ConcatNSSolver2D::mu_shared_boundary_field_update()
{
    if (mu_var == nullptr)
        throw std::runtime_error("ConcatNSSolver2D::mu_shared_boundary_field_update: mu_var is null");
    if (mu_var->position_type != VariablePositionType::Corner)
        throw std::runtime_error("ConcatNSSolver2D::mu_shared_boundary_field_update: mu must be a corner field");

    // mu is stored redundantly on corner nodes along shared interfaces. Collapse
    // those duplicates before stress reconstruction so every physical node has one value.
    shared_corner_field_average_update(domains, mu_field_map, u_var->boundary_type_map);
}

void ConcatNSSolver2D::tau_xy_shared_boundary_field_update()
{
    if (tau_xy_var == nullptr)
        throw std::runtime_error("ConcatNSSolver2D::tau_xy_shared_boundary_field_update: tau_xy_var is null");
    if (tau_xy_var->position_type != VariablePositionType::Corner)
        throw std::runtime_error(
            "ConcatNSSolver2D::tau_xy_shared_boundary_field_update: tau_xy must be a corner field");

    // tau_xy uses the same corner-node layout as mu, so shared-interface nodes
    // must also be averaged before they enter the predictor div(tau) stencil.
    shared_corner_field_average_update(domains, tau_xy_field_map, u_var->boundary_type_map);
}

void ConcatNSSolver2D::stress_buffer_update()
{
    PhysicsConfig& physics_cfg = PhysicsConfig::Get();

    for (auto& domain : domains)
    {
        field2& u = *u_field_map[domain];
        field2& v = *v_field_map[domain];

        int    nx = u.get_nx();
        int    ny = u.get_ny();
        double hx = domain->hx;
        double hy = domain->hy;

        double* u_xneg_buffer = u_buffer_map[domain][LocationType::XNegative];
        double* v_xneg_buffer = v_buffer_map[domain][LocationType::XNegative];

        double* u_yneg_buffer = u_buffer_map[domain][LocationType::YNegative];
        double* v_yneg_buffer = v_buffer_map[domain][LocationType::YNegative];

        double* tau_xx_xneg_buffer = tau_xx_buffer_map[domain][LocationType::XNegative];
        double* tau_yy_yneg_buffer = tau_yy_buffer_map[domain][LocationType::YNegative];

        // Helper to get u at (i, j) handling boundaries
        auto get_u = [&](int i_idx, int j_idx) {
            return get_u_with_boundary(i_idx,
                                       j_idx,
                                       nx,
                                       ny,
                                       u,
                                       u_xneg_buffer,
                                       u_buffer_map[domain][LocationType::XPositive],
                                       u_yneg_buffer,
                                       u_buffer_map[domain][LocationType::YPositive],
                                       xpos_yneg_corner_map[domain]);
        };

        // Helper to get v at (i, j) handling boundaries
        auto get_v = [&](int i_idx, int j_idx) {
            return get_v_with_boundary(i_idx,
                                       j_idx,
                                       nx,
                                       ny,
                                       v,
                                       v_xneg_buffer,
                                       v_buffer_map[domain][LocationType::XPositive],
                                       v_yneg_buffer,
                                       v_buffer_map[domain][LocationType::YPositive],
                                       xneg_ypos_corner_map[domain]);
        };

        // Fill the ghost-cell normal stresses used by the outer momentum stencil.
        // Shared interfaces reuse the neighboring domain's center stress directly;
        // physical boundaries still reconstruct ghost stress from the local field.

        // 1. Update tau_xx xneg buffer (at ghost cell -1, j)
        if (u_var->boundary_type_map[domain][LocationType::XNegative] == PDEBoundaryType::Adjacented)
        {
            Domain2DUniform* adj_domain = adjacency[domain][LocationType::XNegative];
            field2&          adj_tau_xx = *tau_xx_field_map[adj_domain];
            int              adj_nx     = adj_domain->get_nx();

            copy_x_to_buffer(tau_xx_xneg_buffer, adj_tau_xx, adj_nx - 1);
        }
        else
        {
            for (int j = 0; j < ny; j++)
            {
                // Calculate tau_xx at ghost cell (-1, j)
                // 1. du/dx at (-0.5, j+0.5)
                // u(0, j) is at x=0, u_xneg_buffer[j] is at x=-1
                double du_dx_ghost = (u(0, j) - u_xneg_buffer[j]) / hx;

                // 2. dv/dy at (-0.5, j+0.5)
                // v_xneg_buffer is at x=-0.5. Need dy.
                // v_xneg_buffer[j] is at y=j, v_xneg_buffer[j+1] is at y=j+1
                // Use get_v to handle corner cases safely
                double v_xneg_j    = get_v(-1, j);
                double v_xneg_jp1  = get_v(-1, j + 1);
                double dv_dy_ghost = (v_xneg_jp1 - v_xneg_j) / hy;

                // 3. du/dy at (-0.5, j+0.5)
                // Average of du/dy at x=-1 (u_xneg_buffer) and x=0 (u(0, j))
                auto get_du_dy_col = [&](int col_idx) {
                    if (j == 0)
                    {
                        if (ny >= 3)
                            return (-3.0 * get_u(col_idx, 0) + 4.0 * get_u(col_idx, 1) - get_u(col_idx, 2)) /
                                   (2.0 * hy);
                        else
                            return (get_u(col_idx, 1) - get_u(col_idx, 0)) / hy;
                    }
                    else if (j == ny - 1)
                    {
                        if (ny >= 3)
                            return (3.0 * get_u(col_idx, ny - 1) - 4.0 * get_u(col_idx, ny - 2) +
                                    get_u(col_idx, ny - 3)) /
                                   (2.0 * hy);
                        else
                            return (get_u(col_idx, ny - 1) - get_u(col_idx, ny - 2)) / hy;
                    }
                    else
                    {
                        return (get_u(col_idx, j + 1) - get_u(col_idx, j - 1)) / (2.0 * hy);
                    }
                };

                double du_dy_xneg  = get_du_dy_col(-1);
                double du_dy_0     = get_du_dy_col(0);
                double du_dy_ghost = 0.5 * (du_dy_xneg + du_dy_0);

                // 4. dv/dx at (-0.5, j+0.5)
                // Need dv/dx at x=-0.5.
                // Points: x=-0.5 (v_xneg), x=0.5 (v0), x=1.5 (v1)
                auto get_dv_dx_node = [&](int k) {
                    double v_l = get_v(-1, k);
                    double v_0 = get_v(0, k);
                    if (nx >= 2)
                    {
                        double v_1 = get_v(1, k);
                        return (-3.0 * v_l + 4.0 * v_0 - v_1) / (2.0 * hx);
                    }
                    else
                    {
                        return (v_0 - v_l) / hx;
                    }
                };
                double dv_dx_ghost = 0.5 * (get_dv_dx_node(j) + get_dv_dx_node(j + 1));

                // 5. Gamma Dot
                double gamma_dot = std::sqrt(2.0 * (du_dx_ghost * du_dx_ghost + dv_dy_ghost * dv_dy_ghost) +
                                             (du_dy_ghost + dv_dx_ghost) * (du_dy_ghost + dv_dx_ghost));

                // 6. Viscosity
                double mu_val = calc_viscosity_by_model(gamma_dot, physics_cfg);

                tau_xx_xneg_buffer[j] = 2.0 * mu_val * du_dx_ghost;
            }
        }

        // 2. Update tau_yy yneg buffer (at ghost cell i, -1)
        if (u_var->boundary_type_map[domain][LocationType::YNegative] == PDEBoundaryType::Adjacented)
        {
            Domain2DUniform* adj_domain = adjacency[domain][LocationType::YNegative];
            field2&          adj_tau_yy = *tau_yy_field_map[adj_domain];
            int              adj_ny     = adj_domain->get_ny();

            copy_y_to_buffer(tau_yy_yneg_buffer, adj_tau_yy, adj_ny - 1);
        }
        else
        {
            for (int i = 0; i < nx; i++)
            {
                // Calculate tau_yy at ghost cell (i, -1)
                // 1. dv/dy at (i+0.5, -0.5)
                // v(i, 0) is at y=0, v_yneg_buffer[i] is at y=-1
                double dv_dy_ghost = (v(i, 0) - v_yneg_buffer[i]) / hy;

                // 2. du/dx at (i+0.5, -0.5)
                // u_yneg_buffer is at y=-0.5. Need dx.
                // u_yneg_buffer[i] is at x=i, u_yneg_buffer[i+1] is at x=i+1
                // Use get_u to handle corner cases safely
                double u_yneg_i    = get_u(i, -1);
                double u_yneg_ip1  = get_u(i + 1, -1);
                double du_dx_ghost = (u_yneg_ip1 - u_yneg_i) / hx;

                // 3. dv/dx at (i+0.5, -0.5)
                // Average of dv/dx at y=-1 (v_yneg_buffer) and y=0 (v(i, 0))
                auto get_dv_dx_row = [&](int row_idx) {
                    if (i == 0)
                    {
                        if (nx >= 3)
                            return (-3.0 * get_v(0, row_idx) + 4.0 * get_v(1, row_idx) - get_v(2, row_idx)) /
                                   (2.0 * hx);
                        else
                            return (get_v(1, row_idx) - get_v(0, row_idx)) / hx;
                    }
                    else if (i == nx - 1)
                    {
                        if (nx >= 3)
                            return (3.0 * get_v(nx - 1, row_idx) - 4.0 * get_v(nx - 2, row_idx) +
                                    get_v(nx - 3, row_idx)) /
                                   (2.0 * hx);
                        else
                            return (get_v(nx - 1, row_idx) - get_v(nx - 2, row_idx)) / hx;
                    }
                    else
                    {
                        return (get_v(i + 1, row_idx) - get_v(i - 1, row_idx)) / (2.0 * hx);
                    }
                };

                double dv_dx_yneg  = get_dv_dx_row(-1);
                double dv_dx_0     = get_dv_dx_row(0);
                double dv_dx_ghost = 0.5 * (dv_dx_yneg + dv_dx_0);

                // 4. du/dy at (i+0.5, -0.5)
                // Need du/dy at y=-0.5.
                // Points: y=-0.5 (u_yneg), y=0.5 (u0), y=1.5 (u1)
                auto get_du_dy_node = [&](int k) {
                    double u_d = get_u(k, -1);
                    double u_0 = get_u(k, 0);
                    if (ny >= 2)
                    {
                        double u_1 = get_u(k, 1);
                        return (-3.0 * u_d + 4.0 * u_0 - u_1) / (2.0 * hy);
                    }
                    else
                    {
                        return (u_0 - u_d) / hy;
                    }
                };
                double du_dy_ghost = 0.5 * (get_du_dy_node(i) + get_du_dy_node(i + 1));

                // 5. Gamma Dot
                double gamma_dot = std::sqrt(2.0 * (du_dx_ghost * du_dx_ghost + dv_dy_ghost * dv_dy_ghost) +
                                             (du_dy_ghost + dv_dx_ghost) * (du_dy_ghost + dv_dx_ghost));

                // 6. Viscosity
                double mu_val = calc_viscosity_by_model(gamma_dot, physics_cfg);

                tau_yy_yneg_buffer[i] = 2.0 * mu_val * dv_dy_ghost;
            }
        }
    }
}

void ConcatNSSolver2D::stress_update()
{
    for (auto& domain : domains)
    {
        field2& u      = *u_field_map[domain];
        field2& v      = *v_field_map[domain];
        field2& mu     = *mu_field_map[domain];
        field2& tau_xx = *tau_xx_field_map[domain];
        field2& tau_yy = *tau_yy_field_map[domain];
        field2& tau_xy = *tau_xy_field_map[domain];

        int    nx = u.get_nx();
        int    ny = u.get_ny();
        double hx = domain->hx;
        double hy = domain->hy;

        double* u_xneg_buffer = u_buffer_map[domain][LocationType::XNegative];
        double* u_xpos_buffer = u_buffer_map[domain][LocationType::XPositive];
        double* u_yneg_buffer = u_buffer_map[domain][LocationType::YNegative];
        double* u_ypos_buffer = u_buffer_map[domain][LocationType::YPositive];

        double* v_xneg_buffer = v_buffer_map[domain][LocationType::XNegative];
        double* v_xpos_buffer = v_buffer_map[domain][LocationType::XPositive];
        double* v_yneg_buffer = v_buffer_map[domain][LocationType::YNegative];
        double* v_ypos_buffer = v_buffer_map[domain][LocationType::YPositive];

        // Helper to get u at (i, j) handling boundaries for tau_xy calculation
        auto get_u = [&](int i, int j) -> double {
            return get_u_with_boundary(i,
                                       j,
                                       nx,
                                       ny,
                                       u,
                                       u_xneg_buffer,
                                       u_xpos_buffer,
                                       u_yneg_buffer,
                                       u_ypos_buffer,
                                       xpos_yneg_corner_map[domain]);
        };

        // Helper to get v at (i, j) handling boundaries for tau_xy calculation
        auto get_v = [&](int i, int j) -> double {
            return get_v_with_boundary(i,
                                       j,
                                       nx,
                                       ny,
                                       v,
                                       v_xneg_buffer,
                                       v_xpos_buffer,
                                       v_yneg_buffer,
                                       v_ypos_buffer,
                                       xneg_ypos_corner_map[domain]);
        };

        // 1. Calculate Normal Stresses (tau_xx, tau_yy) at Centers
        // Loop over cells (i, j)
        OPENMP_PARALLEL_FOR()
        for (int i = 0; i < nx; i++)
        {
            for (int j = 0; j < ny; j++)
            {
                // Calculate mu at center by averaging 4 corners
                // Corners of cell (i, j) are (i, j), (i+1, j), (i, j+1), (i+1, j+1)
                double mu_cen = (i == nx - 1 && j == ny - 1) ?
                                    0.5 * (mu(i + 1, j) + mu(i, j + 1)) : // Only two corners available
                                    0.25 * (mu(i, j) + mu(i + 1, j) + mu(i, j + 1) + mu(i + 1, j + 1));

                // du/dx at center
                double du_dx = (get_u(i + 1, j) - u(i, j)) / hx;

                // dv/dy at center
                double dv_dy = (get_v(i, j + 1) - v(i, j)) / hy;

                tau_xx(i, j) = 2.0 * mu_cen * du_dx;
                tau_yy(i, j) = 2.0 * mu_cen * dv_dy;
            }
        }

        // 2. Calculate Shear Stress (tau_xy) at Nodes
        // Loop over nodes (i, j) from 0 to nx, 0 to ny
        OPENMP_PARALLEL_FOR()
        for (int i = 0; i <= nx; i++)
        {
            int j_end = (i == nx) ? ny : ny + 1;
            for (int j = 0; j < j_end; j++)
            {
                // mu is already at nodes
                double mu_node = mu(i, j);

                // du/dy at node (i, j)
                // u(i, j) is at y=j+0.5, u(i, j-1) is at y=j-0.5
                double val_u_j   = get_u(i, j);
                double val_u_jm1 = get_u(i, j - 1);
                double du_dy     = (val_u_j - val_u_jm1) / hy;

                // dv/dx at node (i, j)
                // v(i, j) is at x=i+0.5, v(i-1, j) is at x=i-0.5
                double val_v_i   = get_v(i, j);
                double val_v_im1 = get_v(i - 1, j);
                double dv_dx     = (val_v_i - val_v_im1) / hx;

                tau_xy(i, j) = mu_node * (du_dy + dv_dx);
            }
        }
    }
}

void ConcatNSSolver2D::euler_conv_diff_inner_nonnewton()
{
    for (auto& domain : domains)
    {
        field2& u      = *u_field_map[domain];
        field2& v      = *v_field_map[domain];
        field2& p      = *p_field_map[domain];
        field2& tau_xx = *tau_xx_field_map[domain];
        field2& tau_yy = *tau_yy_field_map[domain];
        field2& tau_xy = *tau_xy_field_map[domain];

        field2& u_temp = *u_temp_field_map[domain];
        field2& v_temp = *v_temp_field_map[domain];

        int    nx = u.get_nx();
        int    ny = u.get_ny();
        double hx = domain->hx;
        double hy = domain->hy;

        // u (interior only; boundaries handled in euler_conv_diff_outer)
        OPENMP_PARALLEL_FOR()
        for (int i = 1; i < nx - 1; i++)
        {
            for (int j = 1; j < ny - 1; j++)
            {
                double conv_x =
                    u(i + 1, j) * (u(i + 1, j) + 2.0 * u(i, j)) - u(i - 1, j) * (u(i - 1, j) + 2.0 * u(i, j));
                double conv_y = (u(i, j) + u(i, j + 1)) * (v(i - 1, j + 1) + v(i, j + 1)) -
                                (u(i, j - 1) + u(i, j)) * (v(i - 1, j) + v(i, j));

                // Non-Newtonian Diffusion: div(tau)
                // For u-momentum: d(tau_xx)/dx + d(tau_xy)/dy
                // u(i, j) is at xneg face of cell (i, j)
                // tau_xx is at center. d(tau_xx)/dx approx (tau_xx(i, j) - tau_xx(i-1, j)) / hx
                // tau_xy is at node. d(tau_xy)/dy approx (tau_xy(i, j+1) - tau_xy(i, j)) / hy
                // Note: tau_xy(i, j) is at node (i, j) which is yneg-xneg of cell (i, j)
                //       tau_xy(i, j+1) is at node (i, j+1) which is ypos-xneg of cell (i, j)
                //       So this matches the location of u(i, j) which is xneg face center.

                double diff_x = (tau_xx(i, j) - tau_xx(i - 1, j)) / hx;
                double diff_y = (tau_xy(i, j + 1) - tau_xy(i, j)) / hy;
                double diff   = diff_x + diff_y;

                //? if 1/Re needed in non-newtonian case?
                u_temp(i, j) = u(i, j) - dt * (0.25 / hx * conv_x + 0.25 / hy * conv_y - diff);
            }
        }

        // v
        OPENMP_PARALLEL_FOR()
        for (int i = 1; i < nx - 1; i++)
        {
            for (int j = 1; j < ny - 1; j++)
            {
                double conv_x = (v(i, j) + v(i + 1, j)) * (u(i + 1, j - 1) + u(i + 1, j)) -
                                (v(i - 1, j) + v(i, j)) * (u(i, j - 1) + u(i, j));
                double conv_y =
                    v(i, j + 1) * (v(i, j + 1) + 2.0 * v(i, j)) - v(i, j - 1) * (v(i, j - 1) + 2.0 * v(i, j));

                // Non-Newtonian Diffusion: div(tau)
                // For v-momentum: d(tau_xy)/dx + d(tau_yy)/dy
                // v(i, j) is at yneg face of cell (i, j)
                // tau_xy is at node. d(tau_xy)/dx approx (tau_xy(i+1, j) - tau_xy(i, j)) / hx
                // tau_yy is at center. d(tau_yy)/dy approx (tau_yy(i, j) - tau_yy(i, j-1)) / hy
                // Note: tau_xy(i, j) is at node (i, j) which is yneg-xneg of cell (i, j)
                //       tau_xy(i+1, j) is at node (i+1, j) which is yneg-xpos of cell (i, j)
                //       So this matches the location of v(i, j) which is yneg face center.

                double diff_x = (tau_xy(i + 1, j) - tau_xy(i, j)) / hx;
                double diff_y = (tau_yy(i, j) - tau_yy(i, j - 1)) / hy;
                double diff   = diff_x + diff_y;

                v_temp(i, j) = v(i, j) - dt * (0.25 / hx * conv_x + 0.25 / hy * conv_y - diff);
            }
        }
    }
}

void ConcatNSSolver2D::euler_conv_diff_outer_nonnewton()
{
    for (auto& domain : domains)
    {
        field2& u      = *u_field_map[domain];
        field2& v      = *v_field_map[domain];
        field2& u_temp = *u_temp_field_map[domain];
        field2& v_temp = *v_temp_field_map[domain];
        field2& tau_xx = *tau_xx_field_map[domain];
        field2& tau_yy = *tau_yy_field_map[domain];
        field2& tau_xy = *tau_xy_field_map[domain];

        double hx = domain->hx;
        double hy = domain->hy;

        double* v_xneg_buffer = v_buffer_map[domain][LocationType::XNegative];
        double* v_xpos_buffer = v_buffer_map[domain][LocationType::XPositive];
        double* v_yneg_buffer = v_buffer_map[domain][LocationType::YNegative];
        double* v_ypos_buffer = v_buffer_map[domain][LocationType::YPositive];

        double* u_xneg_buffer = u_buffer_map[domain][LocationType::XNegative];
        double* u_xpos_buffer = u_buffer_map[domain][LocationType::XPositive];
        double* u_yneg_buffer = u_buffer_map[domain][LocationType::YNegative];
        double* u_ypos_buffer = u_buffer_map[domain][LocationType::YPositive];

        int nx = domain->get_nx();
        int ny = domain->get_ny();

        // TODO revisit corner closure around mu(nx, ny) / last-cell stress reconstruction
        auto bound_cal_u = [&](int i, int j) {
            double u_xneg = i == 0 ? u_xneg_buffer[j] : u(i - 1, j);
            double u_xpos = i == nx - 1 ? u_xpos_buffer[j] : u(i + 1, j);
            double u_yneg = j == 0 ? u_yneg_buffer[i] : u(i, j - 1);
            double u_ypos = j == ny - 1 ? u_ypos_buffer[i] : u(i, j + 1);

            double v_xneg = i == 0 ? v_xneg_buffer[j] : v(i - 1, j);
            double v_ypos = j == ny - 1 ? v_ypos_buffer[i] : v(i, j + 1);

            double v_xneg_ypos = i == 0 ? (j == ny - 1 ? xneg_ypos_corner_map[domain] : v_xneg_buffer[j + 1]) :
                                          (j == ny - 1 ? v_ypos_buffer[i - 1] : v(i - 1, j + 1));

            double u_conv_x = u_xpos * (u_xpos + 2.0 * u(i, j)) - u_xneg * (u_xneg + 2.0 * u(i, j));
            double u_conv_y = (u(i, j) + u_ypos) * (v_xneg_ypos + v_ypos) - (u_yneg + u(i, j)) * (v_xneg + v(i, j));

            // Non-Newtonian Diffusion
            double diff_x, diff_y;

            // diff_x = (tau_xx(i, j) - tau_xx(i-1, j)) / hx
            // If i=0, tau_xx(-1, j) is needed.
            double t_xx_curr = tau_xx(i, j);
            double t_xx_prev = (i == 0) ? tau_xx_buffer_map[domain][LocationType::XNegative][j] : tau_xx(i - 1, j);

            diff_x = (t_xx_curr - t_xx_prev) / hx;

            // diff_y = (tau_xy(i, j+1) - tau_xy(i, j)) / hy
            // tau_xy indices are valid.
            diff_y = (tau_xy(i, j + 1) - tau_xy(i, j)) / hy;

            double diff = diff_x + diff_y;

            u_temp(i, j) = u(i, j) - dt * (0.25 / hx * u_conv_x + 0.25 / hy * u_conv_y - diff);
        };

        auto bound_cal_v = [&](int i, int j) {
            double v_xneg = i == 0 ? v_xneg_buffer[j] : v(i - 1, j);
            double v_xpos = i == nx - 1 ? v_xpos_buffer[j] : v(i + 1, j);
            double v_yneg = j == 0 ? v_yneg_buffer[i] : v(i, j - 1);
            double v_ypos = j == ny - 1 ? v_ypos_buffer[i] : v(i, j + 1);

            double u_xpos = i == nx - 1 ? u_xpos_buffer[j] : u(i + 1, j);
            double u_yneg = j == 0 ? u_yneg_buffer[i] : u(i, j - 1);

            double u_xpos_yneg = j == 0 ? (i == nx - 1 ? xpos_yneg_corner_map[domain] : u_yneg_buffer[i + 1]) :
                                          (i == nx - 1 ? u_xpos_buffer[j - 1] : u(i + 1, j - 1));

            double v_conv_x = (v(i, j) + v_xpos) * (u_xpos_yneg + u_xpos) - (v_xneg + v(i, j)) * (u_yneg + u(i, j));
            double v_conv_y = v_ypos * (v_ypos + 2.0 * v(i, j)) - v_yneg * (v_yneg + 2.0 * v(i, j));

            // Non-Newtonian Diffusion
            double diff_x, diff_y;

            // diff_x = (tau_xy(i+1, j) - tau_xy(i, j)) / hx
            // tau_xy indices are valid.
            diff_x = (tau_xy(i + 1, j) - tau_xy(i, j)) / hx;

            // diff_y = (tau_yy(i, j) - tau_yy(i, j-1)) / hy
            // If j=0, tau_yy(i, -1) is needed.
            double t_yy_curr = tau_yy(i, j);
            double t_yy_prev = (j == 0) ? tau_yy_buffer_map[domain][LocationType::YNegative][i] : tau_yy(i, j - 1);
            diff_y           = (t_yy_curr - t_yy_prev) / hy;

            double diff = diff_x + diff_y;

            v_temp(i, j) = v(i, j) - dt * (0.25 / hx * v_conv_x + 0.25 / hy * v_conv_y - diff);
        };

        // XNegative
        for (int j = 0; j < ny; j++)
        {
            if (u_var->boundary_type_map[domain][LocationType::XNegative] == PDEBoundaryType::Adjacented)
                bound_cal_u(0, j);
            bound_cal_v(0, j);
        }

        // XPositive
        for (int j = 0; j < ny; j++)
        {
            bound_cal_u(nx - 1, j);
            bound_cal_v(nx - 1, j);
        }

        // YNegative
        for (int i = 0; i < nx; i++)
        {
            bound_cal_u(i, 0);
            if (v_var->boundary_type_map[domain][LocationType::YNegative] == PDEBoundaryType::Adjacented)
                bound_cal_v(i, 0);
        }

        // YPositive
        for (int i = 0; i < nx; i++)
        {
            bound_cal_u(i, ny - 1);
            bound_cal_v(i, ny - 1);
        }

        // Swap data pointers: u <-> u_temp, v <-> v_temp
        swap_field_data(u, u_temp);
        swap_field_data(v, v_temp);
    }
}
