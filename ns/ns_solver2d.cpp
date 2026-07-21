#include "ns_solver2d.h"
#include "boundary_2d_utils.h"
#include "mhd_module_2d_mac.h"

ConcatNSSolver2D::ConcatNSSolver2D(Variable2D*            in_u_var,
                                   Variable2D*            in_v_var,
                                   Variable2D*            in_p_var,
                                   ConcatPoissonSolver2D* in_p_solver)
    : u_var(in_u_var)
    , v_var(in_v_var)
    , p_var(in_p_var)
    , p_solver(in_p_solver)
    , xneg_ypos_corner_map(v_var->xneg_ypos_corner_map)
    , xpos_yneg_corner_map(u_var->xpos_yneg_corner_map)
{
    TimeAdvancingConfig& time_cfg    = TimeAdvancingConfig::Get();
    PhysicsConfig&       physics_cfg = PhysicsConfig::Get();

    dt      = time_cfg.dt;
    num_it  = time_cfg.num_iterations;
    corr_it = time_cfg.corr_iter;

    nu = physics_cfg.nu;

    // check u v p share one geometry
    if (u_var->geometry != v_var->geometry || u_var->geometry != p_var->geometry)
        throw std::runtime_error("ConcatNSSolver2D: u v p do not share one geometry");

    // geometry double check
    if (u_var->geometry == nullptr)
        throw std::runtime_error("ConcatNSSolver2D: u->geometry is null");
    if (!u_var->geometry->is_checked)
        u_var->geometry->check();
    if (u_var->geometry->tree_root == nullptr || u_var->geometry->tree_map.empty())
        u_var->geometry->solve_prepare();

    domains   = u_var->geometry->domains;
    adjacency = u_var->geometry->adjacency;

    u_field_map = u_var->field_map;
    v_field_map = v_var->field_map;
    p_field_map = p_var->field_map;

    u_buffer_map = u_var->buffer_map;
    v_buffer_map = v_var->buffer_map;
    p_buffer_map = p_var->buffer_map;

    // Construct the temp field for each domain
    for (auto& [domain, field] : u_field_map)
        u_temp_field_map[domain] = new field2(field->get_nx(), field->get_ny(), field->get_name() + "_temp");
    for (auto& [domain, field] : v_field_map)
        v_temp_field_map[domain] = new field2(field->get_nx(), field->get_ny(), field->get_name() + "_temp");

    // update boundary for first step
    phys_boundary_update();
    nondiag_shared_boundary_update();
    diag_shared_boundary_update();
}

void ConcatNSSolver2D::init_mhd(Variable2D* in_phi_var)
{
    PhysicsConfig& physics_cfg = PhysicsConfig::Get();
    if (!physics_cfg.enable_mhd)
        return;

    if (!mhd_module)
        mhd_module = std::unique_ptr<MHDModule2D>(new MHDModule2D(u_var, v_var));

    mhd_module->init(in_phi_var);
}

void ConcatNSSolver2D::setTimeStep(double in_dt)
{
    dt = in_dt;
    if (mhd_module)
        mhd_module->setTimeStep(in_dt);
}

void ConcatNSSolver2D::variable_check()
{
    if (u_var->position_type != VariablePositionType::XFace)
        throw std::runtime_error("ConcatNSSolver2D: u->position_type is not XFace");
    if (v_var->position_type != VariablePositionType::YFace)
        throw std::runtime_error("ConcatNSSolver2D: v->position_type is not YFace");
    if (p_var->position_type != VariablePositionType::Center)
        throw std::runtime_error("ConcatNSSolver2D: p->position_type is not Center");
}

void ConcatNSSolver2D::raw_vorticity_update(Variable2D& vorticity_var) const
{
    raw_vorticity_update(*u_var, *v_var, vorticity_var);
}

void ConcatNSSolver2D::raw_vorticity_update(const Variable2D& u_var, const Variable2D& v_var, Variable2D& vorticity_var)
{
    if (u_var.geometry == nullptr || v_var.geometry == nullptr || vorticity_var.geometry == nullptr)
        throw std::runtime_error("raw_vorticity_update: variables must have geometry");
    if (u_var.geometry != v_var.geometry || u_var.geometry != vorticity_var.geometry)
        throw std::runtime_error("raw_vorticity_update: u, v, vorticity must share one geometry");
    if (u_var.position_type != VariablePositionType::XFace)
        throw std::runtime_error("raw_vorticity_update: u must be XFace");
    if (v_var.position_type != VariablePositionType::YFace)
        throw std::runtime_error("raw_vorticity_update: v must be YFace");
    if (vorticity_var.position_type != VariablePositionType::Center)
        throw std::runtime_error("raw_vorticity_update: vorticity output must be center-sized");

    for (auto* domain : u_var.geometry->domains)
    {
        const field2& u     = *u_var.field_map.at(domain);
        const field2& v     = *v_var.field_map.at(domain);
        field2&       omega = *vorticity_var.field_map.at(domain);

        const int nx = domain->get_nx();
        const int ny = domain->get_ny();
        if (u.get_nx() != nx || u.get_ny() != ny || v.get_nx() != nx || v.get_ny() != ny || omega.get_nx() != nx ||
            omega.get_ny() != ny)
            throw std::runtime_error("raw_vorticity_update: field size mismatch on domain " + domain->name);

        const double hx            = domain->get_hx();
        const double hy            = domain->get_hy();
        double*      v_xneg_buffer = v_var.buffer_map.at(domain).at(LocationType::XNegative);
        double*      u_yneg_buffer = u_var.buffer_map.at(domain).at(LocationType::YNegative);

        OPENMP_PARALLEL_FOR()
        for (int i = 0; i < nx; ++i)
        {
            for (int j = 0; j < ny; ++j)
            {
                const double v_left = i == 0 ? v_xneg_buffer[j] : v(i - 1, j);
                const double u_down = j == 0 ? u_yneg_buffer[i] : u(i, j - 1);
                omega(i, j)         = (v(i, j) - v_left) / hx - (u(i, j) - u_down) / hy;
            }
        }
    }
}

void ConcatNSSolver2D::solve()
{
    PhysicsConfig& physics_cfg = PhysicsConfig::Get();

    // NS
    euler_conv_diff_inner();
    euler_conv_diff_outer();

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

    // Update boundary data for div(u*). The pressure-correction path only reads
    // direct neighbor buffers, so diagonal corner data is not needed here.
    phys_boundary_update();
    nondiag_shared_boundary_update();

    for (int it = 0; it < corr_it; it++)
    {
        // divu
        velocity_div_inner();
        velocity_div_outer();

        // PE
        normalize_pressure();
        p_solver->solve();

        // update buffer for p
        pressure_buffer_update();

        // p grad
        add_pressure_gradient();
    }

    // Restore the full shared-boundary state for the next predictor step and for
    // other modules that may read diagonal corner values.
    phys_boundary_update();
    nondiag_shared_boundary_update();
    diag_shared_boundary_update();
}

void ConcatNSSolver2D::euler_conv_diff_inner()
{
    for (auto& domain : domains)
    {
        field2& u = *u_field_map[domain];
        field2& v = *v_field_map[domain];
        field2& p = *p_field_map[domain];

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
                double diff = (u(i + 1, j) + u(i - 1, j) - 2.0 * u(i, j)) / hx / hx +
                              (u(i, j + 1) + u(i, j - 1) - 2.0 * u(i, j)) / hy / hy;

                u_temp(i, j) = u(i, j) - dt * (0.25 / hx * conv_x + 0.25 / hy * conv_y - nu * diff);
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
                double diff = (v(i + 1, j) + v(i - 1, j) - 2.0 * v(i, j)) / hx / hx +
                              (v(i, j + 1) + v(i, j - 1) - 2.0 * v(i, j)) / hy / hy;

                v_temp(i, j) = v(i, j) - dt * (0.25 / hx * conv_x + 0.25 / hy * conv_y - nu * diff);
            }
        }
    }
}

void ConcatNSSolver2D::euler_conv_diff_outer()
{
    for (auto& domain : domains)
    {
        field2& u      = *u_field_map[domain];
        field2& v      = *v_field_map[domain];
        field2& u_temp = *u_temp_field_map[domain];
        field2& v_temp = *v_temp_field_map[domain];

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
            double u_diff   = (u_xpos + u_xneg - 2.0 * u(i, j)) / hx / hx + (u_ypos + u_yneg - 2.0 * u(i, j)) / hy / hy;

            u_temp(i, j) = u(i, j) - dt * (0.25 / hx * u_conv_x + 0.25 / hy * u_conv_y - nu * u_diff);
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
            double v_diff   = (v_xpos + v_xneg - 2.0 * v(i, j)) / hx / hx + (v_ypos + v_yneg - 2.0 * v(i, j)) / hy / hy;

            v_temp(i, j) = v(i, j) - dt * (0.25 / hx * v_conv_x + 0.25 / hy * v_conv_y - nu * v_diff);
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
