#include "ns_solver2d.h"
#include "boundary_2d_utils.h"

ConcatNSSolver2D::ConcatNSSolver2D(Variable*            in_u_var,
                                   Variable*            in_v_var,
                                   Variable*            in_p_var,
                                   TimeAdvancingConfig* in_time_config,
                                   PhysicsConfig*       in_physics_config,
                                   EnvironmentConfig*   in_env_config)
    : u_var(in_u_var)
    , v_var(in_v_var)
    , p_var(in_p_var)
    , time_config(in_time_config)
    , phy_config(in_physics_config)
    , env_config(in_env_config)
    , left_up_corner_value_map(v_var->left_up_corner_value_map)
    , right_down_corner_value_map(u_var->right_down_corner_value_map)
{
    // config load
    if (in_env_config)
    {
        // Maybe useful in future
    }

    dt      = time_config->dt;
    num_it  = time_config->num_iterations;
    corr_it = time_config->corr_iter;

    nu = phy_config->nu;

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

    p_solver = new ConcatPoissonSolver2D(p_var, env_config);
}

ConcatNSSolver2D::~ConcatNSSolver2D()
{
    for (auto& [domain, field] : u_field_map)
        delete u_temp_field_map[domain];
    for (auto& [domain, field] : v_field_map)
        delete v_temp_field_map[domain];
    delete p_solver;
}

void ConcatNSSolver2D::variable_check()
{
    if (u_var->position_type != VariablePositionType::XEdge)
        throw std::runtime_error("ConcatNSSolver2D: u->position_type is not XEdge");
    if (v_var->position_type != VariablePositionType::YEdge)
        throw std::runtime_error("ConcatNSSolver2D: v->position_type is not YEdge");
    if (p_var->position_type != VariablePositionType::Center)
        throw std::runtime_error("ConcatNSSolver2D: p->position_type is not Center");
}

void ConcatNSSolver2D::solve()
{
    // update boundary for NS
    phys_boundary_update();
    nondiag_shared_boundary_update();
    diag_shared_boundary_update();

    // NS
    euler_conv_diff_inner();
    euler_conv_diff_outer();

    // update boundary for divu
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

        // Optimization: Precompute inverse values to replace division with multiplication
        double inv_hx = 1.0 / hx;
        double inv_hy = 1.0 / hy;
        double inv_hx2 = inv_hx * inv_hx;
        double inv_hy2 = inv_hy * inv_hy;
        double factor_x = 0.25 * inv_hx;
        double factor_y = 0.25 * inv_hy;

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
                double diff = (u(i + 1, j) + u(i - 1, j) - 2.0 * u(i, j)) * inv_hx2 +
                              (u(i, j + 1) + u(i, j - 1) - 2.0 * u(i, j)) * inv_hy2;

                u_temp(i, j) = u(i, j) - dt * (factor_x * conv_x + factor_y * conv_y - nu * diff);
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
                double diff = (v(i + 1, j) + v(i - 1, j) - 2.0 * v(i, j)) * inv_hx2 +
                              (v(i, j + 1) + v(i, j - 1) - 2.0 * v(i, j)) * inv_hy2;

                v_temp(i, j) = v(i, j) - dt * (factor_x * conv_x + factor_y * conv_y - nu * diff);
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

        // Optimization: Precompute inverse values
        double inv_hx = 1.0 / hx;
        double inv_hy = 1.0 / hy;
        double inv_hx2 = inv_hx * inv_hx;
        double inv_hy2 = inv_hy * inv_hy;
        double factor_x = 0.25 * inv_hx;
        double factor_y = 0.25 * inv_hy;

        double* v_left_buffer  = v_buffer_map[domain][LocationType::Left];
        double* v_right_buffer = v_buffer_map[domain][LocationType::Right];
        double* v_down_buffer  = v_buffer_map[domain][LocationType::Down];
        double* v_up_buffer    = v_buffer_map[domain][LocationType::Up];

        double* u_left_buffer  = u_buffer_map[domain][LocationType::Left];
        double* u_right_buffer = u_buffer_map[domain][LocationType::Right];
        double* u_down_buffer  = u_buffer_map[domain][LocationType::Down];
        double* u_up_buffer    = u_buffer_map[domain][LocationType::Up];

        int nx = domain->get_nx();
        int ny = domain->get_ny();

        auto bound_cal_u = [&](int i, int j) {
            double u_left  = i == 0 ? u_left_buffer[j] : u(i - 1, j);
            double u_right = i == nx - 1 ? u_right_buffer[j] : u(i + 1, j);
            double u_down  = j == 0 ? u_down_buffer[i] : u(i, j - 1);
            double u_up    = j == ny - 1 ? u_up_buffer[i] : u(i, j + 1);

            double v_left = i == 0 ? v_left_buffer[j] : v(i - 1, j);
            double v_up   = j == ny - 1 ? v_up_buffer[i] : v(i, j + 1);

            double v_left_up = i == 0 ? (j == ny - 1 ? left_up_corner_value_map[domain] : v_left_buffer[j + 1]) :
                                        (j == ny - 1 ? v_up_buffer[i - 1] : v(i - 1, j + 1));

            double u_conv_x = u_right * (u_right + 2.0 * u(i, j)) - u_left * (u_left + 2.0 * u(i, j));
            double u_conv_y = (u(i, j) + u_up) * (v_left_up + v_up) - (u_down + u(i, j)) * (v_left + v(i, j));
            double u_diff   = (u_right + u_left - 2.0 * u(i, j)) * inv_hx2 + (u_up + u_down - 2.0 * u(i, j)) * inv_hy2;

            u_temp(i, j) = u(i, j) - dt * (factor_x * u_conv_x + factor_y * u_conv_y - nu * u_diff);
        };

        auto bound_cal_v = [&](int i, int j) {
            double v_left  = i == 0 ? v_left_buffer[j] : v(i - 1, j);
            double v_right = i == nx - 1 ? v_right_buffer[j] : v(i + 1, j);
            double v_down  = j == 0 ? v_down_buffer[i] : v(i, j - 1);
            double v_up    = j == ny - 1 ? v_up_buffer[i] : v(i, j + 1);

            double u_right = i == nx - 1 ? u_right_buffer[j] : u(i + 1, j);
            double u_down  = j == 0 ? u_down_buffer[i] : u(i, j - 1);

            double u_right_down = j == 0 ? (i == nx - 1 ? right_down_corner_value_map[domain] : u_down_buffer[i + 1]) :
                                           (i == nx - 1 ? u_right_buffer[j - 1] : u(i + 1, j - 1));

            double v_conv_x = (v(i, j) + v_right) * (u_right_down + u_right) - (v_left + v(i, j)) * (u_down + u(i, j));
            double v_conv_y = v_up * (v_up + 2.0 * v(i, j)) - v_down * (v_down + 2.0 * v(i, j));
            double v_diff   = (v_right + v_left - 2.0 * v(i, j)) * inv_hx2 + (v_up + v_down - 2.0 * v(i, j)) * inv_hy2;

            v_temp(i, j) = v(i, j) - dt * (factor_x * v_conv_x + factor_y * v_conv_y - nu * v_diff);
        };

        // Left
        for (int j = 0; j < ny; j++)
        {
            if (u_var->boundary_type_map[domain][LocationType::Left] == PDEBoundaryType::Adjacented)
                bound_cal_u(0, j);
            bound_cal_v(0, j);
        }

        // Right
        for (int j = 0; j < ny; j++)
        {
            bound_cal_u(nx - 1, j);
            bound_cal_v(nx - 1, j);
        }

        // Down
        for (int i = 0; i < nx; i++)
        {
            bound_cal_u(i, 0);
            if (v_var->boundary_type_map[domain][LocationType::Down] == PDEBoundaryType::Adjacented)
                bound_cal_v(i, 0);
        }

        // Up
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