#include "base/domain/domain2d.h"
#include "base/domain/geometry2d.h"
#include "base/domain/variable.h"
#include "base/field/field2.h"
#include "base/location_boundary.h"
#include "io/common.h"
#include "io/config.h"
#include "io/csv_writer_2d.h"
#include "ns/ns_solver2d.h"
#include "poisson_base/io/progress_bar.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
/**
 *
 * y
 * ▲
 * │
 * │      ┌──────┐
 * │      │      │
 * │      │  A5  │
 * │      │      │
 * ├──────┼──────┼──────┐
 * │      │      │      │
 * │  A1  │  A2  │  A3  │
 * │      │      │      │
 * ├──────┼──────┼──────┘
 * │      │      │
 * │      │  A4  │
 * │      │      │
 * └──────┴──────┴──────────►
 * O                         x
 *
 */
int main(int argc, char* argv[])
{
    // 参数读取与物理配置
    Geometry2D geo_cross;
    double     h = 0.02;

    EnvironmentConfig* env_config    = new EnvironmentConfig();
    env_config->showGmresRes         = false;
    env_config->showCurrentStep      = false;
    TimeAdvancingConfig* time_config = new TimeAdvancingConfig();
    time_config->dt                  = 1.0 / 10 * h; // CFL condition
    time_config->num_iterations      = 70 * 1.5 / (time_config->dt);
    // time_config->num_iterations   = 10;
    PhysicsConfig* physics_config = new PhysicsConfig();
    physics_config->set_Re(200);

    int step_to_save = time_config->num_iterations / 10;
    // int step_to_save = 1;

    double lx2 = 1;
    double ly2 = 1;
    double lx1 = 10 * lx2;
    double lx3 = lx1;
    double ly4 = 60 * ly2;
    double ly5 = ly4;

    int nx2 = lx2 / h;
    int ny2 = ly2 / h;
    int nx1 = lx1 / h;
    int nx3 = lx3 / h;
    int ny4 = ly4 / h;
    int ny5 = ly5 / h;

    int ny1 = ny2;
    int ny3 = ny2;
    int nx4 = nx2;
    int nx5 = nx2;
    int ly1 = ly2;
    int ly3 = ly2;
    int lx4 = lx2;
    int lx5 = lx2;

    std::cout << "Construct cross-shaped channel geometry..." << std::endl;
    std::cout << "Domain A2: " << nx2 << " x " << ny2 << " (" << lx2 << " x " << ly2 << ")" << std::endl;
    std::cout << "Domain A1: " << nx1 << " x " << ny1 << " (" << lx1 << " x " << ly1 << ")" << std::endl;
    std::cout << "Domain A3: " << nx3 << " x " << ny3 << " (" << lx3 << " x " << ly3 << ")" << std::endl;
    std::cout << "Domain A4: " << nx4 << " x " << ny4 << " (" << lx4 << " x " << ly4 << ")" << std::endl;
    std::cout << "Domain A5: " << nx5 << " x " << ny5 << " (" << lx5 << " x " << ly5 << ")" << std::endl;

    // 输出总网格数
    int total_grid = nx1 * ny1 + nx2 * ny2 + nx3 * ny3 + nx4 * ny4 + nx5 * ny5;
    std::cout << "Total grid points: " << total_grid << std::endl;

    Domain2DUniform A2(nx2, ny2, lx2, ly2, "A2");
    Domain2DUniform A1(nx1, ny1, lx1, ly1, "A1");
    Domain2DUniform A3(nx3, ny3, lx3, ly3, "A3");
    Domain2DUniform A4(nx4, ny4, lx4, ly4, "A4");
    Domain2DUniform A5(nx5, ny5, lx5, ly5, "A5");
    geo_cross.add_domain(A1);
    geo_cross.add_domain(A2);
    geo_cross.add_domain(A3);
    geo_cross.add_domain(A4);
    geo_cross.add_domain(A5);

    // Construct cross connectivity
    geo_cross.connect(A2, LocationType::Left, A1);
    geo_cross.connect(A2, LocationType::Right, A3);
    geo_cross.connect(A2, LocationType::Down, A4);
    geo_cross.connect(A2, LocationType::Up, A5);

    Variable u("u"), v("v"), p("p");
    u.set_geometry(geo_cross);
    v.set_geometry(geo_cross);
    p.set_geometry(geo_cross);

    // Fields on each domain
    field2 u_A1("u_A1"), u_A2("u_A2"), u_A3("u_A3"), u_A4("u_A4"), u_A5("u_A5");
    field2 v_A1("v_A1"), v_A2("v_A2"), v_A3("v_A3"), v_A4("v_A4"), v_A5("v_A5");
    field2 p_A1("p_A1"), p_A2("p_A2"), p_A3("p_A3"), p_A4("p_A4"), p_A5("p_A5");

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

    // Helper setters
    auto set_dirichlet_zero = [](Variable& var, Domain2DUniform* d, LocationType loc) {
        var.set_boundary_type(d, loc, PDEBoundaryType::Dirichlet);
        var.set_boundary_value(d, loc, 0.0);
    };
    auto set_neumann_zero = [](Variable& var, Domain2DUniform* d, LocationType loc) {
        var.set_boundary_type(d, loc, PDEBoundaryType::Neumann);
    };
    auto is_adjacented = [&](Domain2DUniform* d, LocationType loc) {
        return geo_cross.adjacency.count(d) && geo_cross.adjacency[d].count(loc);
    };
    // Default outer boundaries
    std::vector<Domain2DUniform*> domains = {&A1, &A2, &A3, &A4, &A5};
    std::vector<LocationType> dirs = {LocationType::Left, LocationType::Right, LocationType::Down, LocationType::Up};

    for (auto* d : domains)
    {
        for (auto loc : dirs)
        {
            if (is_adjacented(d, loc))
                continue; // internal boundaries handled automatically
            // velocity: default wall (Dirichlet 0)
            set_dirichlet_zero(u, d, loc);
            set_dirichlet_zero(v, d, loc);
            // pressure: default Neumann (zero gradient)
            set_neumann_zero(p, d, loc);
        }
    }

    // Inlet profiles for symmetry validation (Poiseuille)
    const double U0 = 1.0;

    u.set_boundary_type(&A1, LocationType::Left, PDEBoundaryType::Dirichlet);
    u.has_boundary_value_map[&A1][LocationType::Left] = true;
    set_dirichlet_zero(v, &A1, LocationType::Left);
    // A1 Left: u(y_norm) = +6*U0*y_norm*(1-y_norm)
    for (int j = 0; j < u_A1.get_ny(); ++j)
    {
        double y_norm                                    = (j + 0.5) / static_cast<double>(u_A1.get_ny());
        double u_val                                     = 6.0 * U0 * y_norm * (1.0 - y_norm);
        u.boundary_value_map[&A1][LocationType::Left][j] = u_val;
    }
    set_dirichlet_zero(v, &A1, LocationType::Left);

    // A3 Right: u(y_norm) = -6*U0*y_norm*(1-y_norm)
    u.set_boundary_type(&A3, LocationType::Right, PDEBoundaryType::Dirichlet);
    u.has_boundary_value_map[&A3][LocationType::Right] = true;
    set_dirichlet_zero(v, &A3, LocationType::Right);
    for (int j = 0; j < u_A3.get_ny(); ++j)
    {
        double y_norm                                     = (j + 0.5) / static_cast<double>(u_A3.get_ny());
        double u_val                                      = -6.0 * U0 * y_norm * (1.0 - y_norm);
        u.boundary_value_map[&A3][LocationType::Right][j] = u_val;
    }

    // A4 Down: open/symmetry as Neumann for u and v
    set_neumann_zero(u, &A4, LocationType::Down);
    set_neumann_zero(v, &A4, LocationType::Down);

    // A5 Up: open/symmetry as Neumann for u and v
    set_neumann_zero(u, &A5, LocationType::Up);
    set_neumann_zero(v, &A5, LocationType::Up);

    ConcatNSSolver2D ns_solver(&u, &v, &p, time_config, physics_config, env_config);

    ns_solver.p_solver->set_parameter(10, 1.e-7, 1000);
    // Generate timestamp directory
    std::string nowtime_dir = "result/cross_shaped_channel_2d/" + IO::create_timestamp();

    IO::create_directory(nowtime_dir);

    // ns_solver.phys_boundary_update();
    // ns_solver.nondiag_shared_boundary_update();
    // ns_solver.diag_shared_boundary_update();
    // IO::var_to_csv(u, nowtime_dir + "/u_step_" + std::to_string(0));
    // IO::var_to_csv(v, nowtime_dir + "/v_step_" + std::to_string(0));
    // IO::var_to_csv(p, nowtime_dir + "/p_step_" + std::to_string(0));

    // Solve
    IO::ProgressBar pb(time_config->num_iterations, 50, 100);
    std::cout << "Starting simulation with " << time_config->num_iterations << " iterations..." << std::endl;

    for (int step = 1; step <= time_config->num_iterations; step++)
    {
        pb.update(step);

        // 只在前5步和每200步输出残差
        if (step % 200 == 0 || step <= 5)
        {
            env_config->showGmresRes = true;
            // std::cout << "step: " << step << "/" << time_config->num_iterations << "\n";
        }
        else
        {
            env_config->showGmresRes = false;
        }
        ns_solver.solve();
        if (step % step_to_save == 0)
        {
            std::cout << "Saving step " << (step) << " to CSV files." << std::endl;
            // update boundary for NS
            ns_solver.phys_boundary_update();
            ns_solver.nondiag_shared_boundary_update();
            ns_solver.diag_shared_boundary_update();
            IO::var_to_csv_full(u, nowtime_dir + "/u/u_" + std::to_string(step));
            IO::var_to_csv_full(v, nowtime_dir + "/v/v_" + std::to_string(step));
            IO::var_to_csv_full(p, nowtime_dir + "/p/p_" + std::to_string(step));
        }
        if (std::isnan(u_A1(1, 1)))
        {
            std::cout << "=== DIVERGENCE ===" << std::endl;
            return -1;
        }
    }
    pb.finish();
    std::cout << "Simulation finished." << std::endl;
    IO::var_to_csv_full(u, nowtime_dir + "final/u_" + std::to_string(time_config->num_iterations));
    IO::var_to_csv_full(v, nowtime_dir + "final/v_" + std::to_string(time_config->num_iterations));
    IO::var_to_csv_full(p, nowtime_dir + "final/p_" + std::to_string(time_config->num_iterations));
    return 0;
}