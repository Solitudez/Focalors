#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <numeric>
#include <vector>

#include "base/location_boundary.h"
#include "pe_mpi_distributed/d_domain_2d.h"
#include "pe_mpi_distributed/d_field_2d.h"
#include "pe_mpi_distributed/mpi_poisson_solver2d.h"

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int N = 256; // default domain size
    if (argc > 1)
    {
        N = std::atoi(argv[1]);
    }

    const int nx      = N;
    const int ny      = N;
    const int repeats = 5;

    const double lx = 1.0, ly = 1.0;
    const double hx = lx / nx;
    const double hy = ly / ny;
    const double pi = 3.14159265358979323846;

    // distributed domain & field
    DDomain2D domain(MPI_COMM_WORLD, nx, ny, hx, hy);
    DField2D  f(&domain, "f");

    field2& f_local  = f.get_local_data();
    int     i_start  = domain.get_local_i_start();
    int     local_nx = domain.get_local_nx();

    // init RHS
    auto init_rhs = [&]() {
        for (int i = 0; i < local_nx; ++i)
        {
            int    ig = i_start + i;
            double x  = (ig + 0.5) * hx;
            for (int j = 0; j < ny; ++j)
            {
                double y      = (j + 0.5) * hy;
                f_local(i, j) = -2.0 * pi * pi * std::sin(pi * x) * std::sin(pi * y);
            }
        }
    };

    init_rhs();

    // solver
    MPIDistributedPoissonSolver2D solver(&domain,
                                         PDEBoundaryType::Dirichlet,
                                         PDEBoundaryType::Dirichlet,
                                         PDEBoundaryType::Dirichlet,
                                         PDEBoundaryType::Dirichlet);

    // timing
    std::vector<double> times;
    times.reserve(repeats);

    for (int r = 0; r < repeats; ++r)
    {
        init_rhs();
        MPI_Barrier(MPI_COMM_WORLD);

        double t0 = 0.0, t1 = 0.0;

        if (rank == 0)
            t0 = MPI_Wtime();

        solver.solve(f);

        MPI_Barrier(MPI_COMM_WORLD);

        if (rank == 0)
        {
            t1 = MPI_Wtime();
            times.push_back(t1 - t0);
        }
    }

    if (rank == 0)
    {
        double sum = std::accumulate(times.begin(), times.end(), 0.0);
        double avg = sum / times.size();

        double sq = 0.0;
        for (double t : times)
            sq += (t - avg) * (t - avg);
        double stddev = (times.size() > 1 ? std::sqrt(sq / (times.size() - 1)) : 0.0);

        std::cout << avg << " " << stddev << "\n";
    }

    // Use to validate process is actually runing at designed nodes
    // You can use the command to get running process at node:
    // ps -ef | grep mpi_distributed_time_test | grep -v grep | wc -l
    // {
    //     auto start = std::chrono::steady_clock::now();
    //     while (std::chrono::steady_clock::now() - start < std::chrono::minutes(5))
    //     {
    //         // busy wait
    //     }
    // }

    MPI_Finalize();
    return 0;
}
