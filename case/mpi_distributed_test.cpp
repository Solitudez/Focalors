#include <cmath>
#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <vector>

#include "base/location_boundary.h"
#include "pe_mpi_distributed/d_domain_2d.h"
#include "pe_mpi_distributed/d_field_2d.h"
#include "pe_mpi_distributed/mpi_poisson_solver2d.h"

// Analytical solution for testing (same as pe_test.cpp logic simplified)
// u(x, y) = sin(pi*x) * sin(pi*y)
// f(x, y) = -2*pi^2 * sin(pi*x) * sin(pi*y)
// Domain: [0, 1] x [0, 1]

void gather_and_print(DField2D& f, DDomain2D& domain, std::string name, int nx, int ny, int rank, int size)
{
    field2& local_data = f.get_local_data();
    int     local_nx   = domain.get_local_nx();

    std::vector<double> send_buf;
    send_buf.reserve(local_nx * ny);
    for (int i = 0; i < local_nx; ++i)
    {
        for (int j = 0; j < ny; ++j)
        {
            send_buf.push_back(local_data(i, j));
        }
    }

    std::vector<int> recv_counts(size);
    int              send_count = local_nx * ny;
    MPI_Gather(&send_count, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int> displs(size);
    if (rank == 0)
    {
        displs[0] = 0;
        for (int i = 1; i < size; ++i)
            displs[i] = displs[i - 1] + recv_counts[i - 1];
    }

    std::vector<double> recv_buf;
    if (rank == 0)
        recv_buf.resize(nx * ny);

    MPI_Gatherv(send_buf.data(),
                send_count,
                MPI_DOUBLE,
                recv_buf.data(),
                recv_counts.data(),
                displs.data(),
                MPI_DOUBLE,
                0,
                MPI_COMM_WORLD);

    if (rank == 0)
    {
        std::cout << name << " = [" << std::endl;
        for (int i = 0; i < nx; ++i)
        {
            for (int j = 0; j < ny; ++j)
            {
                std::cout << std::setw(12) << recv_buf[i * ny + j] << " ";
            }
            std::cout << ";" << std::endl;
        }
        std::cout << "];" << std::endl;
    }
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int    nx = 128;
    int    ny = 128;
    double lx = 1.0;
    double ly = 1.0;
    double hx = lx / nx;
    double hy = ly / ny;

    // 1. Create Distributed Domain
    DDomain2D domain(MPI_COMM_WORLD, nx, ny, hx, hy);

    // 2. Create Distributed Fields
    DField2D f(&domain, "f");
    DField2D u(&domain, "u");

    // 3. Initialize RHS (f) locally
    // Each rank initializes its own part of the domain
    field2& f_local  = f.get_local_data();
    int     i_start  = domain.get_local_i_start();
    int     local_nx = domain.get_local_nx();

    double pi = 3.14159265358979323846;

    for (int i = 0; i < local_nx; ++i)
    {
        int    i_global = i_start + i;
        double x        = (i_global + 0.5) * hx; // Cell center
        for (int j = 0; j < ny; ++j)
        {
            double y      = (j + 0.5) * hy; // Cell center
            double val    = -2.0 * pi * pi * std::sin(pi * x) * std::sin(pi * y);
            f_local(i, j) = val;
        }
    }

    // Print RHS
    gather_and_print(f, domain, "RHS", nx, ny, rank, size);

    // 4. Create Solver
    // Dirichlet boundaries on all sides
    MPIDistributedPoissonSolver2D solver(&domain,
                                         PDEBoundaryType::Dirichlet,
                                         PDEBoundaryType::Dirichlet,
                                         PDEBoundaryType::Dirichlet,
                                         PDEBoundaryType::Dirichlet);

    // 5. Solve
    // Input f is RHS, Output f is Solution (in-place)
    solver.solve(f);

    // Print Result
    gather_and_print(f, domain, "Result", nx, ny, rank, size);

    // 6. Verify Result
    // Compare f (now u) with analytical solution
    double local_error_sq = 0.0;
    for (int i = 0; i < local_nx; ++i)
    {
        int    i_global = i_start + i;
        double x        = (i_global + 0.5) * hx;
        for (int j = 0; j < ny; ++j)
        {
            double y     = (j + 0.5) * hy;
            double exact = std::sin(pi * x) * std::sin(pi * y);
            double diff  = f_local(i, j) * hx * hy - exact;
            local_error_sq += diff * diff;
        }
    }

    double global_error_sq = 0.0;
    MPI_Reduce(&local_error_sq, &global_error_sq, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        double l2_error = std::sqrt(global_error_sq * hx * hy);
        std::cout << "[Test] Grid: " << nx << "x" << ny << std::endl;
        std::cout << "[Test] L2 Error: " << l2_error << std::endl;

        if (l2_error < 1e-3)
            std::cout << "[Test] PASSED" << std::endl;
        else
            std::cout << "[Test] FAILED" << std::endl;
    }

    MPI_Finalize();
    return 0;
}

/*
% MATLAB Verification Program
% Copy the output 'RHS' and 'Result' matrices from the C++ output into MATLAB.

nx = 128;
ny = 128;
lx = 1.0;
ly = 1.0;
hx = lx / nx;
hy = ly / ny;

% Generate Analytical Solution
[Y, X] = meshgrid((0.5:ny-0.5)*hy, (0.5:nx-0.5)*hx); % Note: X is rows, Y is cols to match C++ (i,j)
U_exact = sin(pi*X) .* sin(pi*Y);

% Verify RHS (Optional)
% F_exact = -2 * pi^2 * sin(pi*X) .* sin(pi*Y);
% rhs_error = norm(RHS - F_exact, 'fro') / sqrt(nx*ny);
% fprintf('RHS L2 Error: %e\n', rhs_error);

% Verify Result
% Assuming 'Result' variable exists from C++ output
if exist('Result', 'var')
    error_diff = Result - U_exact;
    l2_error = norm(error_diff, 'fro') * sqrt(hx * hy);
    fprintf('Solution L2 Error: %e\n', l2_error);

    figure;
    subplot(1,3,1); imagesc(Result); title('Computed Solution'); colorbar;
    subplot(1,3,2); imagesc(U_exact); title('Exact Solution'); colorbar;
    subplot(1,3,3); imagesc(abs(error_diff)); title('Absolute Error'); colorbar;
else
    fprintf('Variable Result not found. Please paste the C++ output.\n');
end
*/
