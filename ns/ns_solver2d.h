#pragma once

#include "base/pch.h"

#include "mhd_module_2d_mac.h"
#include "pe/concat/concat_solver2d.h"

#include <memory>
#include <unordered_map>
#include <vector>

class ConcatNSSolver2D
{
    // Simple: Only for single main domain geometry
public:
    Variable2D *           u_var = nullptr, *v_var = nullptr, *p_var = nullptr;
    ConcatPoissonSolver2D* p_solver = nullptr;

    // Non-Newtonian fields
    Variable2D *mu_var = nullptr, *tau_xx_var = nullptr, *tau_yy_var = nullptr, *tau_xy_var = nullptr;

    ConcatNSSolver2D(Variable2D*            in_u_var,
                     Variable2D*            in_v_var,
                     Variable2D*            in_p_var,
                     ConcatPoissonSolver2D* in_p_solver);

    // void init();
    void variable_check();
    void init_mhd(Variable2D* in_phi_var = nullptr);
    void setTimeStep(double in_dt);
    void solve();
    void normalize_pressure();

    // Non-Newtonian methods
    void init_nonnewton(Variable2D* in_mu_var,
                        Variable2D* in_tau_xx_var,
                        Variable2D* in_tau_yy_var,
                        Variable2D* in_tau_xy_var,
                        Variable2D* in_phi_var = nullptr);
    void solve_nonnewton();
    void viscosity_update();
    void mu_shared_boundary_field_update();
    void tau_xy_shared_boundary_field_update();
    void stress_update();
    void stress_buffer_update();
    void euler_conv_diff_inner_nonnewton();
    void euler_conv_diff_outer_nonnewton();

    /**
     * Physics boundary update.
     *
     * Step 1 of PISO.
     *
     * Iterate through each boundary that may require updating,
     * and determine whether updating is needed based on the connection conditions.
     *
     * If a boundary should be update, the updating strategy is:
     * A field-related row (or column) is denoted as fb, and a physical boundary is denoted as pb.
     * (1) When fb coincides with pb, compute fb based on the pb type and pb value.
     * (2) When fb and its corresponding buffer are symmetric with respect to pb, compute the buffer based on the pb
     * type, pb value, and fb.
     * For the first case, if the boundary condition is of Dirichlet type,
     * the assignment only needs to be done once, without repeating it in each iteration of the solution loop.
     * However, if the boundary condition is of Neumann type,
     * it will depend on the adjacent row/column near the boundary,
     * which in turn requires the solution of the Navier–Stokes equations.
     * Therefore, this type of boundary needs to be updated in every iteration of the solution loop.
     * For simplicity in programming, we will not distinguish
     * whether a boundary needs to be assigned only once or in every iteration,
     * since the computational cost of the boundary handling function is already very small.
     */
    void phys_boundary_update();
    /**
     * Non-diagonal shared boundary update.
     *
     * Step 2 of PISO.
     *
     * Iterate through each boundary that may require updating,
     * and determine whether updating is needed based on the connection conditions.
     *
     * If a boundary should be update, the updating strategy is:
     *
     * When a local buffer or corner has a physically coincident position pos in a non-diagonal neighboring field
     * (denoted as fn), assign fn(pos) to the local buffer or corner.
     */
    void nondiag_shared_boundary_update();

    /**
     * Diagonal shared boundary update.
     *
     * Step 3 of PISO.
     *
     * Iterate through each boundary that may require updating,
     * and determine whether updating is needed based on the connection conditions.
     *
     * YPositivedating strategy is similar to nondiag_shared_boundary_update.
     *
     * The importance of diagonal shared boundary updating:
     * When performing NS calculations on shared boundary, if the calculation point is an endpoint of a column or row,
     * then certain buffer values ​​need to be taken.
     * Among these buffer dependencies, one buffer is special because it:
     * (1) When a diagonal domain exists, it originates from the diagonal domain.
     * (2) When a diagonal domain does not exist, it originates from the directly adjacent domain.
     *
     * This step is needed by the predictor's outer convection-diffusion stencil,
     * but not by the div(u*)/pressure-gradient path, which only reads orthogonal buffers.
     *
     * In some cases, the domain might not use the updated buffer/corner at this step, but that's okay.
     * To avoid programming complexity, we won't determine whether to update based on whether it will be used.
     * We'll simply determine it based on the connection conditions.
     */
    void diag_shared_boundary_update();

    /**
     * Euler convection and diffusion term inner calculation.
     *
     * Step 4 of PISO.
     *
     * It is independent of the buffer/corner.
     */
    void euler_conv_diff_inner();

    /**
     * Euler convection and diffusion term outer calculation.
     *
     * Step 5 of PISO.
     *
     * It is dependent of the buffer/corner.
     */
    void euler_conv_diff_outer();

    void velocity_div_inner();
    void velocity_div_outer();
    void pressure_buffer_update();
    void add_pressure_gradient();

    /**
     * Compute raw MAC vorticity without first interpolating velocity to cell centers.
     *
     * The result is stored on the domain's lower-left node stencil with the same
     * array shape as a center field (nx by ny):
     *   omega(i,j) = [v(i,j)-v(i-1,j)]/hx - [u(i,j)-u(i,j-1)]/hy.
     * Boundary values are taken from the synchronized x-negative and y-negative
     * velocity buffers. The output variable must share the same geometry and use
     * center-sized fields on each domain.
     */
    void raw_vorticity_update(Variable2D& vorticity_var) const;
    static void raw_vorticity_update(const Variable2D& u_var,
                                     const Variable2D& v_var,
                                     Variable2D&       vorticity_var);

private:
    void neumann_corner_closure_update();

protected:
    std::vector<Domain2DUniform*>                                                            domains;
    std::unordered_map<Domain2DUniform*, std::unordered_map<LocationType, Domain2DUniform*>> adjacency;

    std::unordered_map<Domain2DUniform*, field2*> u_field_map, v_field_map, p_field_map;
    std::unordered_map<Domain2DUniform*, std::unordered_map<LocationType, double*>> u_buffer_map, v_buffer_map,
        p_buffer_map;
    std::unordered_map<Domain2DUniform*, field2*> u_temp_field_map, v_temp_field_map;

    // Non-Newtonian field maps
    std::unordered_map<Domain2DUniform*, field2*> mu_field_map;
    std::unordered_map<Domain2DUniform*, field2*> tau_xx_field_map;
    std::unordered_map<Domain2DUniform*, field2*> tau_yy_field_map;
    std::unordered_map<Domain2DUniform*, field2*> tau_xy_field_map;

    std::unordered_map<Domain2DUniform*, std::unordered_map<LocationType, double*>> tau_xx_buffer_map,
        tau_yy_buffer_map;

    std::unordered_map<Domain2DUniform*, double>& xneg_ypos_corner_map;
    std::unordered_map<Domain2DUniform*, double>& xpos_yneg_corner_map;

    std::unique_ptr<MHDModule2D> mhd_module;

    double dt;
    int    num_it;
    int    corr_it;
    double nu;
};
