#include "ns_solver2d.h"

#include "boundary_2d_utils.h"

void ConcatNSSolver2D::phys_boundary_update()
{
    for (auto& domain : domains)
    {
        field2& u = *u_field_map[domain];
        field2& v = *v_field_map[domain];

        int nx = u.get_nx();
        int ny = u.get_ny();

        {
            auto& bound_type_map    = u_var->boundary_type_map[domain];
            auto& has_bound_val_map = u_var->has_boundary_value_map[domain];
            auto& bound_val_map     = u_var->boundary_value_map[domain];
            auto& buffer_map        = u_buffer_map[domain];

            for (auto& [loc, type] : bound_type_map)
            {
                auto bound_val = has_bound_val_map[loc] ? bound_val_map[loc] : nullptr;

                if (loc == LocationType::XNegative)
                {
                    if (type == PDEBoundaryType::Dirichlet)
                        assign_x(u, 0, bound_val, 0.0);
                    else if (type == PDEBoundaryType::Neumann)
                        copy_x(u, 1, 0);
                    else if (type == PDEBoundaryType::Periodic)
                        copy_x(u, nx - 1, 0);
                }
                else if (loc == LocationType::XPositive)
                {
                    if (type == PDEBoundaryType::Dirichlet)
                        assign_val_to_buffer(buffer_map[loc], ny, bound_val, 0.0);
                    else if (type == PDEBoundaryType::Neumann)
                        copy_x_to_buffer(buffer_map[loc], u, nx - 1);
                    else if (type == PDEBoundaryType::Periodic)
                        copy_x_to_buffer(buffer_map[loc], u, 1);
                }
                else if (loc == LocationType::YNegative)
                {
                    if (type == PDEBoundaryType::Dirichlet)
                        mirror_y_to_buffer(buffer_map[loc], u, 0, bound_val, 0.0);
                    else if (type == PDEBoundaryType::Neumann)
                        copy_y_to_buffer(buffer_map[loc], u, 0);
                    else if (type == PDEBoundaryType::Periodic)
                        copy_y_to_buffer(buffer_map[loc], u, ny - 1);
                }
                else if (loc == LocationType::YPositive)
                {
                    if (type == PDEBoundaryType::Dirichlet)
                        mirror_y_to_buffer(buffer_map[loc], u, ny - 1, bound_val, 0.0);
                    else if (type == PDEBoundaryType::Neumann)
                        copy_y_to_buffer(buffer_map[loc], u, ny - 1);
                    else if (type == PDEBoundaryType::Periodic)
                        copy_y_to_buffer(buffer_map[loc], u, 0);
                }
            }
        }

        {
            auto& bound_type_map    = v_var->boundary_type_map[domain];
            auto& has_bound_val_map = v_var->has_boundary_value_map[domain];
            auto& bound_val_map     = v_var->boundary_value_map[domain];
            auto& buffer_map        = v_buffer_map[domain];

            for (auto& [loc, type] : bound_type_map)
            {
                auto bound_val = has_bound_val_map[loc] ? bound_val_map[loc] : nullptr;

                if (loc == LocationType::XNegative)
                {
                    if (type == PDEBoundaryType::Dirichlet)
                        mirror_x_to_buffer(buffer_map[loc], v, 0, bound_val, 0.0);
                    else if (type == PDEBoundaryType::Neumann)
                        copy_x_to_buffer(buffer_map[loc], v, 0);
                    else if (type == PDEBoundaryType::Periodic)
                        copy_x_to_buffer(buffer_map[loc], v, nx - 1);
                }
                else if (loc == LocationType::XPositive)
                {
                    if (type == PDEBoundaryType::Dirichlet)
                        mirror_x_to_buffer(buffer_map[loc], v, nx - 1, bound_val, 0.0);
                    else if (type == PDEBoundaryType::Neumann)
                        copy_x_to_buffer(buffer_map[loc], v, nx - 1);
                    else if (type == PDEBoundaryType::Periodic)
                        copy_x_to_buffer(buffer_map[loc], v, 0);
                }
                else if (loc == LocationType::YNegative)
                {
                    if (type == PDEBoundaryType::Dirichlet)
                        assign_y(v, 0, bound_val, 0.0);
                    else if (type == PDEBoundaryType::Neumann)
                        copy_y(v, 1, 0);
                    else if (type == PDEBoundaryType::Periodic)
                        copy_y(v, ny - 1, 0);
                }
                else if (loc == LocationType::YPositive)
                {
                    if (type == PDEBoundaryType::Dirichlet)
                        assign_val_to_buffer(buffer_map[loc], nx, bound_val, 0.0);
                    else if (type == PDEBoundaryType::Neumann)
                        copy_y_to_buffer(buffer_map[loc], v, ny - 1);
                    else if (type == PDEBoundaryType::Periodic)
                        copy_y_to_buffer(buffer_map[loc], v, 1);
                }
            }
        }
    }
}

void ConcatNSSolver2D::nondiag_shared_boundary_update()
{
    for (auto& domain : domains)
    {
        field2& u = *u_field_map[domain];
        field2& v = *v_field_map[domain];
        field2& p = *p_field_map[domain];

        int nx = u.get_nx();
        int ny = u.get_ny();

        for (auto& [loc, type] : u_var->boundary_type_map[domain])
        {
            // While u has adjacented boundary, it means v and p also have adjacented boundary
            if (type == PDEBoundaryType::Adjacented)
            {
                double* u_buffer = u_buffer_map[domain][loc];
                double* v_buffer = v_buffer_map[domain][loc];

                Domain2DUniform* adj_domain = adjacency[domain][loc];
                field2&          adj_u      = *u_field_map[adj_domain];
                field2&          adj_v      = *v_field_map[adj_domain];
                int              adj_nx     = adj_u.get_nx();
                int              adj_ny     = adj_u.get_ny();
                switch (loc)
                {
                    case LocationType::XNegative:
                        copy_x_to_buffer(u_buffer, adj_u, adj_nx - 1);
                        copy_x_to_buffer(v_buffer, adj_v, adj_nx - 1);
                        xneg_ypos_corner_map[domain] = v_buffer_map[adj_domain][LocationType::YPositive][adj_nx - 1];
                        break;
                    case LocationType::XPositive:
                        copy_x_to_buffer(u_buffer, adj_u, 0);
                        copy_x_to_buffer(v_buffer, adj_v, 0);
                        break;
                    case LocationType::YNegative:
                        copy_y_to_buffer(u_buffer, adj_u, adj_ny - 1);
                        copy_y_to_buffer(v_buffer, adj_v, adj_ny - 1);
                        xpos_yneg_corner_map[domain] = u_buffer_map[adj_domain][LocationType::XPositive][adj_ny - 1];
                        break;
                    case LocationType::YPositive:
                        copy_y_to_buffer(u_buffer, adj_u, 0);
                        copy_y_to_buffer(v_buffer, adj_v, 0);
                        break;
                    default:
                        throw std::runtime_error("ConcatNSSolver2D: invalid location type");
                }
            }
        }
    }
}

void ConcatNSSolver2D::diag_shared_boundary_update()
{
    for (auto& domain : domains)
    {
        field2& u = *u_field_map[domain];

        int nx = u.get_nx();
        int ny = u.get_ny();

        for (auto& [loc, type] : u_var->boundary_type_map[domain])
        {
            // While u has adjacented boundary, it means v and p also have adjacented boundary
            if (type == PDEBoundaryType::Adjacented)
            {
                Domain2DUniform* adj_domain         = adjacency[domain][loc];
                auto&            adj_bound_type_map = u_var->boundary_type_map[adj_domain];

                if (loc == LocationType::XNegative)
                {
                    if (adj_bound_type_map[LocationType::YNegative] == PDEBoundaryType::Adjacented)
                    {
                        Domain2DUniform* diag_domain  = adjacency[adj_domain][LocationType::YNegative];
                        double*          diag_buffer  = u_buffer_map[diag_domain][LocationType::XPositive];
                        double*          local_buffer = u_buffer_map[domain][LocationType::YNegative];

                        local_buffer[0] = diag_buffer[diag_domain->get_ny() - 1];
                    }
                    if (adj_bound_type_map[LocationType::YPositive] == PDEBoundaryType::Adjacented)
                    {
                        Domain2DUniform* diag_domain  = adjacency[adj_domain][LocationType::YPositive];
                        double*          diag_buffer  = u_buffer_map[diag_domain][LocationType::XPositive];
                        double*          local_buffer = u_buffer_map[domain][LocationType::YPositive];

                        local_buffer[0] = diag_buffer[0];
                    }
                }
                else if (loc == LocationType::XPositive)
                {
                    if (adj_bound_type_map[LocationType::YNegative] == PDEBoundaryType::Adjacented)
                    {
                        Domain2DUniform* diag_domain = adjacency[adj_domain][LocationType::YNegative];
                        auto&            diag_u      = *u_field_map[diag_domain];

                        xpos_yneg_corner_map[domain] = diag_u(0, diag_domain->get_ny() - 1);
                    }
                }
                else if (loc == LocationType::YNegative)
                {
                    if (adj_bound_type_map[LocationType::XNegative] == PDEBoundaryType::Adjacented)
                    {
                        Domain2DUniform* diag_domain  = adjacency[adj_domain][LocationType::XNegative];
                        double*          diag_buffer  = v_buffer_map[diag_domain][LocationType::YPositive];
                        double*          local_buffer = v_buffer_map[domain][LocationType::XNegative];

                        local_buffer[0] = diag_buffer[diag_domain->get_nx() - 1];
                    }
                    if (adj_bound_type_map[LocationType::XPositive] == PDEBoundaryType::Adjacented)
                    {
                        Domain2DUniform* diag_domain  = adjacency[adj_domain][LocationType::XPositive];
                        double*          diag_buffer  = v_buffer_map[diag_domain][LocationType::YPositive];
                        double*          local_buffer = v_buffer_map[domain][LocationType::XPositive];

                        local_buffer[0] = diag_buffer[0];
                    }
                }
                else if (loc == LocationType::YPositive)
                {
                    if (adj_bound_type_map[LocationType::XNegative] == PDEBoundaryType::Adjacented)
                    {
                        Domain2DUniform* diag_domain = adjacency[adj_domain][LocationType::XNegative];
                        auto&            diag_v      = *v_field_map[diag_domain];

                        xneg_ypos_corner_map[domain] = diag_v(diag_domain->get_nx() - 1, 0);
                    }
                }
            }
        }
    }

    neumann_corner_closure_update();
}

void ConcatNSSolver2D::neumann_corner_closure_update()
{
    // Close the two MAC corners that are not stored in the face fields.  A
    // zero-normal-gradient tangential velocity on a symmetry boundary must
    // copy the value already reconstructed on the orthogonal boundary.  This
    // also covers a physical symmetry boundary meeting a shared interface.
    for (auto& domain : domains)
    {
        const auto& u_types = u_var->boundary_type_map[domain];
        const auto& v_types = v_var->boundary_type_map[domain];

        if (v_types.at(LocationType::XNegative) == PDEBoundaryType::Neumann)
            xneg_ypos_corner_map[domain] = v_buffer_map[domain][LocationType::YPositive][0];

        if (u_types.at(LocationType::YNegative) == PDEBoundaryType::Neumann)
            xpos_yneg_corner_map[domain] = u_buffer_map[domain][LocationType::XPositive][0];
    }
}
