function dP_sol = drone_opt(X, H, r_drone, r_safe, P_objects, R_objects, n_drones, n_objects, Ts)
    import casadi.*

    opti = Opti(); % CasADi optimization object
    dP = opti.variable(3*n_drones, H - 1);

    obj = 0;

    for i = 1:n_drones
        % Compute objective of drones
        for k = 2:H
            dP_ik = dP(3*(i-1)+1:3*i, k-1);
            dv = X(7:9, i, k) - dP_ik;
            obj = obj + dv'*dv;
        end

        % Append drone-to-object constraints
        for j = 1:n_objects
            % Drone position (x, y, z)
            P_i = X(1:3, i, 1);

            % Object position (x, y, z)
            P_j = P_objects(:, j);

            for k = 2:H
                % Extract drone i control
                dP_ik = dP(3*(i-1)+1:3*i, k-1);

                % Update drone position
                P_i = P_i + Ts*dP_ik;

                % Control barrier function constraint
                d_ij = P_i - P_j;
                h_ij = d_ij'*d_ij - (R_objects(j) + r_safe)^2;
                grad_h = 2*d_ij;
                alpha = 100*h_ij^3;

                opti.subject_to(grad_h'*dP_ik + alpha >= 0);
            end
        end

        % Append drone-to-drone constraints
        for j = (i+1):n_drones
            % Drone position (x, y, z)
            P_i = X(1:3, i, 1);

            % Drone position (x, y, z)
            P_j = X(1:3, j, 1);

            for k = 2:H
                % Extract drone controls
                dP_ik = dP(3*(i-1)+1:3*i, k-1);
                dP_jk = dP(3*(j-1)+1:3*j, k-1);

                % Update drone positions
                P_i = P_i + Ts*dP_ik;
                P_j = P_j + Ts*dP_jk;

                % Control barrier function constraint
                d_ij = P_i - P_j;
                h_ij = d_ij'*d_ij - r_drone^2;
                grad_h = 2*[d_ij; -d_ij];

                dP_stack = [dP_ik; dP_jk];
                alpha = 100*h_ij^3;

                opti.subject_to(grad_h'*dP_stack + alpha >= 0);
            end
        end
    end

    % minimize(obj)
    opti.minimize(obj);

    % subject to constraints (implicit)
    opts.ipopt.print_level = 0;
    opts.print_time = false;

    opti.solver('ipopt', opts);

    sol = opti.solve();
    dP_flat = sol.value(dP);   % size = (3*n_drones) × (H-1)
    
    % reshape dP into original 3 x n_drones x (H-1)
    dP_sol = zeros(3, n_drones, H-1);
    
    for i = 1:n_drones
        rows = 3*(i-1)+1:3*i;
        for k = 1:(H-1)
            dP_sol(:, i, k) = dP_flat(rows, k);
        end
    end
end