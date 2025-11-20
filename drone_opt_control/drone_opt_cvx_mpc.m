function dP = drone_opt_cvx_mpc(X, H, r_drone, r_safe, P_objects, R_objects, n_drones, n_objects, Ts)
    cvx_begin quiet
        variables dP(3, n_drones, H-1)
        obj = 0;
        constraints = [];
        
        for i = 1:n_drones
            %% Compute objective of drones
            for k = 2:H
                dv = X(7:9, i, k) - dP(:, i, k-1);
                obj = obj + dv' * dv;
            end
            
            % Drone initial position
            P_i = X(1:3, i, k);

            %% Drone-to-object constraints
            for j = 1:n_objects
                % Object position
                P_j = P_objects(:, j);

                for k = 2:H
                    % Control barrier function
                    d_ij = P_i - P_j;
                    h_ij = d_ij' * d_ij - (R_objects(j) + r_safe)^2;
                    grad_h = 2 * d_ij;
                    alpha = 100 * h_ij^3;

                    constraints = [constraints, grad_h' * dP(:, i, k-1) + alpha >= 0];
                end
            end

            %% Drone-to-drone constraints
            for j = (i+1):n_drones
                % Drone positions
                P_j = X(1:3, j, k);

                for k = 2:H
                    % Control barrier function
                    d_ij = P_i - P_j;
                    h_ij = d_ij' * d_ij - r_drone^2;
                    grad_h = 2 * [d_ij; -d_ij];
                    dP_stack = [dP(:, i, k-1); dP(:, j, k-1)];
                    alpha = 100 * h_ij^3;

                    constraints = [constraints, grad_h' * dP_stack + alpha >= 0];
                end
            end
        end

        minimize(obj)
        subject to
            constraints;
    cvx_end
end