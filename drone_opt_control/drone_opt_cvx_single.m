function dP_sol = drone_opt_cvx_single(X, H, r_drone, r_safe, P_objects, R_objects, n_drones, n_objects, Ts)
    dP_sol = zeros(3, n_drones, H-1);

    for k = 2:H
        cvx_begin quiet
            variables dP(3, n_drones)
            obj = 0;
            constraints = [];
            
            for i = 1:n_drones
                %% Compute objective of drones
                dv = X(7:9, i, k) - dP(:, i);
                obj = obj + dv' * dv;
                
                % Drone position
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
    
                        constraints = [constraints, grad_h' * dP(:, i) + alpha >= 0];
                    end
                end
    
                %% Drone-to-drone constraints
                for j = (i+1):n_drones
                    % Drone position
                    P_j = X(1:3, j, k);
    
                    for k = 2:H
                        % Control barrier function
                        d_ij = P_i - P_j;
                        h_ij = d_ij' * d_ij - r_drone^2;
                        grad_h = 2 * [d_ij; -d_ij];
                        dP_stack = [dP(:, i); dP(:, j)];
                        alpha = 100 * h_ij^3;
    
                        constraints = [constraints, grad_h' * dP_stack + alpha >= 0];
                    end
                end
            end
    
            minimize(obj)
            subject to
                constraints;
        cvx_end

        dP_sol(:, :, k-1) = dP;
    end
end