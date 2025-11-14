function dP = drone_opt(X, k, r_drone, P_objects, R_objects, n_drones, n_objects)
    % CVX optimization
    cvx_begin quiet
        variables dP(3, n_drones)
        obj = 0;
        constraints = [];

        for i = 1:n_drones
            % Compute objective of drones
            dv = X(7:9, i, k) - dP(:, i);
            obj = obj + dv'*dv;
            
            % Drone position (x, y, z)
            P_i = X(1:3, i, k);

            % Append drone-to-object constraints
            for j = 1:n_objects
                % Object position (x, y, z)
                P_j = P_objects(:, j);
    
                % Control barrier function constraint
                d_ij = P_i - P_j;
                h_ij = d_ij'*d_ij - R_objects(j)^2;
                grad_h = 2*d_ij;
                alpha = h_ij^3;
                constraints = [constraints, grad_h'*dP(:, i) + alpha >= 0];
                % disp(['Drone ', num2str(i), ' near obstacle ', num2str(j)]);
            end
            
            % Append drone-to-drone constraints
            for j = (i+1):n_drones
                % Drone position (x, y, z)
                P_j = X(1:3, j, k);
    
                % Control barrier function constraint
                d_ij = P_i - P_j;
                h_ij = d_ij'*d_ij - r_drone^2;
                grad_h = 2*[d_ij; -d_ij];
                alpha = h_ij^3;
                dP_stack = [dP(:, i); dP(:, j)];
                constraints = [constraints, grad_h'*dP_stack + alpha>= 0];
                % disp(['Drone ', num2str(i), ' near drone ', num2str(j)]);
            end
        end
        
        % fprintf('\n');

        minimize(obj)
        subject to
            constraints;
    cvx_end
end