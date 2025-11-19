function dP = drone_opt(X, H, r_drone, r_safe, P_objects, R_objects, n_drones, n_objects, Ts)
    % CVX optimization
    cvx_begin quiet
        variables dP(3, n_drones, H)
        obj = 0;
        constraints = [];

        for i = 1:n_drones
            for k = 1:H
                % Compute objective of drones
                dv = X(7:9, i, k) - dP(:, i, k);
                obj = obj + dv'*dv;
                
                % Optimal drone position (x, y, z)
                if k > 2
                    P_i = P_i + Ts*dP(:, i, k);
                else
                    P_i = X(1:3, i, k + 1);
                end

                % % Drone position (x, y, z)
                % P_i = X(1:3, i, k);
    
                % Append drone-to-object constraints
                for j = 1:n_objects
                    % Object position (x, y, z)
                    P_j = P_objects(:, j);
        
                    % Control barrier function constraint
                    d_ij = P_i - P_j;
                    h_ij = d_ij'*d_ij - (R_objects(j) + r_safe)^2;
                    grad_h = 2*d_ij;
                    alpha = 100*pow_p(h_ij, 3);
                    constraints = [constraints, grad_h'*dP(:, i, k) + alpha >= 0];
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
                    alpha = 100*pow_p(h_ij, 3);
                    dP_stack = [dP(:, i, k); dP(:, j, k)];
                    constraints = [constraints, grad_h'*dP_stack + alpha>= 0];
                    % disp(['Drone ', num2str(i), ' near drone ', num2str(j)]);
                end
            end
        end
        
        % fprintf('\n');

        minimize(obj)
        subject to
            constraints;
    cvx_end
end