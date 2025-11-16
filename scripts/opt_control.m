clear; close all; clc;

%% Sampling rate
Ts = 0.02;

%% Linearized drone dynamics (continuous)
[A, B, Ad, Bd] = linearized_drone(Ts);

%% Dimensions
[n, m] = size(B); % States (dim n), inputs (dim m)

%% LQR Design

% Cost matrices
R = diag([1/15; 1000; 1000; 100]);
Q = diag([0.1; 0.1; 10; 0.01; 0.01; 0.01; 0.01; 0.01; 1; 0.1; 0.1; 0.1]);

% LQR gains
[K, ~, ~] = lqrd(A, B, Q, R, Ts);

%% Scenario Setup

% Simulation variables
H = 5;             % prediction horizon
n_drones  = 2;     % number of drones
n_objects = 2;     % number of obstacles

% Constants
m_drone = 0.063; % kg
I_x = 0.5829e-4; % kgm^2
I_y = 0.7169e-4; % kgm^2
I_z = 1.000e-4;  % kgm^2
g = 9.8;         % m/s^2

% Initialize states and control inputs
X = zeros(n, n_drones, H);
U = zeros(m, n_drones, H);

% Initial conditions (x0, y0, z0)
% X(:, 1, 1) = [-2; 0.5; 0.5; zeros(n - 3, 1)];
% X(:, 2, 1) = [-2; -0.5; -0.5; zeros(n - 3, 1)];
X(:, 1, 1) = [0; 0; 0; zeros(n - 3, 1)];
X(:, 2, 1) = [4; 0; 0; zeros(n - 3, 1)];

% Goal position for each drone (xf, yf, zf)
X_goal = zeros(n, n_drones);
% X_goal(1:3, 1) = [2; -0.5; -0.5];   % drone 1 target
% X_goal(1:3, 2) = [2; 0.5; 0.5];     % drone 2 target
X_goal(1:3, 1) = [4; 4; 4];   % drone 1 target
X_goal(1:3, 2) = [0; 2; 3];   % drone 2 target

% Unit vector for shortest path of each drone
s = zeros(n, n_drones);

% Reference position
X_ref = zeros(n, n_drones);

% Setup
d = 0.75;                  % waypoint distance
for i = 1:n_drones
    p0 = X(1:3, i, 1);     % initial position of drone i
    pf = X_goal(1:3, i);   % goal position of drone i
    v  = pf - p0;          % direction vector
    s(1:3, i) = v/norm(v); % normalized direction

    X_ref(1:3, i) = p0;
end

% Safety radii
r_safe = 0.6;

% Obstacle positions and radii
% P_objects = [[0.3; -0.25; 0.25], [0.3; 0.25; -0.5]]; % Array of column vectors (x, y, z)
% R_objects = [0.3, 0.3]; % Radii of objects
P_objects = [[2; 2; 3], [4; 0.5; 2]]; % Array of column vectors (x, y, z)
R_objects = [1, 0.8]; % Radii of objects

% Data log
x_log = cell(n_drones, 1);
u_log = cell(n_drones, 1);
u_lqr_log = cell(n_drones, 1);
for i = 1:n_drones
    x_log{i} = X(:, i, 1);
    u_log{i} = U(:, i, 1);
    u_lqr_log{i} = U(:, i, 1);
end

%% Simulation of response
iter = 0;
waypoint = true;

while norm(X(:, 1, 1) - X_goal(:, 1)) > 0.01 || norm(X(:, 2, 1) - X_goal(:, 2)) > 0.01
    % Compute waypoint
    if mod(iter, 30) == 0 && waypoint
        for i = 1:n_drones
            X_ref(:, i) = X_ref(:, i) + d*s(:, i);

            if norm(X_ref(:, i) - X_goal(:, i)) < d
                X_ref(:, i) = X_goal(:, i);
            end
        end

        if norm(X_ref(:, i) - X_goal(:, i)) == 0
            waypoint = false;
        end
    end
    
    %% Predict linearized state-dynamics for each drone using baseline controller (LQR)
    for i = 1:n_drones
        for j = 1:(H-1)
            % Basline control input via LQR
            U(:, i, j) = -K*(X(:, i, j) - X_ref(:, i));
            % U(1, i, j) = U(1, i, j) + m_drone*g;
            % Linearized state dynamics
            X(:, i, j + 1) = Ad*X(:, i, j) + Bd*U(:, i, j);
        end
        u_lqr_log{i} = [u_lqr_log{i}, U(:, i, 1)];
    end
    
    %% Collision and obstacle detection
    active = false;

    for i = 1:n_drones
        % Drone i position (x, y, z)
        P_i = X(1:3, i, end);
        
        % Drone-to-object check
        for j = 1:n_objects
            % Object j position (x, y, z)
            P_j = P_objects(:, j);

            % Distance check
            d_ij = norm(P_i - P_j);
            if d_ij <= r_safe
                active = true;
                % disp(['Drone ', num2str(i), ' near obstacle ', num2str(j), ' (distance = ', num2str(d_ij, '%.2f'), ')']);
                break;
            end
        end
        
        if active; break; end

        % Drone-to-drone check
        for j = (i+1):n_drones
            % Drone j position (x, y, z)
            P_j = X(1:3, j, end);

            % Distance check
            d_ij = norm(P_i - P_j);
            if d_ij <= r_safe
                active = true;
                % disp(['Drone ', num2str(i), ' near drone ', num2str(j), ' (distance = ', num2str(d_ij, '%.2f'), ')']);
                break;
            end
        end

        if active; break; end
    end

    %% Apply optimization flag true
    if active
        % disp('Optimization active');
        
        % Optimization
        k = 1;
        dP_k1 = drone_opt(X, k + 1, r_safe, P_objects, R_objects, n_drones, n_objects); % 1-step ahead for z velocity
        dP_k3 = drone_opt(X, k + 3, r_safe, P_objects, R_objects, n_drones, n_objects); % 3-step ahead for x, y velocity

        for i = 1:n_drones
            %% Approach 1
            % %% 4-step ahead approach for x, y
            % 
            % % Extract x, y velocities
            % C = zeros(2, 12);
            % C(1, 7) = 1;
            % C(2, 8) = 1;
            % 
            % % Compute difference in nominal and optimal input
            % r = 3;
            % M = C*Ad^(r - 1)*Bd;
            % dv = dP_k3(1:2, i, end) - X(7:8, i, k + r);
            % du = pinv(M)*dv;
            % 
            % %% 2-step ahead approach for z
            % 
            % % Extract z velocities
            % C = zeros(1, 12);
            % C(1, 9) = 1;
            % 
            % % Compute difference in nominal and optimal input
            % r = 1;
            % M = C*Ad^(r - 1)*Bd;
            % dv = dP_k1(3, i, end) - X(9, i, k + r);
            % du = du + pinv(M)*dv;
            % du(1) = du(1) + m_drone*g;
            
            %% Approach 2
            % G = diag([Ts/m, -g*Ts^3/I_x, g*Ts^3/I_y, 0]);
            % G = diag([Ts/m, 0, 0, 0]); % This worked better
            % dv = [dP_k1(3, i); flipud(dP_k3(1:2, i)); 0];
            % du = G*dv;

            % G = diag([Ts/m, -g*Ts^3/I_x, g*Ts^3/I_y, 0]);
            % disp(G*dv);
            
            %% Approach 3

            % 4-step ahead approach for x, y

            % Extract x, y velocities
            C = zeros(1, 12);
            C(1, 7) = 1;
            C(2, 8) = 1;

            % Back-solve for [~, U_phi, U_theta, ~]
            r = 4;
            k = 1;
            sums = Ad^(r - 1)*(Ad - eye(n))*X(:, i, k);
            for k = 1:(r-1)
                sums = sums + Ad^(r - 1 - k)*Bd*U(:, i, k);
            end
            for k = 1:(r-2)
                sums = sums - Ad^(r - 2 - k)*Bd*U(:, i, k);
            end
            M = C*Ad^(r - 2)*(Ad - eye(n))*Bd;
            u = pinv(M)*(Ts*dP_k3(1:2, i) - C*sums);
            
            % 2-step ahead approach for z
            dv = (dP_k3(3, i) - X(9, i, 1))/Ts;
            u(1) = m_drone*(dv + g);
            
            % Update input for U_phi
            u(4) = U(4, i, 1);

            %% Compute dynamics

            %% If using non-linear dynamics
            % u0 = U(:, i, k) + du; % Approach 2
            u0 = u;
            x0 = X(:, i, 1);
            x0 = x0 + Ts*f(x0, u0);
            % disp(u0);
            
            %% If using linearized dynamics
            % u0 = U(:, i, k) + du; % Approach 2
            % u0(1) = u0(1) - m_drone*g;
            % x0 = X(:, i, 1);
            % x0 = Ad*x0 + Bd*u0;
    
            %% Update variables and log trajectory
            X(:, i, 1) = x0;
            x_log{i} = [x_log{i}, x0];
            u_log{i} = [u_log{i}, u0];
        end
    else
        % disp('LQR active');

        for i = 1:n_drones
            %% If using non-linear dynamics
            % u0 = U(:, i, 1);
            % x0 = X(:, i, 1);
            % x0 = x0 + Ts*f(x0, u0);
            % disp(u0);

            %% If using linearized dynamics
            x0 = X(:, i, 2);
            u0 = U(:, i, 1);
            % disp(u0);
            
            %% Update variables and log trajectory
            X(:, i, 1) = x0;
            x_log{i} = [x_log{i}, x0];
            u_log{i} = [u_log{i}, u0];
        end
    end

    for i = 1:n_drones
        % Drone i position (x, y, z)
        P_i = X(1:3, i, 1);
        
        % Drone-to-object check
        for j = 1:n_objects
            % Object j position (x, y, z)
            P_j = P_objects(:, j);

            % Distance check
            d_ij = norm(P_i - P_j);
            if d_ij <= r_safe
                disp(['Drone ', num2str(i), ' near obstacle ', num2str(j), '. Constraint violated.']);
            end
        end

        % Drone-to-drone check
        for j = (i+1):n_drones
            % Drone j position (x, y, z)
            P_j = X(1:3, j, 1);

            % Distance check
            d_ij = norm(P_i - P_j);
            if d_ij <= r_safe
                disp(['Drone ', num2str(i), ' near drone ', num2str(j), '. Constraint violated.']);
            end
        end
    end

    %% Increment iteration
    disp(['iter ',num2str(iter), ' done'])
    iter = iter + 1;
end

%% Plot results

% 3D plot settings
figure; hold on; grid on; view(45, 45);
xlabel('$x$ (m)','Interpreter','latex');
ylabel('$y$ (m)','Interpreter','latex');
zlabel('$z$ (m)','Interpreter','latex');
title('\textbf{Multi-Drone trajectories with obstacle avoidance}', 'Interpreter','latex');
axis equal;

% 3D trajectory plot
colors = lines(n_drones);
for i = 1:n_drones
    xd = x_log{i};
    scatter3(xd(1, :), xd(2, :), xd(3, :), 20, colors(i, :), 'filled');
end

% Plot obstacles
for j = 1:n_objects
    [Xs, Ys, Zs] = sphere(30);
    Xs = R_objects(j)*Xs + P_objects(1, j);
    Ys = R_objects(j)*Ys + P_objects(2, j);
    Zs = R_objects(j)*Zs + P_objects(3, j);
    surf(Xs, Ys, Zs, 'FaceAlpha', 0.4, 'EdgeColor', 'none', 'FaceColor', [0 0.5 1]);
end

legend(arrayfun(@(d) sprintf('Drone %d',d), 1:n_drones, 'UniformOutput',false));

%% plot controls

time = (0:iter-1) * Ts;
colors = lines(n_drones);

for u_row = 1:m 
    figure; hold on; grid on;
    
    for i = 1:n_drones
        % Solid = applied control (from u_log)
        plot(time(1:200), u_log{i}(u_row, 1:200), ...
             'LineWidth', 1.8, 'Color', colors(i,:));

        % Dashed = LQR baseline
        plot(time(1:200), u_lqr_log{i}(u_row, 1:200), ...
             '--', 'LineWidth', 1.8, 'Color', colors(i,:));
    end
    
    % --- Legend ---
    legend_entries = cell(1, 2*n_drones);
    for i = 1:n_drones
        legend_entries{2*i-1} = sprintf('Drone %d - applied', i);
        legend_entries{2*i}   = sprintf('Drone %d - LQR', i);
    end
    legend(legend_entries, 'Location', 'bestoutside');

    % --- Labels and title ---
    xlabel('Time [s]');
    ylabel(sprintf('$U_{%d}(t)$', u_row), 'Interpreter', 'latex');
    title(sprintf('Control Input $U_{%d}(t)$ per Drone', u_row), 'Interpreter', 'latex');
end
