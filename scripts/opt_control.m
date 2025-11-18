clear; close all; clc;

%% Sampling rate
Ts = 0.02;

%% Linearized drone dynamics
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
X(:, 1, 1) = [-2; 0.5; 0.5; zeros(n - 3, 1)];
X(:, 2, 1) = [-2; -0.5; -0.5; zeros(n - 3, 1)];
% X(:, 1, 1) = [0; 0; 0; zeros(n - 3, 1)];
% X(:, 2, 1) = [4; 0; 0; zeros(n - 3, 1)];

% Goal position for each drone (xf, yf, zf)
X_goal = zeros(n, n_drones);
X_goal(1:3, 1) = [2; -0.5; -0.5];   % drone 1 target
X_goal(1:3, 2) = [2; 0.5; 0.5];     % drone 2 target
% X_goal(1:3, 1) = [4; 4; 4];   % drone 1 target
% X_goal(1:3, 2) = [0; 2; 3];   % drone 2 target

% Safety radii
r_safe = 0.3;
r_drone = 0.6;

% Obstacle positions and radii
P_objects = [[0.3; -0.25; 0.25], [0.3; 0.25; -0.5]]; % Array of column vectors (x, y, z)
R_objects = [0.3, 0.3]; % Radii of objects
% P_objects = [[2; 2; 3], [4; 0.5; 2]]; % Array of column vectors (x, y, z)
% R_objects = [1, 0.8]; % Radii of objects

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
waypoint = true(n_drones, 1);

while any(vecnorm(X(:,:,1) - X_goal, 2, 1) > 0.01)
    % Compute waypoint
    if mod(iter, 30) == 0
        % Update waypoint
        X_ref(:, waypoint) = X_ref(:, waypoint) + d*s(:, waypoint);

        % Compute distance to goal
        e = X_ref - X_goal;
        dist = vecnorm(e, 2, 1);

        % Set reference if at goal
        terminal = dist < d;
        X_ref(:, terminal) = X_goal(:, terminal);

        % Update waypoint flag
        waypoint = ~terminal;
    end
    
    %% Predict linearized state-dynamics for each drone using baseline controller (LQR)
    for i = 1:n_drones
        for j = 1:(H-1)
            % Basline control input via LQR
            U(:, i, j) = -K*(X(:, i, j) - X_ref(:, i));

            % Linearized state dynamics
            X(:, i, j + 1) = Ad*X(:, i, j) + Bd*U(:, i, j);
            U(1, i, j) = U(1, i, j) + m_drone*g; % For completeness
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
            if d_ij <= (R_objects(j) + r_safe)
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
            if d_ij <= r_drone
                active = true;
                % disp(['Drone ', num2str(i), ' near drone ', num2str(j), ' (distance = ', num2str(d_ij, '%.2f'), ')']);
                break;
            end
        end

        if active; break; end
    end

    %% Apply optimization flag true
    if active
        disp('Optimization active');
        
        % Optimization
        k = 1;
        dP_k1 = drone_opt(X, k + 1, r_drone, r_safe, P_objects, R_objects, n_drones, n_objects); % 1-step ahead for z velocity
        dP_k3 = drone_opt(X, k + 3, r_drone, r_safe, P_objects, R_objects, n_drones, n_objects); % 3-step ahead for x, y velocity

        for i = 1:n_drones
            % Apply finite differences and linearized dynamics to find U_coll (related to z)
            zdot = X(9, i, :);
            zddot = (dP_k1(3, i, k + 1) - zdot(k))/Ts;
            U_coll = (zddot + g)*m_drone;
            
            % Apply finite differences and linearized dynamics to find U_phi (related to y)
            ydot = X(8, i, :);
            yddot = (-ydot(k) + 3*dP_k3(2, i, k + 1) - 3*dP_k3(2, i, k + 2) + dP_k3(2, i, k + 3))/Ts^3;
            U_phi = yddot*(-I_x/g);
            
            % Apply finite differences and linearized dynamics to find U_theta (related to x)
            xdot = X(7, i, :);
            xddot = (-xdot(k) + 3*dP_k3(1, i, k + 1) - 3*dP_k3(1, i, k + 2) + dP_k3(1, i, k + 3))/Ts^3;
            U_theta = xddot*(I_y/g);
    
            % Apply finite differences and linearized dynamics to find U_psi (related to psi)
            psidot = X(12, i, :);
            psiddot = (psidot(k + 1) - psidot(k))/Ts;
            U_psi = psiddot*(I_z);

            %% Compute non-linear dynamics for optimal control
            u0 = [U_coll; U_phi; U_theta; U_psi];
            x0 = X(:, i, 1);
            x0 = x0 + Ts*f(x0, u0);

            %% Update variables and log trajectory
            X(:, i, 1) = x0;
            x_log{i} = [x_log{i}, x0];
            u_log{i} = [u_log{i}, u0];
        end
    else
        disp('LQR active');

        for i = 1:n_drones
            %% Compute linearized dynamics for LQR
            x0 = X(:, i, 2);
            u0 = U(:, i, 1);
            
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
            if d_ij <= (R_objects(j) + r_safe)
                disp(['Drone ', num2str(i), ' near obstacle ', num2str(j), '. Constraint violated.']);
            end
        end

        % Drone-to-drone check
        for j = (i+1):n_drones
            % Drone j position (x, y, z)
            P_j = X(1:3, j, 1);

            % Distance check
            d_ij = norm(P_i - P_j);
            if d_ij <= r_drone
                disp(['Drone ', num2str(i), ' near drone ', num2str(j), '. Constraint violated.']);
            end
        end
    end

    %% Increment iteration
    % disp(['iter ',num2str(iter), ' done'])
    iter = iter + 1;
end

disp([num2str(iter), ' iterations'])

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