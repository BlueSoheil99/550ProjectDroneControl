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
t = 0;             % time variable
T = 4;             % total simulation time
H = 5;             % prediction horizon
n_drones  = 2;     % number of drones
n_objects = 2;     % number of obstacles

% Constants
m_drone = 0.063; % kg
I_x = 0.5829e-4; % kgm^2
I_y = 0.7169e-4; % kgm^2
I_z = 1.000e-4; % kgm^2
g = 9.8; % m/s^2

% Initialize states and control inputs
X = zeros(n, n_drones, H);
U = zeros(m, n_drones, H);

% Initial conditions (x0, y0, z0)
X(:, 1, 1) = [-2; 0.5; 0.5; zeros(n - 3, 1)];
X(:, 2, 1) = [-2; -0.5; -0.5; zeros(n - 3, 1)];

% Reference position for each drone (xf, yf, zf)
X_ref = zeros(n, n_drones);
X_ref(1:3, 1) = [2; -0.5; -0.5];   % drone 1 target
X_ref(1:3, 2) = [2; 0.5; 0.5];   % drone 2 target

% Drone radius
r_drone = 0.6;

% Obstacle positions and radii
P_objects = [[0.3; -0.25; 0.25], [0.3; 0.25; -0.5]]; % Array of column vectors (x, y, z)
r_objects = [0.3;0.3];

% Data log
x_log = cell(n_drones, 1);
for d = 1:n_drones
    x_log{d} = X(:, d, 1);
end

%% Simulation of response
while t <= T
    % Predict linearized state-dynamics for each drone using baseline controller (LQR)
    for d = 1:n_drones
        for k = 1:(H-1)
            % Basline control input via LQR
            U(:, d, k) = -K*(X(:, d, k) - X_ref(:, d));
            U(1, d, k) = U(1, d, k) + m_drone*g;
    
            % Linearized dynamics
            X(:, d, k+1) = Ad*X(:, d, k) + Bd*U(:, d, k);
        end
    end

    %% Apply optimization
    for i = 1:n_drones
        %% Optimize velocity at k + 1 (2-step ahead for z)
        k = 1;
        dP = drone_opt(X, k + 1, r_drone, r_objects, P_objects, n_drones, n_objects);

        % Apply finite differences and linearized dynamics to find U_coll (related to z)
        zdot = X(9, i, :);
        zddot = (dP(3, i) - zdot(k))/Ts;
        U_coll = (zddot + g)*m_drone;
        
        %% Optimize velocity at k + 3 (4-step ahead for x, y)
        k = 1;
        dP = drone_opt(X, k + 3, r_drone, r_objects, P_objects, n_drones, n_objects);
        
        % Apply finite differences and linearized dynamics to find U_phi (related to y)
        ydot = X(8, i, :);
        yddot = (-ydot(k) + 3*ydot(k + 1) - 3*ydot(k + 2) + dP(2, i))/Ts^3;
        U_phi = yddot*(-I_x/g);
        
        % Apply finite differences and linearized dynamics to find U_theta (related to x)
        xdot = X(7, i, :);
        xddot = (-xdot(k) + 3*xdot(k + 1) - 3*xdot(k + 2) + dP(1, i))/Ts^3;
        U_theta = xddot*(I_y/g);
        
        %% Two-step ahead

        % Apply finite differences and linearized dynamics to find U_psi (related to psi)
        psidot = X(12, i, :);
        psiddot = (psidot(k + 1) - psidot(k))/Ts;
        U_psi = psiddot*(I_z);
        
        %% Compute dynamics
        u0 = [U_coll; U_phi; U_theta; U_psi];
        % x0 = X(:, i, 1) + Ts*f(X(:, i, 1), u0);
        x0 = Ad*X(:, i, 1) + Bd*u0;

        % Update variables
        X(:, i, 1) = x0;
        disp(u0)

        % Log trajectory
        x_log{i} = [x_log{i}, x0];
    end
    
    % Increment time
    t = t + Ts;
end

%% Plot results

% 3D plot settings
figure; hold on;
xlabel('$x$ (m)','Interpreter','latex');
ylabel('$y$ (m)','Interpreter','latex');
zlabel('$z$ (m)','Interpreter','latex');
title('\textbf{Multi-Drone trajectories with obstacle avoidance}', 'Interpreter','latex');
grid on;
view([-50, 15]);
axis equal
daspect([1 1 1])
pbaspect([1 1 1])
xlim([-2.5 2.5]); ylim([-1 1]); zlim([-1 1]);


% 3D trajectory plot
colors = lines(n_drones);
for d = 1:n_drones
    xd = x_log{d};
    scatter3(xd(1, :), xd(2, :), xd(3, :), 10, colors(d, :), 'filled');
end

% --- Obstacles as translucent spheres ---
% [xs, ys, zs] = sphere(30);
% surf(r_objects.*xs + xo1(1), r_objects.*ys + xo1(2), r_objects.*zs + xo1(3), ...
%     'FaceColor', 'interp', 'FaceAlpha', 0.8, 'EdgeColor', 'none');
% surf(r_objects.*xs + xo2(1), r_objects.*ys + xo2(2), r_objects.*zs + xo2(3), ...
%     'FaceColor', 'interp', 'FaceAlpha', 0.8, 'EdgeColor', 'none');

% Plot obstacles
for j = 1:n_objects
    [Xs, Ys, Zs] = sphere(30);
    Xs = 0.75*r_objects(j)*Xs + P_objects(1, j);
    Ys = 0.75*r_objects(j)*Ys + P_objects(2, j);
    Zs = 0.75*r_objects(j)*Zs + P_objects(3, j);
    surf(Xs, Ys, Zs, 'FaceAlpha', 0.4, 'EdgeColor', 'none', 'FaceColor', [0 0.5 1]);
end

legend(arrayfun(@(d) sprintf('Drone %d',d), 1:n_drones, 'UniformOutput',false));