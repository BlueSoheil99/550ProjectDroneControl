clear; close all; clc;

%% Linearized drone dynamics (continuous)
[A, B, ~, ~] = linearized_drone();

%% State-space dimensions
[n, m] = size(B);

%% Baseline controller using LQR

% Sampling rate
Ts = 0.01;

% Cost matrices
R = diag([1/15; 1000; 1000; 100]); 
Q = diag([0.1; 0.1; 10; 0.01; 0.01; 0.01; 0.01; 0.01; 1; 0.1; 0.1; 0.1]);

% LQR gains
[K, ~, ~] = lqrd(A, B, Q, R, Ts);

% Steady-state output
x_ref = zeros(n, 1);
x_ref(1:3) = 4;

% Steady-state input
u_ss = -((B'*B)\B')*A*x_ref;

%% Discrete drone dynamics
Ad = eye(n) + Ts*A;
Bd = Ts*B;

%% Simulation of response

% Time
t = 0;
T = 2;

% Horizon
H = 4;

% Data arrays
x = zeros(n, H);
u = zeros(m, H);
z_opt = zeros(n/4, 1);

% Initial conditions
x(:, 1) = [0; 0; 0; zeros(n - 3, 1)]; % (x0, y0, z0)
u(:, 1) = -K*(x(:, 1) - x_ref);       % u0

% Store data
X = x(:, 1);

% Object position
z_object = [2; 2; 3];
r = 1;

while t <= T
    for i = 1:(H - 1)
        x(:, i + 1) = Ad*x(:, i) + Bd*u(:, i);
        u(:, i + 1) = -K*(x(:, i + 1) - x_ref);
    end
    
    if (x(1:3, end) - z_object)'*(x(1:3, end) - z_object) - r^2 < 0
        % Nominal positions and velocities of drones
        z_nom = x(1:3, :);
        dz_nom = x(7:9, :);
        
        % Optimization
        cvx_begin quiet
            % Functions
            variables dz(n/4, H)
    
            % Objective
            obj = 0;
    
            % Constraints
            constraints = [];
    
            for i = 1:H
                % Norm
                diff_pos = dz_nom(:, i) - dz(:, i);
                
                % Compute cost (x, y, z) velocities with a horizon of 4
                % Consider seperate horizon for z?
                obj = obj + diff_pos'*diff_pos;
                
                % Object avoidance constraint
                diff_obj = z_nom(:, i) - z_object;
                h_i = diff_obj'*diff_obj - r^2;
                constraints = [constraints, 2*diff_obj'*dz(:, i) + h_i^3 >= 0];
            end
    
            % Minimize cost subject to constraints
            minimize(obj)
            subject to 
                constraints;
        cvx_end

        z_opt = dz(:, 2);
        disp("MPC")
    else
        z_opt = x(7:9, 2);
        disp("LQR")
    end

    % Update variables
    x(:, 1) = x(:, 2);
    x(7:9, 1) = z_opt;
    X = horzcat(X, x(:, 1));
    u(:, 1) = -K*(x(:, 1) - x_ref);
    t = t + Ts;
end

%% 3D plot of drone trajectory

figure;

% Baseline controller
scatter3(X(1, :), X(2, :), X(3, :), 10, 'b', 'filled');
grid on; view(45, 45); hold on;
xlim([0 5]); ylim([0 5]); zlim([0 5]);
axis equal;

% Generate sphere surface
[Xs, Ys, Zs] = sphere(50);
Xs = 0.5*r * Xs + z_object(1);
Ys = 0.5*r * Ys + z_object(2);
Zs = 0.5*r * Zs + z_object(3);

% Plot the sphere
surf(Xs, Ys, Zs, 'FaceAlpha', 0.5, 'EdgeColor', 'none', 'FaceColor', [0 0.5 1]);