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
x_ss = zeros(n, 1);
x_ss(1:3) = 4;

% Steady-state input
u_ss = -((B'*B)\B')*A*x_ss;

%% Closed-loop system with LQR gains
A_cl = Ts*(A - B*K) + eye(n);
B_cl = -Ts*(A - B*K);

%% Simulation of response

% Constants
m = 0.063;
g = 9.8;

% Setup
t = 0;
T = 2;
t_out = 0:Ts:T;
N = length(t_out);
x_out = zeros(N, n);

% Sphere parameters
r = 1;                % radius
center = [2, 2, 3];   % (x, y, z)

% Iterate through discrete time
for i = 1:(N-1)
    % Basline control input via LQR
    u = -K*(x_out(i, :)' - x_ss);
    u(1) = u(1) + m*g;
    
    % States
    x_out(i + 1, :) = x_out(i, :) + Ts*f(x_out(i, :), u)';
    
    % Step time
    t = t + Ts;
end

% Get inputs
u = -K*(x_out' - x_ss);

%% 3D plot of drone trajectory

figure;

% Baseline controller
scatter3(x_out(:, 1), x_out(:, 2), x_out(:, 3), 10, 'b', 'filled');
grid on; view(45, 45); hold on;
xlim([0 5]); ylim([0 5]); zlim([0 5]);
axis equal;

% Generate sphere surface
[Xs, Ys, Zs] = sphere(50);
Xs = r * Xs + center(1);
Ys = r * Ys + center(2);
Zs = r * Zs + center(3);

% Plot the sphere
surf(Xs, Ys, Zs, 'FaceAlpha', 0.5, 'EdgeColor', 'none', 'FaceColor', [0 0.5 1]);

% Plot endpoints
scatter3(x_out(1,1), x_out(1,2), x_out(1,3), 80, 'g', 'filled'); % start
scatter3(x_out(end,1), x_out(end,2), x_out(end,3), 80, 'r', 'filled'); % end

% Plot settings
xlabel('$x~(\mathrm{m})$', 'Interpreter', 'latex');
ylabel('$y~(\mathrm{m})$', 'Interpreter', 'latex');
zlabel('$z~(\mathrm{m})$', 'Interpreter', 'latex');
title('\textbf{Drone trajectory}', 'Interpreter', 'latex');
legend({'Trajectory', 'Object', 'Start', 'End'}, 'Interpreter','latex', 'Location','best');

%% Plot control inputs

figure;

% Inputs
plot(t_out, u, 'LineWidth', 1.5);
grid on;

% Plot settings
xlabel('$t~(\mathrm{s})$', 'Interpreter', 'latex');
ylabel('$u$', 'Interpreter', 'latex');
title('\textbf{Control inputs}', 'Interpreter', 'latex');
legend({'$U_{\mathrm{coll}}$', '$U_{\phi}$', '$U_{\theta}$', '$U_{\psi}$'}, 'Interpreter', 'latex', 'Location', 'best');