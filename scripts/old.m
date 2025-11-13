clear; close all; clc;

%% Linearized drone dynamics (continuous)
[A, B, ~, ~] = linearized_drone();

%% Dimensions
[n, m] = size(B); % State (dim n), input (dim m)

%% LQR Design
Ts = 0.01; % Sampling rate

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
n_drones  = 1;     % number of drones
n_objects = 1;     % number of obstacles

% Constants
m_drone = 0.063; % kg
I_x = 0.5829e-4; % kgm^2
I_y = 0.7169e-4; % kgm^2
I_z = 1.000e-4; % kgm^2
g = 9.8; % m/s^2

% Initialize states and control inputs
x_all = zeros(n, n_drones, H);
u_all = zeros(m, n_drones, H);

% Initial conditions (x0, y0, z0)
x_all(:, 1, 1) = [0; 0; 0; zeros(n - 3, 1)];
x_all(:, 2, 1) = [4; 0; 0; zeros(n - 3, 1)];

% Reference position for each drone (xf, yf, zf)
xref_all = zeros(n, n_drones);
xref_all(1:3, 1) = [4; 4; 4];   % drone 1 target
xref_all(1:3, 2) = [0; 2; 3];   % drone 2 target

% Drone radius
r_drone = 0.75;

% Obstacle positions and radii
z_objects = [[2; 2; 3], [4; 0.5; 2]]; % Array of column vectors (x, y, z)
r_objects = [1, 0.8];

% Data log
x_log = cell(n_drones, 1);
for d = 1:n_drones
    x_log{d} = x_all(:, d, 1);
end

%% Simulation of response
while t <= T
    % Predict state-dynamics using baseline LQR controller
    for d = 1:n_drones
        for k = 1:(H-1)
            % Non-linear dynamics
            x_all(:, d, k + 1) = x_all(:, d, k) + Ts*f(x_all(:, d, k), u_all(:, d, k));

            % Basline control input via LQR
            u_all(:, d, k + 1) = -K*(x_all(:, d, k + 1) - xref_all(:, d));
            u_all(1, d, k + 1) = u_all(1, d, k + 1) + m_drone*g;
        end
    end

    % Collision and obstacle checking
    for d = 1:n_drones
        z_curr = x_all(1:3, d, end);
        active_obj = false;
        active_drone = false;

        % Check distance to obstacles
        for j = 1:n_objects
            dist_obj = norm(z_curr - z_objects(:, j));
            if dist_obj <= r_objects(j)
                active_obj = true;
                disp(['Drone ', num2str(d), ' near obstacle ', num2str(j), ' (distance = ', num2str(dist_obj, '%.2f'), ')']);
                break;
            end
        end

        % Check distance to other drones
        for other = 1:n_drones
            if other ~= d
                z_other = x_all(1:3, other, end);
                dist_dd = norm(z_curr - z_other);
                if dist_dd <= r_drone % collision radius
                    active_drone = true;
                    disp(['Drone ', num2str(d), ' near drone ', num2str(other), ' (distance = ', num2str(dist_dd, '%.2f'), ')']);
                    break;
                end
            end
        end

        % Apply MPC if close to obstacle or another drone
        if active_obj || active_drone
            disp(['Drone ', num2str(d), ' → MPC active']);

            % Nominal predicted states
            z_nom  = reshape(x_all(1:3, d, :), 3, H);
            dz_nom = reshape(x_all(7:9, d, :), 3, H);

            % CVX optimization
            cvx_begin quiet
                variables dz(3, H) dz_all(3, n_drones, H)
                obj = 0;
                constraints = [];

                for k = 1:H
                    % Track nominal velocity
                    diff_vel = dz_nom(:, k) - dz(:, k);
                    obj = obj + diff_vel'*diff_vel;

                    % Obstacle avoidance
                    if active_obj
                        for j = 1:n_objects
                            dist_obj = z_nom(:, k) - z_objects(:, j);
                            h_ij = dist_obj'*dist_obj - r_objects(j)^2;
                            grad_hz = 2*dist_obj;
                            alpha = h_ij^3;
                            constraints = [constraints, grad_hz'*dz(:, k) + alpha >= 0];
                        end
                    end
                    
                    % Inter-drone avoidance
                    if active_obj
                        for other = 1:n_drones
                            if other ~= d
                                z_other = x_all(1:3, other, k);
                                dz_other = x_all(7:9, other, k);
                                dist_dd = z_nom(:, k) - z_other;
                                h_ij = dist_dd'*dist_dd - r_drone^2;
                                grad_hz = 2*[dist_dd; -dist_dd];
                                dz_stack = [dz(:, k); dz_other];
                                alpha = h_ij^3;
                                constraints = [constraints, grad_hz'*dz_stack + alpha >= 0];
                            end
                        end
                    end
                end

                minimize(obj)
                subject to
                    constraints;
            cvx_end

            % Apply optimized velocity
            x_all(7:9, d, 2:end) = dz(:, 2:end);
            
            % Finite differences to sovle for optimal input
            % zddot
            S = squeeze(x_all(3, d, 1:3));
            Sp = squeeze(x_all(9, d, 1:3));
            U_coll = (ndiff2(S, Sp, Ts) + g)*m;
            disp(ndiff2(S, Sp, Ts));
            
            % yddot
            S = squeeze(x_all(2, d, :));
            Sp = squeeze(x_all(8, d, :));
            U_phi = ndiff4(S, Sp, Ts)*(-I_x/g);

            % xddot
            S = squeeze(x_all(1, d, :));
            Sp = squeeze(x_all(7, d, :));
            U_theta = ndiff4(S, Sp, Ts)*(I_y/g);

            % psidot
            S = squeeze(x_all(6, d, 1:3));
            Sp = squeeze(x_all(12, d, 1:3));
            U_psi = ndiff2(S, Sp, Ts)*(I_z);

            % Compute dynamics
            u_new = [U_coll; U_phi; U_theta; U_psi];
            x_new = x_all(:, d, 1) + Ts*f(x_all(:, d, 1), u_new);
        else
            disp(['Drone ', num2str(d), ' → LQR only']);
            u_new = u_all(:, d, 2);
            x_new = x_all(:, d, 2);
        end
        
        x_all(:, d, 1) = x_new;
        u_all(:, d, 1) = u_new;

        % Log trajectory
        x_log{d} = [x_log{d}, x_new];
    end

    t = t + Ts;
end

%% Plot results

% 3D plot settings
figure; hold on; grid on; view(45,45);
xlim([0 5]); ylim([0 5]); zlim([0 5]);
xlabel('$x$ (m)','Interpreter','latex');
ylabel('$y$ (m)','Interpreter','latex');
zlabel('$z$ (m)','Interpreter','latex');
title('\textbf{Multi-Drone trajectories with obstacle avoidance}', 'Interpreter','latex');
axis equal;

% 3D trajectory plot
colors = lines(n_drones);
for d = 1:n_drones
    xd = x_log{d};
    scatter3(xd(1, :), xd(2, :), xd(3, :), 20, colors(d, :), 'filled');
end

% Plot obstacles
for j = 1:n_objects
    [Xs, Ys, Zs] = sphere(30);
    Xs = 0.75*r_objects(j)*Xs + z_objects(1, j);
    Ys = 0.75*r_objects(j)*Ys + z_objects(2, j);
    Zs = 0.75*r_objects(j)*Zs + z_objects(3, j);
    surf(Xs, Ys, Zs, 'FaceAlpha', 0.4, 'EdgeColor', 'none', 'FaceColor', [0 0.5 1]);
end

legend(arrayfun(@(d) sprintf('Drone %d',d), 1:n_drones, 'UniformOutput',false));