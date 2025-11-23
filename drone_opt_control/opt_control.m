clear; close all; clc;

%% Optimizer option
opts = ["cvx_single", "cvx_mpc", "casdi"];
opt  = "cvx_mpc";

%% Define drone optimizer
idx = -1;

for i = 1:length(opts)
    if opt == opts(i)
        idx = i;
    end
end

if idx ~= -1
    drone_opt = str2func("drone_opt_" + opts(idx));
end

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

% Goal position for each drone (xf, yf, zf)
X_goal = zeros(n, n_drones);
X_goal(1:3, 1) = [4; -0.5; -0.5];   % drone 1 target
X_goal(1:3, 2) = [4; 0.5; 0.5];     % drone 2 target

% Safety radii
r_safe = 0.0;
r_drone = 0.6;

% Obstacle positions and radii
P_objects = [[0.3; -0.25; 0.25], [0.3; 0.25; -0.5]]; % Array of column vectors (x, y, z)
R_objects = [0.3, 0.3, 0.3]; % Radii of objects

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
d_obj_log = cell(n_drones, n_objects);
d_drone_log = cell(n_drones, n_drones);

for i = 1:n_drones
    x_log{i} = X(:, i, 1);
    u_log{i} = U(:, i, 1);
    u_lqr_log{i} = U(:, i, 1);
end

%% Simulation of response
iter = 0;
waypoint = true(n_drones, 1);

%% Initialize termination flags
terminated = false(n_drones,1);

%% Main loop
while any(~terminated)
    disp(['Iteration: ', num2str(iter)])

    % Check per-drone termination
    for i = 1:n_drones
        if ~terminated(i)
            if vecnorm(X(:,i,1) - X_goal(:,i), 2, 1) < 0.01
                terminated(i) = true;
                disp(['Iteration ', num2str(iter), ': Drone ', num2str(i), ' has terminated.']);
            end
        end
    end

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
        if terminated(i), continue; end
        
        for j = 1:(H-1)
            % Basline control input via LQR
            U(:, i, j) = -K*(X(:, i, j) - X_ref(:, i));

            % Linearized state dynamics
            X(:, i, j + 1) = Ad*X(:, i, j) + Bd*U(:, i, j);
            
            if opt ~= "lqr"
                U(1, i, j) = U(1, i, j) + m_drone*g; % For completeness
            end
        end
        
        u_lqr_log{i} = [u_lqr_log{i}, U(:, i, 1)];
    end
    
    %% Optimization flag
    active = false;
    
    %% Collision and obstacle detection
    for i = 1:n_drones
        if terminated(i), continue; end
        
        % Drone i position (x, y, z)
        P_i = X(1:3, i, end);
        
        % Drone-to-object check
        for j = 1:n_objects
            P_j = P_objects(:, j);
            d_ij = norm(P_i - P_j);
            d_obj_log{i, j} = [d_obj_log{i, j}, d_ij];
            if d_ij <= (R_objects(j) + r_safe)
                active = true;
                % disp(['Drone ', num2str(i), ' near obstacle ', num2str(j), '. Constraint violated.']);
                break;
            end
        end
        
        if active, break; end

        % Drone-to-drone check
        for j = (i+1):n_drones
            if terminated(j), continue; end
            P_j = X(1:3, j, end);
            d_ij = norm(P_i - P_j);
            d_drone_log{i, j} = [d_drone_log{i, j}, d_ij];
            if d_ij <= r_drone
                active = true;
                % disp(['Drone ', num2str(i), ' near drone ', num2str(j), '. Constraint violated.']);
                break;
            end
        end

        if active, break; end
    end

    %% Apply optimization when needed
    if active && opt ~= "lqr"
        % disp('Optimization active');
        
        % Optimization
        k = 1;
        dP_k1 = drone_opt(X, k + 1, r_drone, r_safe, P_objects, R_objects, n_drones, n_objects, Ts); % 1-step ahead for z velocity
        dP_k3 = drone_opt(X, k + 3, r_drone, r_safe, P_objects, R_objects, n_drones, n_objects, Ts); % 3-step ahead for x, y velocity

        for i = 1:n_drones
            if terminated(i), continue; end
            k = 0;
            
            % Apply finite differences and linearized dynamics to find U_coll (related to z)
            zdot = X(9, i, :);
            zddot = (dP_k1(3, i, k + 1) - zdot(k + 1))/Ts;
            U_coll = (zddot + g)*m_drone;
            
            % Apply finite differences and linearized dynamics to find U_phi (related to y)
            ydot = X(8, i, :);
            yddot = (-ydot(k + 1) + 3*dP_k3(2, i, k + 1) - 3*dP_k3(2, i, k + 2) + dP_k3(2, i, k + 3))/Ts^3;
            U_phi = yddot*(-I_x/g);
            
            % Apply finite differences and linearized dynamics to find U_theta (related to x)
            xdot = X(7, i, :);
            xddot = (-xdot(k + 1) + 3*dP_k3(1, i, k + 1) - 3*dP_k3(1, i, k + 2) + dP_k3(1, i, k + 3))/Ts^3;
            U_theta = xddot*(I_y/g);

            % Apply finite differences and linearized dynamics to find U_psi (related to psi)
            psidot = X(12, i, :);
            psiddot = (psidot(k + 2) - psidot(k + 1))/Ts;
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
        % disp('LQR active');

        for i = 1:n_drones
            if terminated(i), continue; end

            %% Compute linearized dynamics for LQR
            x0 = X(:, i, 2);
            u0 = U(:, i, 1);

            %% Update variables and log trajectory
            X(:, i, 1) = x0;
            x_log{i} = [x_log{i}, x0];
            u_log{i} = [u_log{i}, u0];
        end
    end
    
    active = false;
    
    %% Collision and obstacle detection check
    for i = 1:n_drones
        if terminated(i), continue; end
        
        % Drone i position (x, y, z)
        P_i = X(1:3, i, 1);
        
        % Drone-to-object check
        for j = 1:n_objects
            P_j = P_objects(:, j);
            d_ij = norm(P_i - P_j);
            d_obj_log{i, j} = [d_obj_log{i, j}, d_ij];
            if d_ij <= (R_objects(j) + r_safe)
                disp(['Drone ', num2str(i), ' near obstacle ', num2str(j), '. Constraint violated.']);
            end
        end

        % Drone-to-drone check
        for j = (i+1):n_drones
            if terminated(j), continue; end
            P_j = X(1:3, j, 1);
            d_ij = norm(P_i - P_j);
            d_drone_log{i, j} = [d_drone_log{i, j}, d_ij];
            if d_ij <= r_drone
                disp(['Drone ', num2str(i), ' near drone ', num2str(j), '. Constraint violated.']);
            end
        end
    end

    %% Increment iteration
    iter = iter + 1;
end

disp([num2str(iter), ' iterations'])

%% Plot results

do_plot = true;

%% ============================================================
%  3D Trajectory Plot
% ============================================================

views = [[-50, 25]; [50, 25]];
for v = 1:size(views,1)

    fig = figure('Visible','off'); hold on; grid on;
    view(views(v, 1), views(v, 2));

    xlabel('$x$ [m]','Interpreter','latex');
    ylabel('$y$ [m]','Interpreter','latex');
    zlabel('$z$ [m]','Interpreter','latex');
    axis equal;

    % Make z-order respect creation order, not depth
    ax = gca;
    ax.SortMethod = 'childorder';

    colors = lines(n_drones);

    % Pre-store positions for start/end markers
    start_pts = zeros(3,n_drones);
    end_pts   = zeros(3,n_drones);

    % ---- Trajectories first ----
    h_drones = gobjects(n_drones,1);
    for i = 1:n_drones
        xd = x_log{i};
        start_pts(:,i) = xd(1:3,1);
        end_pts(:,i)   = xd(1:3,end);

        h_drones(i) = scatter3(xd(1,:), xd(2,:), xd(3,:), ...
                               5, colors(i,:), 'filled');   % smaller trajectory markers
    end

    % ---- Obstacles second ----
    for j = 1:n_objects
        [Xs, Ys, Zs] = sphere(20);
        Xs = R_objects(j)*Xs + P_objects(1, j);
        Ys = R_objects(j)*Ys + P_objects(2, j);
        Zs = R_objects(j)*Zs + P_objects(3, j);

        C = Zs;
        s = surf(Xs, Ys, Zs, C);
        s.EdgeColor = 'k';
        s.FaceColor = 'interp';
        s.FaceAlpha = 0.4;

        text(P_objects(1, j), P_objects(2, j), P_objects(3, j) + R_objects(j)*0.75, ...
             sprintf('Obj. %d', j), ...
             'HorizontalAlignment', 'center', ...
             'Interpreter', 'latex', ...
             'FontSize', 8, ...
             'FontWeight', 'bold');
    end

    colormap(parula);
    shading interp;

    % ---- Start/End markers LAST (so they sit on top in childorder) ----
    h_start = gobjects(n_drones,1);
    h_end   = gobjects(n_drones,1);

    for i = 1:n_drones
        h_start(i) = scatter3(start_pts(1,i), start_pts(2,i), start_pts(3,i), ...
                              10, 'g', 'filled', 'o', ...
                              'MarkerEdgeColor','k', 'LineWidth',0.7);

        h_end(i)   = scatter3(end_pts(1,i), end_pts(2,i), end_pts(3,i), ...
                              10, 'r', 'filled', 'o', ...
                              'MarkerEdgeColor','k', 'LineWidth',0.7);
    end

    % ---- Legend ----
    legend([h_drones; h_start(1); h_end(1)], ...
        [arrayfun(@(d) sprintf('Drone %d',d), 1:n_drones, 'UniformOutput', false), ...
         {'Start'}, {'End'}], ...
        'Interpreter','latex');

    exportgraphics(fig, "3D_trajectory_" + opt + "_view" + v + ".pdf", ...
                   'ContentType','vector');

    close(fig);
end

if do_plot && opt ~= "lqr"
    %% ============================================================
    %  Control Input Plots
    % ============================================================
    
    u_labels = {'$U_{\mathrm{coll}}$', '$U_{\phi}$', '$U_{\theta}$', '$U_{\psi}$'};
    colors = lines(n_drones);
    
    for u_row = 1:m
        fig = figure('Visible','off'); hold on; grid on;
    
        for i = 1:n_drones
            time_u = (0:size(u_log{i},2)-1)*Ts;
            time_lqr = (0:size(u_lqr_log{i},2)-1)*Ts;
            plot(time_u, u_log{i}(u_row,:), 'LineWidth',1.8, 'Color', colors(i,:));
            plot(time_lqr, u_lqr_log{i}(u_row,:), '--', 'LineWidth',1.8, 'Color', colors(i,:));
        end
    
        legend_entries = cell(1,2*n_drones);
        for i = 1:n_drones
            legend_entries{2*i-1} = sprintf('Drone %d - Optimal control', i);
            legend_entries{2*i}   = sprintf('Drone %d - LQR', i);
        end
        legend(legend_entries, 'Location','bestoutside');
    
        ylabel([u_labels{u_row} '(t)'],'Interpreter','latex');
        xlabel('Time, $t$ [s]','Interpreter','latex');
    
        exportgraphics(fig, "control_input" + u_row + "_" + opt + ".pdf");
    end
    
    %% ============================================================
    %  State vs State-Derivative Plots (Per Drone)
    % ============================================================
    
    state_labels = {'$x$','$y$','$z$','$\phi$','$\theta$','$\psi$', ...
                    '$\dot{x}$','$\dot{y}$','$\dot{z}$', ...
                    '$\dot{\phi}$','$\dot{\theta}$','$\dot{\psi}$'};
    
    for i = 1:n_drones
        xdata = x_log{i};
        T = size(xdata,2);
        time = (0:T-1)*Ts;
    
        fig = figure('Visible','off');
    
        for k = 1:6
            subplot(3,2,k); hold on; grid on;
            plot(time, xdata(k,:), 'LineWidth',1.6);
            plot(time, xdata(k+6,:), 'LineWidth',1.6);
            xlabel('Time, $t$ [s]','Interpreter','latex');
            ylabel(state_labels{k},'Interpreter','latex');
            legend({state_labels{k}, state_labels{k+6}}, 'Interpreter','latex');
        end
    
        exportgraphics(fig, "drone" + i + "_state_transitions_" + opt + ".pdf");
    end
    
    %% ============================================================
    %  2x2 Triple-State Overview (Per Drone)
    % ============================================================
    
    triples = {
        [1 2 3],      {'$x$','$y$','$z$'},                              'Positions';
        [7 8 9],      {'$\dot{x}$','$\dot{y}$','$\dot{z}$'},            'Linear velocities';
        [4 5 6],      {'$\phi$','$\theta$','$\psi$'},                   'Euler angles';
        [10 11 12],   {'$\dot{\phi}$','$\dot{\theta}$','$\dot{\psi}$'}, 'Angular velocities'
    };
    
    for drone = 1:n_drones
        xdata = x_log{drone};
        time = (0:size(xdata,2)-1)*Ts;
    
        fig = figure('Visible','off');
    
        for sp = 1:4
            idxs = triples{sp,1};
            labels = triples{sp,2};
            group_label = triples{sp,3};
    
            subplot(2,2,sp); hold on; grid on;
    
            for k = 1:3
                plot(time, xdata(idxs(k),:), 'LineWidth',1.4);
            end
    
            ylabel(group_label,'Interpreter','latex');
            xlabel('Time, $t$ [s]','Interpreter','latex');
            legend(labels,'Interpreter','latex');
        end
    
        exportgraphics(fig, "drone" + i + "_states_" + opt + ".pdf");
    end
    
    %% ============================================================
    %  Drone–Object Distances
    % ============================================================
    
    colors = lines(n_drones);
    fig = figure('Visible','off');
    
    for j = 1:n_objects
        subplot(n_objects,1,j); hold on; grid on;
    
        for i = 1:n_drones
            dvec = d_obj_log{i,j};
            time = (0:length(dvec)-1)*Ts;
            plot(time, dvec, 'LineWidth',1.5, 'Color', colors(i,:));
        end
    
        yline(r_safe + R_objects(j), 'r--', 'LineWidth',1.4);
    
        ylabel('$d_{ij}$ [m]','Interpreter','latex');
        title(sprintf('Distance to Object %d', j),'Interpreter','latex');
    
        leg = cell(1,n_drones+1);
        for i = 1:n_drones, leg{i} = sprintf('Drone %d', i); end
        leg{end} = 'Minimum safe distance';
        legend(leg, 'Interpreter','latex');
    end
    
    exportgraphics(fig, "drone_obj_dist_" + opt + ".pdf");
    
    %% ============================================================
    %  Drone–Drone Distances (Unique Pairs)
    % ============================================================
    
    colors = lines(n_drones);
    fig = figure('Visible','off'); hold on; grid on;
    
    h_list = [];
    leg_list = {};
    
    for i = 1:n_drones
        for j = (i+1):n_drones
            dvec = d_drone_log{i,j};
            time = (0:length(dvec)-1)*Ts;
            h = plot(time, dvec, 'LineWidth',1.5, 'Color', colors(i,:));
            h_list(end+1) = h;
            leg_list{end+1} = sprintf('Drone %d to Drone %d', i, j);
        end
    end
    
    h_safe = yline(r_drone, 'r--', 'LineWidth',1.4);
    h_list(end+1) = h_safe;
    leg_list{end+1} = 'Minimum drone separation';
    
    legend(h_list, leg_list, 'Interpreter','latex');
    xlabel('Time, $t$ [s]','Interpreter','latex');
    ylabel('$d_{ij}$ [m]','Interpreter','latex');
    
    exportgraphics(fig, "drone_drone_dist_" + opt + ".pdf");
end