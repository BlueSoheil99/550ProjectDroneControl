clear; close all; clc;

%% Optimization approach
opts = ["cvx_single", "cvx_mpc", "casdi", "lqr"];
opt  = "cvx_mpc";

%% Add noise
noise = false;

%% Sampling rate
Ts = 0.02;

%% Linearized drone dynamics
[A, B, Ad, Bd] = helper.linearized_drone(Ts);

%% Non-linear drone dynamics
f = str2func("helper.f");

%% Dimensions
[n, m] = size(B); % States (dim n), inputs (dim m)

%% LQR Design

% Cost matrices
R_e = diag([1/15; 1000; 1000; 100]);
Q_e = diag([0.1; 0.1; 10; 0.01; 0.01; 0.01; 0.01; 0.01; 1; 0.1; 0.1; 0.1]);

% LQR gains
[K, ~, ~] = lqrd(A, B, Q_e, R_e, Ts);

%% Scenario Setup

% Simulation variables
H = 5;         % prediction horizon
n_drones  = 2; % number of drones
n_objects = 2; % number of obstacles

% Drone parameters
m_drone = 0.063; % kg
I_x = 0.5829e-4; % kg m^2
I_y = 0.7169e-4; % kg m^2
I_z = 1.000e-4;  % kg m^2
g = 9.8;         % m/s^2

% Initialize states and control inputs
X = zeros(n, n_drones, H);
U = zeros(m, n_drones, H);

% Initial position for each drone (x0, y0, z0)
X(:, 1, 1) = [-2; 0.5; 0.5; zeros(n - 3, 1)];
X(:, 2, 1) = [-2; -0.5; -0.5; zeros(n - 3, 1)];

% Target position for each drone (xf, yf, zf)
X_target = zeros(n, n_drones);
X_target(1:3, 1) = [4; -0.5; -0.5];
X_target(1:3, 2) = [4; 0.5; 0.5];

% Safety radii
r_safe = 0.0; % Design parameter
r_drone = 0.6;

% Obstacle positions and radii
P_objects = [[0.3; -0.25; 0.25], [0.3; 0.25; -0.5]]; % Array of column vectors (x, y, z)
R_objects = [0.3, 0.3];                              % Radii of objects

% Unit vector for shortest path of each drone
s = zeros(n, n_drones);

% Reference position
X_ref = zeros(n, n_drones);

% Setup
d = 0.75;                  % Waypoint distance
for i = 1:n_drones
    p0 = X(1:3, i, 1);     % Initial position of drone i
    pf = X_target(1:3, i); % Target position of drone i
    v  = pf - p0;          % Direction vector
    s(1:3, i) = v/norm(v); % Normalized direction

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

% Noise
Q = 1e-4*eye(n);
Lw = chol(Q, 'lower');

R = 1e-4*eye(n);
Lv = chol(R, 'lower');

% Estimated states
Xp = X;

% Covariance of estimation error
P = zeros(n, n*2, n_drones);
Pp = zeros(n, n, n_drones);
Kk = zeros(n, n, n_drones);

% Helper class
util = helper(r_drone, r_safe, P_objects, R_objects, n_drones, n_objects, Ts, opt);

%% Simulation variable
iter = 0;                       % Iteration count
waypoint = true(n_drones, 1);   % Waypoint terminal flag
terminated = false(n_drones,1); % Target position achieved flag

%% Main loop
while any(~terminated)
    % Stop condition
    if iter > 500
        break;
    end

    % Check per-drone termination
    for i = 1:n_drones
        if ~terminated(i)
            if vecnorm(X(:, i, 1) - X_target(:, i), 2, 1) < 0.1
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
        e = X_ref - X_target;
        dist = vecnorm(e, 2, 1);

        % Set reference if at goal
        terminal = dist < d;
        X_ref(:, terminal) = X_target(:, terminal);

        % Update waypoint flag
        waypoint = ~terminal;
    end
    
    %% Predict linearized state-dynamics for each drone using baseline controller (LQR)
    for i = 1:n_drones
        if terminated(i), continue; end
        
        for j = 1:(H-1)
            if noise
                % Process noise
                w = Lw*rand(n, 1);
                v = Lv*rand(n, 1);
            else
                w = 0;
                v = 0;
            end
            
            % Basline control input via LQR
            U(:, i, j) = -K*(X(:, i, j) - X_ref(:, i));
            X(:, i, j + 1) = Ad*X(:, i, j) + Bd*U(:, i, j) + w;
            y = X(:, i, j + 1) + v;
            
            % Dynamics with process noise
            Xp(:, i, j + 1) = Ad*X(:, i, j) + Bd*U(:, i, j) + w;
            
            % Covariance of estimation error
            Pp(:, :, i) = Ad*P(:, 1:n, i)*Ad' + Q;
            Kk(:, :, i) = Pp(:, :, i)*((Pp(:, :, i) + R)\eye(n));
            P(:, n+1:end, i) = (eye(n) - Kk(:, :, i))*Pp(:, :, i);
            P(:, 1:n, i) = P(:, n+1:end, i);

            % Kalman filter
            X(:, i, j + 1) = Xp(:, i, j + 1) + Kk(:, :, i)*(y - Xp(:, i, j + 1));
            U(1, i, j) = U(1, i, j) + m_drone*g; % For completeness
        end
        
        u_lqr_log{i} = [u_lqr_log{i}, U(:, i, 1)];
    end
    
    %% Optimization trigger flag
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
                disp(['Drone ', num2str(i), ' near obstacle ', num2str(j), '. Constraint violated.']);
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
                disp(['Drone ', num2str(i), ' near drone ', num2str(j), '. Constraint violated.']);
                break;
            end
        end
        if active, break; end
    end

    %% Apply optimization when needed
    if active && opt ~= "lqr"
        disp(['Iteration: ', num2str(iter), ' - Optimization active']);
        
        % Optimization
        k = 1;
        dP_sol = util.drone_opt(X, k + 3); % 3-step ahead optimization

        for i = 1:n_drones
            if terminated(i), continue; end
        
            % Apply finite differences and linearized dynamics to find U_coll (related to z)
            zdot = X(9, i, :);
            zddot = (dP_sol(3, i, 1) - zdot(1))/Ts;
    
            % Apply finite differences and linearized dynamics to find U_phi (related to y)
            ydot = X(8, i, :);
            yddot = (-ydot(1) + 3*dP_sol(2, i, 1) - 3*dP_sol(2, i, 2) + dP_sol(2, i, 3))/Ts^3;
    
            % Apply finite differences and linearized dynamics to find U_theta (related to x)
            xdot = X(7, i, :);
            xddot = (-xdot(1) + 3*dP_sol(1, i, 1) - 3*dP_sol(1, i, 2) + dP_sol(1, i, 3))/Ts^3;
    
            % Apply finite differences and linearized dynamics to find U_psi (related to psi)
            psidot = X(12, i, :);
            psiddot = (psidot(2) - psidot(1))/Ts;

            % Optimal control inputs
            U_coll = (zddot + g)*m_drone;
            U_phi = yddot*(-I_x/g);
            U_theta = xddot*(I_y/g);
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
        disp(['Iteration: ', num2str(iter), ' - LQR active']);

        for i = 1:n_drones
            if terminated(i), continue; end

            %% Compute linearized dynamics for LQR
            x0 = X(:, i, 2);
            u0 = U(:, i, 1);

            %% Update variables and log trajectory
            X(:, i, 1) = x0;
            Xp(:, i, 1) = x0;
            x_log{i} = [x_log{i}, x0];
            u_log{i} = [u_log{i}, u0];
        end
    end
       
    %% Collision and obstacle detection post-check
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

%% Plot results
util.plotTrajectories(x_log, opt);

do_plot = false;
if do_plot && opt ~= "lqr"
    util.plotControlInputs(u_log, u_lqr_log, opt);
    util.plotStateTransitions(x_log, opt);
    util.plotGroupedStates(x_log, opt);
    util.plotDroneObjectDistances(d_obj_log, opt);
    util.plotDroneDroneDistances(d_drone_log, opt);
end

do_animate = false;
if do_animate
    util.animateDrones(x_log, opt);
end