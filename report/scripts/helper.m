classdef helper
    properties
        r_drone
        r_safe
        P_objects
        R_objects
        n_drones
        n_objects
        Ts
        opt
    end

    %% ===========================================================
    % Constructor
    methods
        function self = helper(r_drone, r_safe, P_objects, R_objects, n_drones, n_objects, Ts, opt)
            self.r_drone   = r_drone;
            self.r_safe    = r_safe;
            self.P_objects = P_objects;
            self.R_objects = R_objects;
            self.n_drones  = n_drones;
            self.n_objects = n_objects;
            self.Ts        = Ts;
            self.opt       = opt;
        end
    end
    
    methods (Static)
        %% ===========================================================
        % Linearized dynamics
        function [A, B, Ad, Bd] = linearized_drone(Ts)
            % Ts is the sampling period, this function outputs continunous A, B if Ts
            % is empty.
            % Discrete Ad and Bd using forward Euler
            % state x is 12x1
            % input u is 4x1
            
            % this allows input with or without Ts
            if nargin < 1
                Ts = [];
            end
            
            % system parameters 
            m  = 0.063; % kg
            I_x = 0.5829e-4; % kg*m^2
            I_y = 0.7169e-4; % kg*m^2
            I_z = 1.000e-4; % kg*m^2
            g  = 9.8; % m/s^2
            
            % Continuous time A, B
            A = zeros(12,12);
            
            A(1,7) = 1;
            A(2,8) = 1;
            A(3,9) = 1;
            A(4,10)= 1;
            A(5,11)= 1;
            A(6,12)= 1;
            
            % simplified model
            A(7,5) = g; % x_2dot
            A(8,4) = -g; % y_2dot
            
            B = zeros(12,4);
            
            % u_l = u - u_e
            B(9,1)  = 1/m; % this only computes U_coll/m, full term should be z_2dot = U_coll/m - g
            B(10,2) = 1/I_x;   
            B(11,3) = 1/I_y;   
            B(12,4) = 1/I_z;  
            
            % if we have Ts, discretized time via forward Euler:
            if ~isempty(Ts)
                Ad = eye(12) + Ts * A;
                Bd = Ts * B;
            else
                Ad = []; Bd = [];
            end
        end

        %% ===========================================================
        % Non-linear dynamics
        function xdot = f(x, u)
            % continous time dynamics 
            % state x is 12x1
            % input u is 4x1
            
            % system parameters
            m = 0.063; % kg
            I_x = 0.5829e-4; % kgm^2
            I_y = 0.7169e-4; % kgm^2
            I_z = 1.000e-4; % kgm^2
            g = 9.8; % m/s^2
            
            % state
            p_x = x(1);
            p_y = x(2);
            p_z = x(3);
            phi = x(4);
            theta = x(5);
            psi = x(6);
            v_x = x(7);
            v_y = x(8);
            v_z = x(9);
            phi_dot = x(10);
            theta_dot = x(11);
            psi_dot = x(12);
            
            % input 
            U_coll = u(1);
            U_phi  = u(2);
            U_theta= u(3);
            U_psi  = u(4);
            
            % representation trigs
            cphi = cos(phi);
            sphi = sin(phi);
            ctheta = cos(theta);
            stheta = sin(theta);
            cpsi = cos(psi);
            spsi = sin(psi);
            
            % Jacobian matrix (phi, theta, psi)
            J11 = I_x; J12 = 0; J13 = -I_x*stheta;
            J21 = 0; J22 = I_y*cphi^2 + I_z*sphi^2; J23 = (I_y - I_z)*cphi*sphi*ctheta;
            J31 = -I_x*stheta; J32 = (I_y - I_z)*cphi*sphi*ctheta; 
            J33 = I_x*stheta^2 + I_y*sphi^2*ctheta^2 + I_z*cphi^2*ctheta^2; 
            J = [J11, J12, J13;
                 J21, J22, J23;
                 J31, J32, J33];
            % J = diag([I_x, I_y, I_z]);
            
            % Coriolis matrix (phi, theta, psi, phi_dot, theta_dot, psi_dot)
            c11 = 0;
            c12 = (I_y - I_z)*(theta_dot*cphi*sphi + psi_dot*sphi^2*ctheta) + ...
                  (I_z - I_y)*psi_dot*cphi^2*ctheta - I_x*psi_dot*ctheta;
            c13 = (I_z - I_y)*psi_dot*(ctheta^2)*sphi*cphi;
            
            c21 = (I_z - I_y)*(theta_dot*sphi*cphi + psi_dot*sphi^2*ctheta) + ...
                  (I_y - I_z)*psi_dot*cphi^2*ctheta + I_x*psi_dot*ctheta;
            c22 = (I_z - I_y)*phi_dot*cphi*sphi;
            c23 = -I_x*psi_dot*stheta*ctheta + I_y*psi_dot*sphi^2*stheta*ctheta + ...
                  I_z*psi_dot*cphi^2*stheta*ctheta;
            
            c31 = (I_y - I_z)*psi_dot*(ctheta^2)*sphi*cphi - I_x*theta_dot*ctheta;
            c32 = (I_z - I_y)*(theta_dot*cphi*sphi*stheta + phi_dot*sphi^2*ctheta) + ...
                  (I_y - I_z)*phi_dot*cphi^2*ctheta + I_x*psi_dot*stheta*ctheta - ...
                  I_y*psi_dot*sphi^2*stheta*ctheta - I_z*psi_dot*cphi^2*stheta*ctheta;
            c33 = (I_y - I_z)*phi_dot*(ctheta^2)*sphi*cphi - ...
                  I_y*theta_dot*sphi^2*stheta*ctheta - I_z*theta_dot*cphi^2*stheta*ctheta + ...
                  I_x*theta_dot*stheta*ctheta;
            
            c = [c11, c12, c13;
                 c21, c22, c23;
                 c31, c32, c33];
            
            x2dot = (cphi*stheta*cpsi + sphi*spsi) * U_coll / m;
            y2dot = (cphi*stheta*cpsi - sphi*spsi) * U_coll / m;
            z2dot = -g + (cphi*ctheta) * U_coll / m;
            
            U = [U_phi; U_theta; U_psi];
            eta_dot = [phi_dot; theta_dot; psi_dot];
            
            paren = U - c*eta_dot;
            eta2dot = J \ (U - c*eta_dot);
            
            phi2dot = eta2dot(1);
            theta2dot = eta2dot(2);
            psi2dot = eta2dot(3);
            
            % xdot
            xdot = zeros(12,1);
            
            xdot(1) = v_x;
            xdot(2) = v_y;
            xdot(3) = v_z;
            xdot(4) = phi_dot;
            xdot(5) = theta_dot;
            xdot(6) = psi_dot;
            
            xdot(7)  = x2dot;
            xdot(8)  = y2dot;
            xdot(9)  = z2dot;
            xdot(10) = phi2dot;
            xdot(11) = theta2dot;
            xdot(12) = psi2dot;
        end
    end
    
    methods
        %% ===========================================================
        % Drone optimization, MPC
        function dP_sol = drone_opt(self, X, H)
            cvx_begin quiet
                % Initialize variables
                variables dP(3, self.n_drones, H - 1)
                obj = 0;
                constraints = [];
                
                % Optimize from k = 2 to H
                for i = 1:self.n_drones
                    for k = 2:H
                        %% Compute objective of drones
                        dv = X(7:9, i, k) - dP(:, i, k - 1);
                        obj = obj + dv'*dv;
        
                        % Drone i position at timestep k
                        P_i = X(1:3, i, k);
        
                        %% Drone-to-object constraints
                        for j = 1:self.n_objects
                            % Object j position
                            P_j = self.P_objects(:, j);
            
                            % Control barrier function
                            d_ij = P_i - P_j;
                            h_ij = d_ij'*d_ij - (self.R_objects(j) + self.r_safe)^2;
                            grad_h = 2*d_ij;
                            alpha = 100*h_ij^3;
                            constraints = [constraints, grad_h'*dP(:, i, k - 1) + alpha >= 0];
                        end
        
                        %% Drone-to-drone constraints
                        for j = (i+1):self.n_drones
                            % Drone j position at timestep k
                            P_j = X(1:3, j, k);
        
                            % Control barrier function
                            d_ij = P_i - P_j;
                            h_ij = d_ij'*d_ij - self.r_drone^2;
                            grad_h = 2*[d_ij; -d_ij];
                            dP_stack = [dP(:, i, k - 1); dP(:, j, k - 1)];
                            alpha = 100*h_ij^3;
                            constraints = [constraints, grad_h'*dP_stack + alpha >= 0];
                        end
                    end      
                end
        
                minimize(obj)
                subject to
                    constraints;
            cvx_end
            
            % Store optimized solution
            dP_sol = dP;
        end

        %% ===========================================================
        % 3D trajectories
        function plotTrajectories(self, x_log, opt)
            % Camera views
            views = [[-60,35]; [20,35]];
            colors = lines(self.n_drones);
        
            for v = 1:size(views,1)
                % Figure setup
                fig = figure();
                
                hold on; grid on;
                view(views(v,1), views(v,2));
                xlabel('$x$ [m]','Interpreter','latex');
                ylabel('$y$ [m]','Interpreter','latex');
                zlabel('$z$ [m]','Interpreter','latex');
                axis equal;
        
                ax = gca;
                ax.SortMethod = 'childorder';
                
                % Label sizes
                ax.XLabel.FontSize = 14;
                ax.YLabel.FontSize = 14;
                ax.ZLabel.FontSize = 14;
                ax.GridAlpha = 0.5;
                
                % Plot drone trajectories
                h_drones  = gobjects(self.n_drones,1);
                start_pts = zeros(3,self.n_drones);
                end_pts   = zeros(3,self.n_drones);
        
                for i = 1:self.n_drones
                    xd = x_log{i};
                    start_pts(:,i) = xd(1:3,1);
                    end_pts(:,i)   = xd(1:3,end);
        
                    h_drones(i) = scatter3( ...
                        xd(1,:), xd(2,:), xd(3,:), ...
                        5, colors(i,:), 'filled');
                end
        
                % Plot obstacles
                for j = 1:self.n_objects
                    [Xs, Ys, Zs] = sphere(20);
        
                    Xs = self.R_objects(j)*Xs + self.P_objects(1,j);
                    Ys = self.R_objects(j)*Ys + self.P_objects(2,j);
                    Zs = self.R_objects(j)*Zs + self.P_objects(3,j);
        
                    s = surf(Xs, Ys, Zs, Zs);
                    s.EdgeColor = 'k';
                    s.FaceColor = 'interp';
                    s.FaceAlpha = 0.45;
        
                    text( ...
                        self.P_objects(1,j), self.P_objects(2,j), ...
                        self.P_objects(3,j) + 0.75*self.R_objects(j), ...
                        sprintf('Obj. %d', j), ...
                        'Interpreter','latex', ...
                        'HorizontalAlignment','center', ...
                        'FontSize',8, 'FontWeight','bold');
                end
        
                shading interp
                colormap(parula)
        
                % Plot start/end markers
                h_start = gobjects(self.n_drones,1);
                h_end   = gobjects(self.n_drones,1);
        
                for i = 1:self.n_drones
                    h_start(i) = scatter3( ...
                        start_pts(1,i), start_pts(2,i), start_pts(3,i), ...
                        10, 'g', 'filled', 'o', 'MarkerEdgeColor','k');
        
                    h_end(i) = scatter3( ...
                        end_pts(1,i), end_pts(2,i), end_pts(3,i), ...
                        10, 'r', 'filled', 'o', 'MarkerEdgeColor','k');
                end
        
                % Legend
                legend( ...
                    [h_drones; h_start(1); h_end(1)], ...
                    [ arrayfun(@(d) sprintf('Drone %d',d), 1:self.n_drones, 'UniformOutput', false), ...
                      {'Start'}, {'End'} ], ...
                    'Interpreter','latex', ...
                    'Location','southoutside', ...
                    'Orientation','horizontal', 'FontSize', 14);
                
                % Export figure
                exportgraphics(fig, ...
                    "figures/3D_trajectory_" + opt + "_view" + v + ".pdf", ...
                    'ContentType','vector');
            end
        end

        %% ===========================================================
        % Control inputs (4x1 tiledlayout)
        function plotControlInputs(self, u_log, u_lqr_log, opt)
            % Control input labels and units
            u_labels = {'$U_{\mathrm{coll}}$', '$U_{\phi}$', '$U_{\theta}$', '$U_{\psi}$'};
            u_units  = {'$\mathrm{N}$', '$\mathrm{N}\cdot\mathrm{m}$', ...
                        '$\mathrm{N}\cdot\mathrm{m}$', '$\mathrm{N}\cdot\mathrm{m}$'};
      
            colors = lines(self.n_drones);
            m = 4;
        
            % Create figure
            fig = figure('Visible','off','Position',[100 100 1200 800]);
            t = tiledlayout(4,1,'TileSpacing','compact','Padding','compact');
        
            % Build legend text
            legend_entries = cell(1, 2*self.n_drones);
            for i = 1:self.n_drones
                legend_entries{2*i-1} = sprintf('Drone %d - Optimal', i);
                legend_entries{2*i}   = sprintf('Drone %d - LQR', i);
            end
        
            % Plot control inputs
            for u_row = 1:m
                ax = nexttile;
                hold(ax,'on');
                grid(ax,'on');
        
                for i = 1:self.n_drones
                    time_u   = (0:size(u_log{i},2)-1)   * self.Ts;
                    time_lqr = (0:size(u_lqr_log{i},2)-1) * self.Ts;
        
                    plot(ax, time_u,   u_log{i}(u_row,:),     'LineWidth',1,   'Color',colors(i,:));
                    plot(ax, time_lqr, u_lqr_log{i}(u_row,:), '--', 'LineWidth', 1.2, 'Color',colors(i,:));
                end
        
                ylabel(ax, u_labels{u_row} + " [" + u_units{u_row} + "]", 'Interpreter','latex','FontSize',14);
                if u_row == m
                    xlabel(ax, 'Time, $t$ [s]', 'Interpreter','latex','FontSize',14);
                end
            end
        
            % Add legend
            legend(legend_entries, 'Interpreter','latex', ...
                'Location','southoutside', 'Orientation','horizontal', 'FontSize', 14);
        
            % Export figure
            exportgraphics(fig, "figures/control_inputs_" + opt + ".pdf");
        end
        
        %% ===========================================================
        % State vs state-derivative (per drone)
        function plotStateTransitions(self, x_log, opt)
            % State labels for positions and velocities
            state_labels = {'$x$','$y$','$z$','$\phi$','$\theta$','$\psi$', ...
                            '$\dot{x}$','$\dot{y}$','$\dot{z}$', ...
                            '$\dot{\phi}$','$\dot{\theta}$','$\dot{\psi}$'};
            state_units = {'[m]','[m]','[m]','[rad]','[rad]','[rad]', ...
                            '[m/s]','[m/s]','[m/s]', ...
                            '[rad/s]','[rad/s]','[rad/s]'};

            for i = 1:self.n_drones
                % Extract data
                xdata = x_log{i};
                T = size(xdata,2);
                time = (0:T-1) * self.Ts;
        
                % Create figure
                fig = figure('Visible','off','Position',[100 100 1200 900]);

                % Plot 6 position and angle states with their derivatives
                for k = 1:6
                    subplot(3,2,k);
                    hold on;
                    grid on;
        
                    plot(time, xdata(k,:),   'LineWidth',1);
                    plot(time, xdata(k+6,:), 'LineWidth',1);
        
                    xlabel('Time, $t$ [s]', 'Interpreter','latex','FontSize',14);
                    label = state_labels(k) + " " + state_units(k) + " and " + state_labels(k+6) + " " + state_units(k+6);
                    ylabel(label,  'Interpreter','latex','FontSize',14);

                    legend({state_labels{k}, state_labels{k+6}}, 'Interpreter','latex', 'FontSize', 14);
                end
        
                % Export figure
                exportgraphics(fig, ...
                    "figures/drone" + i + "_state_transitions_" + opt + ".pdf");
            end
        end

        %% ===========================================================
        % Grouped states overview (2x2 per drone)
        function plotGroupedStates(self, x_log, opt)
            % State groups: indices, labels, titles
            triples = {
                [1 2 3],      {'$x$','$y$','$z$'},                                'Positions [m]';
                [7 8 9],      {'$\dot{x}$','$\dot{y}$','$\dot{z}$'},              'Linear velocities [m/s]';
                [4 5 6],      {'$\phi$','$\theta$','$\psi$'},                     'Euler angles [rad]';
                [10 11 12],   {'$\dot{\phi}$','$\dot{\theta}$','$\dot{\psi}$'},   'Angular velocities [rad/s]'
            };
        
            for i = 1:self.n_drones
                % Extract data and time vector
                xdata = x_log{i};
                time = (0:size(xdata,2)-1) * self.Ts;
        
                % Create figure
                fig = figure('Visible','off','Position',[100 100 1200 600]);

                % Plot four groups of states
                for sp = 1:4
                    idxs   = triples{sp,1};
                    labels = triples{sp,2};
                    title_label = triples{sp,3};
        
                    subplot(2,2,sp);
                    hold on;
                    grid on;
        
                    for k = 1:3
                        plot(time, xdata(idxs(k),:), 'LineWidth',1);
                    end

                    ylabel(title_label, 'Interpreter','latex','FontSize',14);
                    xlabel('Time, $t$ [s]', 'Interpreter','latex','FontSize',14);
                    legend(labels, 'Interpreter','latex', 'FontSize', 14);
                end
        
                % Export figure
                exportgraphics(fig, ...
                    "figures/drone" + i + "_states_" + opt + ".pdf");
            end
        end

        %% ===========================================================
        % Drone-object distances
        function plotDroneObjectDistances(self, d_obj_log, opt)
            % Colors for each drone
            colors = lines(self.n_drones);
        
            % Create figure sized by number of objects
            fig = figure('Visible','off', 'Position',[100 100 1200 300*self.n_objects]);

            for j = 1:self.n_objects
                subplot(self.n_objects,1,j);
                hold on;
                grid on;
        
                leg = {};
        
                % Plot distance from each drone to object j
                for i = 1:self.n_drones
                    dvec = d_obj_log{i,j};
                    time = (0:length(dvec)-1) * self.Ts;
        
                    % Plot every second point to reduce size
                    plot(time(1:2:end), dvec(1:2:end), 'LineWidth',1, 'Color', colors(i,:));
                    leg{end+1} = sprintf('Drone %d', i);
                end
                
                % Minimum safe distance line
                yline(self.r_safe + self.R_objects(j), 'r--', 'LineWidth',1);
                leg{end+1} = 'Minimum safe distance';
                
                % Label sizes
                ax.FontSize = 10;
                ax.XLabel.FontSize = 12;
                ax.YLabel.FontSize = 12;

                if j == self.n_objects
                    xlabel('Time, $t$ [s]', 'Interpreter','latex','FontSize',14);
                end
        
                ylabel('$d_{ij}$ [m]', 'Interpreter','latex','FontSize',14);
                title(sprintf('Distance to object %d', j), 'Interpreter','latex','FontSize',14);
            end
        
            % Combined legend at bottom
            legend(leg, 'Interpreter','latex', ...
                'Location','southoutside', 'Orientation','horizontal', 'FontSize',14);
        
            % Export figure
            exportgraphics(fig, "figures/drone_obj_dist_" + opt + ".pdf");
        end

        %% ===========================================================
        % Drone-drone distances
        function plotDroneDroneDistances(self, d_drone_log, opt)
            % Colors for each drone
            colors = lines(self.n_drones);
        
            % Create figure
            fig = figure('Visible','off','Position',[100 100 1200 500]);
           
            hold on;
            grid on;
        
            leg = {};
        
            % Plot pairwise drone distances
            for i = 1:self.n_drones
                for j = i+1:self.n_drones
                    dvec = d_drone_log{i,j};
                    time = (0:length(dvec)-1) * self.Ts;
        
                    % Downsample to reduce plot size
                    plot(time(1:2:end), dvec(1:2:end), ...
                        'LineWidth',1, 'Color', colors(i,:));
        
                    leg{end+1} = sprintf('Drone %d to Drone %d', i, j);
                end
            end
        
            % Minimum allowed distance line
            yline(self.r_drone, 'r--', 'LineWidth',1);
            leg{end+1} = 'Minimum separation';

            xlabel('Time, $t$ [s]', 'Interpreter','latex','FontSize',14);
            ylabel('$d_{ij}$ [m]', 'Interpreter','latex','FontSize',14);
        
            legend(leg, 'Interpreter','latex', ...
                'Location','southoutside', 'Orientation','horizontal','FontSize',14);
        
            exportgraphics(fig, "figures/drone_drone_dist_" + opt + ".pdf");
        end
    
        %% ===========================================================
        % Multi-drone animation
        function animateDrones(self, x_log, opt)
            gif_name = "3D_trajectory_" + opt + ".gif";
            fps = 20;
            delay = 1/fps;
            
            % Compute fixed axis limits
            all_x = []; all_y = []; all_z = [];
            for i = 1:self.n_drones
                traj = x_log{i};
                all_x = [all_x traj(1,:)];
                all_y = [all_y traj(2,:)];
                all_z = [all_z traj(3,:)];
            end
            
            margin = 0.2;
            xmin = min(all_x) - margin;
            xmax = max(all_x) + margin;
            ymin = min(all_y) - margin;
            ymax = max(all_y) + margin;
            zmin = min(all_z) - margin;
            zmax = max(all_z) + margin;
        
            % Setup figure
            fig_anim = figure('Visible','off', 'Position',[100 100 1600 1200], 'Renderer', 'opengl');

            hold on; grid on;
            view(20,35);
        
            xlabel('$x$ [m]','Interpreter','latex');
            ylabel('$y$ [m]','Interpreter','latex');
            zlabel('$z$ [m]','Interpreter','latex');
        
            axis equal;
            xlim([xmin xmax]);
            ylim([ymin ymax]);
            zlim([zmin zmax]);
        
            ax = gca;
            ax.SortMethod = 'childorder';
            
            % Label sizes
            ax.FontSize = 10;
            ax.XLabel.FontSize = 12;
            ax.YLabel.FontSize = 12;
            ax.ZLabel.FontSize = 12;
    
            colors = lines(self.n_drones);

            % Drone geometry model (simplified quadrotor, ~0.18 x 0.18 x 0.10 m)

            % Body box dimensions
            bx = 0.10;   % length (x)
            by = 0.03;   % width  (y)
            bz = 0.03;   % height (z)
            dx = bx/2; dy = by/2; dz = bz/2;
        
            % Body vertices (centered at origin)
            bodyVerts = [ ...
                -dx -dy -dz;
                 dx -dy -dz;
                 dx  dy -dz;
                -dx  dy -dz;
                -dx -dy  dz;
                 dx -dy  dz;
                 dx  dy  dz;
                -dx  dy  dz];
        
            bodyFaces = [ ...
                1 2 3 4;
                5 6 7 8;
                1 2 6 5;
                2 3 7 6;
                3 4 8 7;
                4 1 5 8];
        
            % Arms (4 arms at +/- 45 degrees, square cross-section)
            arm_half = 0.09;      % arm length from center to near rotor
            arm_w    = 0.01;      % arm thickness
            a        = arm_w/2;
        
            % Base arm along +x
            baseArm = [ ...
                0        -a  -a;
                arm_half -a  -a;
                arm_half  a  -a;
                0         a  -a;
                0        -a   a;
                arm_half -a   a;
                arm_half  a   a;
                0         a   a];
        
            Rz_deg = @(ang) [cosd(ang) -sind(ang) 0; ...
                             sind(ang)  cosd(ang) 0; ...
                             0          0         1];
        
            armVerts = cell(1,4);
            armFaces = repmat({[1 2 3 4; 5 6 7 8; 1 2 6 5; 2 3 7 6; 3 4 8 7; 4 1 5 8]}, 1, 4);
            armAngles = [45, 135, -135, -45];   % front-right, front-left, rear-left, rear-right
        
            for a_i = 1:4
                armVerts{a_i} = (Rz_deg(armAngles(a_i)) * baseArm')';
            end
        
            % Motors (small cylinders) and their offsets from origin
            N_cyl = 20;
            r_motor = 0.012;
            h_motor = 0.02;
        
            [Xc, Yc, Zc] = cylinder(r_motor, N_cyl);
            Zc = Zc * h_motor;
        
            motorX = Xc;
            motorY = Yc;
            motorZ = Zc;
        
            motorOffsets = zeros(4,3);
            motorOffsets(1,:) = (Rz_deg(45)   * [arm_half 0 0]')';   % front-right
            motorOffsets(2,:) = (Rz_deg(135)  * [arm_half 0 0]')';   % front-left
            motorOffsets(3,:) = (Rz_deg(-135) * [arm_half 0 0]')';   % rear-left
            motorOffsets(4,:) = (Rz_deg(-45)  * [arm_half 0 0]')';   % rear-right
        
            % Propellers (flat ellipses)
            Rprop = 0.035;
            t_prop = linspace(0, 2*pi, 40);
            propX = Rprop * cos(t_prop);
            propY = 0.6 * Rprop * sin(t_prop);
            prop_z_offset = h_motor + 0.008;
            propZ = zeros(size(t_prop)) + prop_z_offset;
        
            % Plot obstacles
            for j = 1:self.n_objects
                [Xs, Ys, Zs] = sphere(20);
                Xs = self.R_objects(j)*Xs + self.P_objects(1, j);
                Ys = self.R_objects(j)*Ys + self.P_objects(2, j);
                Zs = self.R_objects(j)*Zs + self.P_objects(3, j);
        
                C = Zs;
                s = surf(Xs, Ys, Zs, C, ...
                         'HandleVisibility','off');
                s.EdgeColor = 'none';
                s.FaceColor = 'interp';
                s.FaceAlpha = 0.4;
                s.LineStyle = 'none';
                
                text(self.P_objects(1, j), self.P_objects(2, j), ...
                     self.P_objects(3, j) + self.R_objects(j)*0.75, ...
                     sprintf('Obj. %d', j), ...
                     'HorizontalAlignment','center', ...
                     'Interpreter','latex', ...
                     'FontSize',10, 'FontWeight','bold', ...
                     'HandleVisibility','off');
            end
        
            colormap(parula);
            shading interp;
      
            % Drone markers & dashed paths
            h_marker = gobjects(self.n_drones,1);
            h_path   = gobjects(self.n_drones,1);
        
            % Drone body / arms / motors / props
            h_body   = gobjects(self.n_drones,1);
            h_arms   = cell(self.n_drones,4);
            h_motors = cell(self.n_drones,4);
            h_props  = cell(self.n_drones,4);
        
            for i = 1:self.n_drones
                % Point marker
                h_marker(i) = scatter3(NaN,NaN,NaN, 25, colors(i,:), 'filled', ...
                                       'HandleVisibility','off');
        
                % Dashed path
                h_path(i)   = plot3(NaN,NaN,NaN, '--', ...
                                    'Color', colors(i,:), ...
                                    'LineWidth', 1.0, ...
                                    'HandleVisibility','off');
        
                % Body patch
                h_body(i) = patch('Vertices', bodyVerts, ...
                                  'Faces', bodyFaces, ...
                                  'FaceColor', colors(i,:), ...
                                  'FaceAlpha', 0.30, ...
                                  'EdgeColor', 'none', ...
                                  'HandleVisibility','off');
        
                % Arm patches
                for a_i = 1:4
                    h_arms{i,a_i} = patch('Vertices', armVerts{a_i}, ...
                                          'Faces', armFaces{a_i}, ...
                                          'FaceColor', colors(i,:), ...
                                          'FaceAlpha', 0.20, ...
                                          'EdgeColor', 'none', ...
                                          'HandleVisibility','off');
                end
        
                % Motors
                for m_i = 1:4
                    h_motors{i,m_i} = surf(motorX, motorY, motorZ, ...
                                           'FaceColor', colors(i,:), ...
                                           'EdgeColor','none', ...
                                           'FaceAlpha', 0.9, ...
                                           'HandleVisibility','off');
                end
        
                % Propellers
                for p_i = 1:4
                    h_props{i,p_i} = fill3(propX, propY, propZ, ...
                                           colors(i,:), ...
                                           'FaceAlpha',0.4, ...
                                           'EdgeColor','none', ...
                                           'HandleVisibility','off');
                end
            end
        
            % Legend handles
            legend_handles = [];
            legend_names   = {};
            
            % Drone markers (for legend only)
            for i = 1:self.n_drones
                h_leg_marker = scatter3(NaN,NaN,NaN,25,colors(i,:),'filled');
                legend_handles = [legend_handles; h_leg_marker];
                legend_names{end+1} = sprintf('Drone %d', i);
            end
            
            % Horizontal legend at bottom
            lgd = legend(legend_handles, legend_names, ...
                         'Interpreter','latex', ...
                         'Orientation','horizontal', ...
                         'NumColumns', self.n_drones, ...
                         'Location','southoutside');
            lgd.FontSize = 10;
        
            % Timestamp text
            t_handle = annotation(fig_anim, 'textbox', ...
            [0.80 0.92 0.18 0.05], ...   % [x y w h] normalized to figure
            'String','t = 0.00 s', ...
            'Interpreter','latex', ...
            'FontSize',14, ...
            'HorizontalAlignment','right', ...
            'VerticalAlignment','top', ...
            'EdgeColor','none', ...
            'BackgroundColor','none');
        
            % Start GIF creation
            T = size(x_log{1}, 2);   % number of samples (columns)
        
            for k = 1:T
        
                % Update drone trajectory + markers + body model
                for i = 1:self.n_drones
                    xi = x_log{i};
        
                    % Position
                    px = xi(1,k);
                    py = xi(2,k);
                    pz = xi(3,k);
                    pos = [px py pz];
        
                    % Orientation (phi, theta, psi)
                    phi   = xi(4,k);   % roll about x
                    theta = xi(5,k);   % pitch about y
                    psi   = xi(6,k);   % yaw about z
        
                    % Rotation matrices
                    Rx = [1 0 0; ...
                          0 cos(phi) -sin(phi); ...
                          0 sin(phi)  cos(phi)];
        
                    Ry = [ cos(theta) 0 sin(theta); ...
                           0          1        0; ...
                          -sin(theta) 0 cos(theta)];
        
                    Rz = [cos(psi) -sin(psi) 0; ...
                          sin(psi)  cos(psi) 0; ...
                          0         0        1];
        
                    % Total rotation
                    R = Rz * Ry * Rx;
        
                    % Update body
                    bodyWorld = (R * bodyVerts')' + pos;
                    set(h_body(i), 'Vertices', bodyWorld);
        
                    % Update arms
                    for a_i = 1:4
                        armWorld = (R * armVerts{a_i}')' + pos;
                        set(h_arms{i,a_i}, 'Vertices', armWorld);
                    end
        
                    % Update motors
                    for m_i = 1:4
                        offset = motorOffsets(m_i,:);
        
                        Xm = motorX + offset(1);
                        Ym = motorY + offset(2);
                        Zm = motorZ + offset(3);
        
                        pts_local = [Xm(:)'; Ym(:)'; Zm(:)'];
                        pts_world = R * pts_local;
        
                        Xm_w = reshape(pts_world(1,:), size(Xm)) + pos(1);
                        Ym_w = reshape(pts_world(2,:), size(Ym)) + pos(2);
                        Zm_w = reshape(pts_world(3,:), size(Zm)) + pos(3);
        
                        set(h_motors{i,m_i}, ...
                            'XData', Xm_w, ...
                            'YData', Ym_w, ...
                            'ZData', Zm_w);
                    end
        
                    % Update propellers
                    for p_i = 1:4
                        offset = motorOffsets(p_i,:);
        
                        prop_local = [propX; propY; propZ];
                        prop_shift = prop_local + offset';
                        prop_world = R * prop_shift + pos';
        
                        set(h_props{i,p_i}, ...
                            'XData', prop_world(1,:), ...
                            'YData', prop_world(2,:), ...
                            'ZData', prop_world(3,:));
                    end
        
                    % Point marker
                    set(h_marker(i), ...
                         'XData', px, ...
                         'YData', py, ...
                         'ZData', pz);
        
                    % Dashed path
                    set(h_path(i), ...
                         'XData', xi(1,1:k), ...
                         'YData', xi(2,1:k), ...
                         'ZData', xi(3,1:k));
                end
        
                % Update time text
                t_handle.String = sprintf('t = %.2f s', (k-1)*self.Ts);
        
                % Capture frame
                frame = getframe(fig_anim);
                [imind, cm] = rgb2ind(frame2im(frame), 256);
        
                if k == 1
                    imwrite(imind, cm, gif_name, "gif", ...
                            "Loopcount", inf, "DelayTime", delay);
                else
                    imwrite(imind, cm, gif_name, "gif", ...
                            "WriteMode", "append", "DelayTime", delay);
                end
            end
        
            close(fig_anim);
        end
    end
end