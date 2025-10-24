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

