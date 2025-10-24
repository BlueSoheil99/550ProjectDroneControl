function xdot = drone_dynamics(t, x, u)
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

eta2dot = J \ paren;

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