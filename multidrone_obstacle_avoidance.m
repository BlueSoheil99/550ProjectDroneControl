x0 = zeros(12,1);
x0(3) = 0.0;                 % initial z
u_const = [0.1*9.8; 0; 0; 0];   % hover input

[T, X] = ode45(@(t,x) drone_dynamics(t,x,u_const), [0 2], x0);

%figure; plot(T, X(:,3));
%xlabel('Time (s)'); ylabel('z (m)'); title('Drone altitude vs time');

%return

% LQR Gain 
R = diag([1/15; 1000; 1000; 100]); 
Q = diag([0.1; 0.1; 10; 0.01; 0.01; 0.01; 0.01; 0.01; 1; 0.1; 0.1; 0.1]);

Ts = 0.02; % 20ms
[A, B, Ad, Bd] = linearized_drone(Ts);

[K,S,P] = lqr(Ad,Bd,Q,R);
