x0 = zeros(12,1);
x0(3) = 0.0;                 % initial z
u_const = [0.063*9.8; 0; 0; 0];   % hover input

[T, X] = ode45(@(t,x) drone_dynamics(t,x,u_const), [0 2], x0);

figure; plot(T, X(:,3));
xlabel('Time (s)'); ylabel('z (m)'); title('Drone altitude vs time');




