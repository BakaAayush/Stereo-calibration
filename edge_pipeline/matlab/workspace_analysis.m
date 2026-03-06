%% =========================================================================
%  workspace_analysis.m - Arm Workspace & Trajectory Analysis
%  =========================================================================
%  Generates:
%    1. Reachable workspace point cloud (Monte Carlo sampling)
%    2. Trajectory overlaid on workspace
%    3. Joint torque estimation (static)
%    4. Workspace cross-sections (XZ and XY planes)
%
%  USAGE:
%    >> workspace_analysis
%  =========================================================================
clear; clc; close all;

%% -- 1. Configuration ----------------------------------------------------
DH = [
    0.000,  pi/2,  0.077,  0;
    0.130,  0.0,   0.000,  0;
    0.124,  0.0,   0.000,  0;
];
n_joints = size(DH, 1);

% Joint limits (rad)
q_min = [-pi, -pi, -pi];
q_max = [ pi,  pi,  pi];

% Link masses (kg) - approximate for hobby arm
link_mass = [0.05, 0.04, 0.03];
gravity = 9.81;  % m/s^2

%% -- 2. Monte Carlo Workspace Sampling ------------------------------------
fprintf('Sampling workspace (10,000 random configs)...\n');
N_samples = 10000;
ws_points = zeros(N_samples, 3);

rng(42);  % Reproducible
for k = 1:N_samples
    q_rand = q_min + (q_max - q_min) .* rand(1, n_joints);
    joints = forward_kinematics(DH, q_rand);
    ws_points(k, :) = joints(end, :);  % End-effector position
end

%% -- 3. Load Trajectory --------------------------------------------------
csv_file = '../output/matlab_scenarios/scenario_pick_sequence.csv';
if ~isfile(csv_file)
    csv_file = '../output/trajectory_3dof.csv';
end
if ~isfile(csv_file)
    csv_file = 'output/matlab_scenarios/scenario_pick_sequence.csv';
end
if ~isfile(csv_file)
    csv_file = 'output/trajectory_3dof.csv';
end

if ~isfile(csv_file)
    error('No trajectory CSV found! Run generate_matlab_data.py first.');
end

data = readmatrix(csv_file, 'CommentStyle', '%');
t = data(:, 1);
q = data(:, 2:end);
N = size(q, 1);

% Compute EE positions for trajectory
ee_traj = zeros(N, 3);
for k = 1:N
    joints = forward_kinematics(DH, q(k,:));
    ee_traj(k,:) = joints(end,:);
end

%% -- 4. Plot: 3D Workspace + Trajectory -----------------------------------
figure('Name', 'Workspace Analysis', 'Position', [50, 50, 1200, 500]);

subplot(1,2,1);
scatter3(ws_points(:,1), ws_points(:,2), ws_points(:,3), ...
    3, [0.6 0.7 0.85], 'filled', 'MarkerFaceAlpha', 0.15);
hold on;
plot3(ee_traj(:,1), ee_traj(:,2), ee_traj(:,3), ...
    '-', 'Color', [1 0.3 0.2], 'LineWidth', 2.5);
scatter3(0, 0, 0, 150, [1 0.9 0.2], 'filled', 'MarkerEdgeColor', 'k');
scatter3(ee_traj(1,1), ee_traj(1,2), ee_traj(1,3), 80, 'g', 'filled');
scatter3(ee_traj(end,1), ee_traj(end,2), ee_traj(end,3), 80, 'r', 'filled');
hold off;
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
title('Reachable Workspace + Planned Trajectory');
legend('Workspace', 'Trajectory', 'Base', 'Start', 'Goal', ...
    'Location', 'best');
axis equal; grid on; view(135, 25);

%% -- 5. Plot: XZ Cross-Section -------------------------------------------
subplot(1,2,2);
% Filter workspace points near Y=0
y_tol = 0.02;
mask = abs(ws_points(:,2)) < y_tol;
scatter(ws_points(mask,1), ws_points(mask,3), ...
    5, [0.6 0.7 0.85], 'filled', 'MarkerFaceAlpha', 0.3);
hold on;
plot(ee_traj(:,1), ee_traj(:,3), '-r', 'LineWidth', 2);
scatter(0, 0, 100, [1 0.9 0.2], 'filled', 'MarkerEdgeColor', 'k');
hold off;
xlabel('X (m)'); ylabel('Z (m)');
title('Workspace Cross-Section (XZ plane, Y~0)');
axis equal; grid on;

%% -- 6. Static Torque Estimation -----------------------------------------
fprintf('\nEstimating static joint torques along trajectory...\n');
torques = zeros(N, n_joints);

for k = 1:N
    joints = forward_kinematics(DH, q(k,:));

    for j = 1:n_joints
        % Torque at joint j = sum of (mass * g * horizontal distance)
        % for all links distal to joint j
        tau = 0;
        for m = j:n_joints
            % Center of mass approximation: midpoint of link
            if m < size(joints,1)
                com = (joints(m,:) + joints(m+1,:)) / 2;
            else
                com = joints(end,:);
            end
            % Horizontal distance from joint j axis
            r_horiz = sqrt(com(1)^2 + com(2)^2);
            tau = tau + link_mass(m) * gravity * r_horiz;
        end
        torques(k, j) = tau;
    end
end

figure('Name', 'Torque Analysis', 'Position', [50, 600, 700, 350]);
plot(t, torques * 100, 'LineWidth', 1.5);  % Convert to N*cm
xlabel('Time (s)'); ylabel('Static Torque (N*cm)');
title('Estimated Static Joint Torques');
legend(arrayfun(@(i) sprintf('Joint %d', i), 1:n_joints, 'UniformOutput', false));
grid on; set(gca, 'FontSize', 11);

fprintf('\nPeak torques:\n');
for j = 1:n_joints
    fprintf('  Joint %d: %.2f N*cm (%.4f N*m)\n', ...
        j, max(abs(torques(:,j)))*100, max(abs(torques(:,j))));
end

%% -- 7. Workspace Statistics ----------------------------------------------
max_reach = max(sqrt(sum(ws_points.^2, 2)));
min_reach = min(sqrt(sum(ws_points.^2, 2)));

fprintf('\n=== Workspace Statistics ===\n');
fprintf('  Max reach:  %.3f m\n', max_reach);
fprintf('  Min reach:  %.3f m\n', min_reach);
fprintf('  X range:    [%.3f, %.3f] m\n', min(ws_points(:,1)), max(ws_points(:,1)));
fprintf('  Y range:    [%.3f, %.3f] m\n', min(ws_points(:,2)), max(ws_points(:,2)));
fprintf('  Z range:    [%.3f, %.3f] m\n', min(ws_points(:,3)), max(ws_points(:,3)));
fprintf('  Trajectory: %.3f m total EE travel\n', ...
    sum(sqrt(sum(diff(ee_traj).^2, 2))));
fprintf('============================\n');

%% =========================================================================
function joints = forward_kinematics(DH, q)
    n = size(DH, 1);
    T = eye(4);
    joints = zeros(n+1, 3);
    joints(1, :) = [0, 0, 0];

    for i = 1:n
        a     = DH(i, 1);
        alpha = DH(i, 2);
        d     = DH(i, 3);
        theta = q(i) + DH(i, 4);

        ct = cos(theta); st = sin(theta);
        ca = cos(alpha); sa = sin(alpha);

        A = [ct,  -st*ca,   st*sa,  a*ct;
             st,   ct*ca,  -ct*sa,  a*st;
              0,      sa,      ca,     d;
              0,       0,       0,     1];

        T = T * A;
        joints(i+1, :) = T(1:3, 4)';
    end
end
