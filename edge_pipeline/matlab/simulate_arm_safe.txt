%% =========================================================================
%  simulate_arm.m - 3-DOF Robotic Arm Trajectory Visualization
%  =========================================================================
%  Reads the CSV trajectory from the edge pipeline and produces:
%    1. Joint angle plots (q vs time)
%    2. Joint velocity plots (dq/dt vs time)
%    3. 3D animated arm visualization
%    4. End-effector path trace
%
%  USAGE:
%    >> simulate_arm
%    then press any key to start animation
%  =========================================================================
clear; clc; close all;

%% -- 1. Configuration ----------------------------------------------------
% DH parameters matching edge_pipeline/src/kinematics/arm_kinematics.py
% [a(m), alpha(rad), d(m), theta_offset(rad)]
DH = [
    0.000,  pi/2,  0.077,  0;   % Joint 1 (Base rotation)
    0.130,  0.0,   0.000,  0;   % Joint 2 (Shoulder)
    0.124,  0.0,   0.000,  0;   % Joint 3 (Elbow)
];
n_joints = size(DH, 1);

% Link colors for visualization
link_colors = [
    0.2, 0.4, 0.8;
    0.8, 0.2, 0.2;
    0.2, 0.7, 0.3;
];

%% -- 2. Load Trajectory ---------------------------------------------------
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

fprintf('Loading trajectory: %s\n', csv_file);

data = readmatrix(csv_file, 'CommentStyle', '%');
t = data(:, 1);           % Time vector (seconds)
q = data(:, 2:end);       % Joint angles (radians)
N = size(q, 1);

fprintf('  %d waypoints, %.2f s duration, %d DOF\n', N, t(end), size(q, 2));

%% -- 3. Forward Kinematics Function --------------------------------------
% Returns [base; joint1; joint2; joint3] positions (4x3)
fk = @(q_in) forward_kinematics(DH, q_in);

%% -- 4. Plot Joint Angles ------------------------------------------------
figure('Name', 'Joint Angles', 'Position', [50, 400, 700, 400]);

subplot(2, 1, 1);
plot(t, rad2deg(q), 'LineWidth', 1.8);
xlabel('Time (s)');
ylabel('Angle (deg)');
title('Joint Angles vs Time');
legend(arrayfun(@(i) sprintf('q_%d', i), 1:n_joints, 'UniformOutput', false));
grid on;
set(gca, 'FontSize', 11);

% Joint velocities (numerical differentiation)
dq = diff(q) ./ diff(t);

subplot(2, 1, 2);
plot(t(1:end-1), rad2deg(dq), 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Velocity (deg/s)');
title('Joint Velocities vs Time');
legend(arrayfun(@(i) sprintf('dq_%d/dt', i), 1:n_joints, 'UniformOutput', false));
grid on;
set(gca, 'FontSize', 11);

%% -- 5. Compute All End-Effector Positions --------------------------------
ee_path = zeros(N, 3);
for k = 1:N
    joints = fk(q(k, :));
    ee_path(k, :) = joints(end, :);
end

%% -- 6. 3D Arm Animation -------------------------------------------------
fig_anim = figure('Name', 'Arm Animation', ...
    'Position', [100, 50, 900, 700], ...
    'Color', [0.12, 0.12, 0.15]);

fprintf('\nPress any key to start animation...\n');
pause;

% Animation parameters
skip = max(1, floor(N / 200));  % Show ~200 frames max

for k = 1:skip:N
    clf;
    joints = fk(q(k, :));  % (n_joints+1) x 3

    hold on;

    % Draw arm links
    for j = 1:size(joints, 1) - 1
        p1 = joints(j, :);
        p2 = joints(j + 1, :);

        % Thick link
        plot3([p1(1), p2(1)], [p1(2), p2(2)], [p1(3), p2(3)], ...
            '-', 'Color', link_colors(min(j, size(link_colors, 1)), :), ...
            'LineWidth', 6);

        % Joint sphere
        scatter3(p2(1), p2(2), p2(3), 80, ...
            link_colors(min(j, size(link_colors, 1)), :), 'filled', ...
            'MarkerEdgeColor', 'w', 'LineWidth', 1.5);
    end

    % Base marker
    scatter3(0, 0, 0, 120, [0.9, 0.9, 0.2], 'filled', 'MarkerEdgeColor', 'w');

    % End-effector cross-hair
    ee = joints(end, :);
    scatter3(ee(1), ee(2), ee(3), 100, 'r', 'x', 'LineWidth', 3);

    % Trace path so far
    idx = 1:skip:k;
    plot3(ee_path(idx, 1), ee_path(idx, 2), ee_path(idx, 3), ...
        '--', 'Color', [1, 0.6, 0.2, 0.5], 'LineWidth', 1.5);

    % Ground plane
    patch([-0.3, 0.3, 0.3, -0.3], [-0.3, -0.3, 0.3, 0.3], [0, 0, 0, 0], ...
        [0.2, 0.2, 0.25], 'FaceAlpha', 0.3, 'EdgeColor', [0.4, 0.4, 0.4]);

    hold off;

    % Axes formatting
    axis equal;
    xlim([-0.4, 0.4]);
    ylim([-0.4, 0.4]);
    zlim([-0.05, 0.4]);
    xlabel('X (m)');
    ylabel('Y (m)');
    zlabel('Z (m)');
    title(sprintf('3-DOF Arm  |  t = %.3f s  |  frame %d/%d', t(k), k, N), ...
        'Color', 'w', 'FontSize', 13);

    view(135, 25);
    set(gca, 'Color', [0.15, 0.15, 0.18], ...
        'XColor', 'w', 'YColor', 'w', 'ZColor', 'w', ...
        'GridColor', [0.3, 0.3, 0.35], 'FontSize', 10);
    grid on;

    drawnow;
end

fprintf('Animation complete.\n');

%% -- 7. Static End-Effector Path ------------------------------------------
figure('Name', 'End-Effector Path', 'Position', [800, 400, 600, 500]);
plot3(ee_path(:, 1), ee_path(:, 2), ee_path(:, 3), ...
    '-', 'LineWidth', 2, 'Color', [0.2, 0.6, 0.9]);
hold on;
scatter3(ee_path(1, 1), ee_path(1, 2), ee_path(1, 3), 100, 'g', 'filled');
scatter3(ee_path(end, 1), ee_path(end, 2), ee_path(end, 3), 100, 'r', 'filled');
hold off;
xlabel('X (m)');
ylabel('Y (m)');
zlabel('Z (m)');
title('End-Effector Path');
legend('Path', 'Start', 'Goal');
axis equal;
grid on;
view(135, 25);

fprintf('\n=== Summary ===\n');
fprintf('  Waypoints:     %d\n', N);
fprintf('  Duration:      %.3f s\n', t(end));
fprintf('  EE start:      [%.4f, %.4f, %.4f] m\n', ee_path(1, :));
fprintf('  EE end:        [%.4f, %.4f, %.4f] m\n', ee_path(end, :));
fprintf('  Max velocity:  %.1f deg/s\n', max(abs(dq(:))) * 180 / pi);
fprintf('==================\n');

%% =========================================================================
%  Forward Kinematics (DH Convention)
%  =========================================================================
function joints = forward_kinematics(DH, q)
    % Returns (n+1) x 3 matrix of joint positions (including base at origin)
    n = size(DH, 1);
    T = eye(4);
    joints = zeros(n + 1, 3);
    joints(1, :) = [0, 0, 0];  % Base origin

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
        joints(i + 1, :) = T(1:3, 4)';
    end
end
