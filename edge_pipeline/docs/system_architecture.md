# System Architecture

## Pipeline Diagram

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│  Stereo      │     │  Object      │     │  Depth       │
│  Cameras     │────▶│  Detection   │────▶│  Extraction  │
│  (external)  │     │  (external)  │     │  (external)  │
└─────────────┘     └──────────────┘     └──────┬───────┘
                                                 │
                    ┌────────────────────────────▼────────────────┐
                    │           EDGE PIPELINE (this repo)         │
                    │                                             │
                    │  ┌─────────┐    ┌─────────────┐           │
                    │  │Transform│───▶│ Inverse     │           │
                    │  │pixel→   │    │ Kinematics  │           │
                    │  │base     │    │ (FK/IK)     │           │
                    │  └─────────┘    └──────┬──────┘           │
                    │                        │                   │
                    │  ┌─────────┐    ┌──────▼──────┐           │
                    │  │Collision│◀──▶│ Path        │           │
                    │  │Checker  │    │ Planning    │           │
                    │  └─────────┘    │ (APF/RRT*)  │           │
                    │                 └──────┬──────┘           │
                    │                        │                   │
                    │                 ┌──────▼──────┐           │
                    │                 │ Trajectory  │           │
                    │                 │ Smoothing   │           │
                    │                 └──────┬──────┘           │
                    │                        │                   │
                    │        ┌───────────────┼───────────┐      │
                    │        ▼               ▼           ▼      │
                    │  ┌──────────┐  ┌────────────┐ ┌────────┐ │
                    │  │ PCA9685  │  │ CSV/JSON   │ │Telemetry│ │
                    │  │ Servo    │  │ Export     │ │ Logs   │ │
                    │  │ Driver   │  │ (+SCP)    │ │ (JSON) │ │
                    │  └──────────┘  └────────────┘ └────────┘ │
                    └─────────────────────────────────────────────┘
```

## DH Parameter Template

Standard (Modified) DH convention — all joints are revolute.

| Joint | a (m) | α (rad) | d (m) | θ offset (rad) | q_min (rad) | q_max (rad) |
|-------|-------|---------|-------|-----------------|-------------|-------------|
| 1 (Base) | 0.000 | π/2 | 0.077 | 0.0 | -π | π |
| 2 (Shoulder) | 0.130 | 0.0 | 0.000 | 0.0 | -π | π |
| 3 (Elbow) | 0.124 | 0.0 | 0.000 | 0.0 | -π | π |
| 4 (Wrist)* | 0.126 | 0.0 | 0.000 | 0.0 | -π | π |

*Joint 4 is only used in 4-DOF configuration.

**To customise:** measure your arm's link lengths with calipers and update `DEFAULT_DH_3DOF` / `DEFAULT_DH_4DOF` in `src/kinematics/arm_kinematics.py`.

## MATLAB Ingestion

### Reading trajectory CSV:

```matlab
% Load the trajectory (% comments are auto-skipped)
data = readmatrix('trajectory.csv', 'CommentStyle', '%');

t = data(:, 1);           % Time vector (seconds)
q = data(:, 2:end);       % Joint angles (radians)

% Plot
figure;
plot(t, q, 'LineWidth', 1.5);
xlabel('Time (s)'); ylabel('Joint Angle (rad)');
legend(arrayfun(@(i) sprintf('q_%d', i), 1:size(q,2), 'UniformOutput', false));
title('Planned Trajectory');
grid on;
```

### Using in Simscape Multibody:

```matlab
% Create a timeseries for Simscape input
for i = 1:size(q, 2)
    joint_ts{i} = timeseries(q(:, i), t);
end

% In Simscape: use "Signal Builder" or "From Workspace" blocks
% to drive each Revolute Joint with joint_ts{i}
```
