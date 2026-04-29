% generate_fixtures.m
% Regenerate reference data for sensmaps v1 Python tests.
%
% Run this script in MATLAB with the working directory set to
%   tests/fixtures
% It will write cw_sd_i_example1.mat alongside itself.
%
% Requirements: DOIT-Public/SensitivityCompendium/deps on MATLAB path
%   relative to this repo: ../../../DOIT-Public/SensitivityCompendium/deps

clear; clc;

%% Locate the compendium deps
here = fileparts(mfilename('fullpath'));
deps = fullfile(here, '..', '..', '..', ...
    'DOIT-Public', 'SensitivityCompendium', 'deps');
assert(isfolder(deps), ...
    'Expected DOIT-Public compendium deps at: %s', deps);
addpath(deps);

%% Canonical inputs (smaller than example1_DT.m to keep fixture under 1 MB)
nin  = 1.333;
nout = 1.0;
musp = 1.1;
mua  = 0.011;

opt_prop = struct('nin', nin, 'nout', nout, 'musp', musp, 'mua', mua);

dr   = 1.0;          % mm
pert = [1, 1, 1];    % mm
xl   = [-5, 40];     % mm
yl   = [0, 0];       % mm
zl   = [0, 20];      % mm
rho  = 25.0;         % mm

rs = [0, 0, 1/musp];       % source (with z-offset)
rd = [rho, 0, 0];           % detector

%% n2A scalar references (sweep both sides of 1)
n_in_vals = [1.0, 1.333, 1.4, 1.5, 0.8];
A_vals = arrayfun(@(n) n2A(n, 1.0), n_in_vals);

%% Single-point references at r_test, omega=0 (CW)
r_test = [10, 0, 5];
phi_test_cw = complexFluence(rs, r_test, 0, opt_prop);
R_test_cw   = complexReflectance(rs, rd, 0, opt_prop);
[L_test_cw, ~] = complexTotPathLen(rs, rd, 0, opt_prop);
l_test_cw   = complexPartPathLen(rs, r_test, rd, dr^3, 0, opt_prop);

%% Full end-to-end CW_SD_I — mirrors makeS.m
x = xl(1):dr:xl(2);
y = yl(1):dr:yl(2);
z = zl(1):dr:zl(2);

[YY, XX, ZZ] = meshgrid(y, x, z);
r_all = [XX(:), YY(:), ZZ(:)];

[L_scalar, ~] = complexTotPathLen(rs, rd, 0, opt_prop);
L = real(L_scalar);

l_vec = complexPartPathLen(rs, r_all, rd, dr^3, 0, opt_prop);
l_vec(isnan(l_vec)) = 0;
l_vec = real(l_vec);
ll = reshape(l_vec, size(XX));

Svox = ll / L;

H = ones(pert / dr);
S = convn(Svox, H, 'same');

%% Save fixture — individual vars (structs load awkwardly via scipy.io.loadmat)
save(fullfile(here, 'cw_sd_i_example1.mat'), ...
    'nin', 'nout', 'musp', 'mua', ...
    'dr', 'pert', 'xl', 'yl', 'zl', 'rho', ...
    'rs', 'rd', ...
    'n_in_vals', 'A_vals', ...
    'r_test', 'phi_test_cw', 'R_test_cw', 'L_test_cw', 'l_test_cw', ...
    'x', 'y', 'z', 'Svox', 'S');

%% v1.1 fixtures — 8 new combos
% Common inputs (same grid as cw_sd_i_example1)
combos_xl   = [-5, 40];
combos_yl   = [0, 0];
combos_zl   = [0, 20];
combos_dr   = 1.0;
combos_pert = [1, 1, 1];
combos_fmod = 100e6;   % Hz (matches GUI default of 100 MHz)

% Optode geometries per arrangement
sd_rs = [0, 0, 0];           sd_rd = [25, 0, 0];
ss_rs = [0, 0, 0];           ss_rd = [20, 0, 0; 30, 0, 0];   % 1×2 form
ds_rs = [0, 0, 0; 5, 0, 0]; ds_rd = [25, 0, 0; 30, 0, 0];

combo_specs = {
    'cw_ss_i', 'CW_SS_I', ss_rs, ss_rd, NaN;
    'cw_ds_i', 'CW_DS_I', ds_rs, ds_rd, NaN;
    'fd_sd_i', 'FD_SD_I', sd_rs, sd_rd, combos_fmod;
    'fd_sd_p', 'FD_SD_P', sd_rs, sd_rd, combos_fmod;
    'fd_ss_i', 'FD_SS_I', ss_rs, ss_rd, combos_fmod;
    'fd_ss_p', 'FD_SS_P', ss_rs, ss_rd, combos_fmod;
    'fd_ds_i', 'FD_DS_I', ds_rs, ds_rd, combos_fmod;
    'fd_ds_p', 'FD_DS_P', ds_rs, ds_rd, combos_fmod;
};

for k = 1:size(combo_specs, 1)
    name      = combo_specs{k, 1};
    type_str  = combo_specs{k, 2};
    rs        = combo_specs{k, 3};
    rd        = combo_specs{k, 4};
    fmod_hz   = combo_specs{k, 5};

    if isnan(fmod_hz)
        [S, params, Svox] = makeS(type_str, rs, rd, opt_prop, ...
            'xl', combos_xl, 'yl', combos_yl, 'zl', combos_zl, ...
            'dr', combos_dr, 'pert', combos_pert);
    else
        [S, params, Svox] = makeS(type_str, rs, rd, opt_prop, ...
            'xl', combos_xl, 'yl', combos_yl, 'zl', combos_zl, ...
            'dr', combos_dr, 'pert', combos_pert, 'fmod', fmod_hz);
    end

    x = params.x; y = params.y; z = params.z;
    xl = combos_xl; yl = combos_yl; zl = combos_zl;
    dr = combos_dr; pert = combos_pert;

    save(fullfile(here, [name, '.mat']), ...
        'nin', 'nout', 'musp', 'mua', ...
        'dr', 'pert', 'xl', 'yl', 'zl', ...
        'rs', 'rd', 'fmod_hz', ...
        'x', 'y', 'z', 'Svox', 'S', 'type_str');
    fprintf('Wrote %s.mat\n', name);
end

%% v1.2 fixtures — TD_*_GI (3 combos)
combos_tend = 10e3;          % ps
combos_ndt  = 10e3;
combos_tg   = [1000; 2000];  % ps (1-2 ns gate, late)
combos_tgE  = [500; 1500];   % ps (0.5-1.5 ns gate, early; for DGI)

td_specs = {
    'td_sd_gi',  'TD_SD_GI',  sd_rs, sd_rd;
    'td_ss_gi',  'TD_SS_GI',  ss_rs, ss_rd;
    'td_ds_gi',  'TD_DS_GI',  ds_rs, ds_rd;
    % v1.3: T and V over SD/SS/DS, plus DGI for SD only
    'td_sd_t',   'TD_SD_T',   sd_rs, sd_rd;
    'td_ss_t',   'TD_SS_T',   ss_rs, ss_rd;
    'td_ds_t',   'TD_DS_T',   ds_rs, ds_rd;
    'td_sd_v',   'TD_SD_V',   sd_rs, sd_rd;
    'td_ss_v',   'TD_SS_V',   ss_rs, ss_rd;
    'td_ds_v',   'TD_DS_V',   ds_rs, ds_rd;
    'td_sd_dgi', 'TD_SD_DGI', sd_rs, sd_rd;
};

for k = 1:size(td_specs, 1)
    name      = td_specs{k, 1};
    type_str  = td_specs{k, 2};
    rs        = td_specs{k, 3};
    rd        = td_specs{k, 4};

    [S, params, Svox] = makeS(type_str, rs, rd, opt_prop, ...
        'xl', combos_xl, 'yl', combos_yl, 'zl', combos_zl, ...
        'dr', combos_dr, 'pert', combos_pert, ...
        'tg', combos_tg, 'tgE', combos_tgE, ...
        'tend', combos_tend, 'ndt', combos_ndt);

    x = params.x; y = params.y; z = params.z;
    xl = combos_xl; yl = combos_yl; zl = combos_zl;
    dr = combos_dr; pert = combos_pert;
    tg_ps = combos_tg; tgE_ps = combos_tgE;
    tend_ps = combos_tend; ndt = combos_ndt;

    save(fullfile(here, [name, '.mat']), ...
        'nin', 'nout', 'musp', 'mua', ...
        'dr', 'pert', 'xl', 'yl', 'zl', ...
        'rs', 'rd', 'tg_ps', 'tgE_ps', 'tend_ps', 'ndt', ...
        'x', 'y', 'z', 'Svox', 'S', 'type_str');
    fprintf('Wrote %s.mat\n', name);
end

%% Single-point physics references for TD primitives (test_physics.py)
td_op    = opt_prop;
td_rs    = [0, 0, 1/musp];   % source with z-offset
td_rd    = [25, 0, 0];
td_t     = [-100, 0, 100, 500, 1000, 2000, 5000];   % ps
td_R_t   = temporalReflectance(td_rs, td_rd, td_t, td_op);
td_PHI_t = temporalFluence(td_rs, td_rd, td_t, td_op);
td_tg    = [1000; 2000];     % ps
td_L_gate = temporalGateTotPathLen(td_rs, td_rd, td_tg, td_op, ...
    'conv_t', combos_tend, 'conv_dt', combos_tend/combos_ndt);

% Partial path-length single-voxel reference
td_r_test = [10, 0, 5];
td_l_gate = temporalGatePartPathLen(td_rs, td_r_test, td_rd, ...
    combos_dr^3, td_tg, td_op, ...
    'conv_t', combos_tend, 'conv_dt', combos_tend/combos_ndt, ...
    'usePar', false, 'FFTconv', true);

% v1.3: Kth-moment and variance primitive references
td_kth_mom_t1 = temporalKthMoment(td_rs, td_rd, 1, td_op);          % ps
td_kth_mom_t2 = temporalKthMoment(td_rs, td_rd, 2, td_op);          % ps^2
td_kth_mom_tot_t1 = temporalKthMomTotPathLen(td_rs, td_rd, 1, td_op);  % mm
td_kth_mom_tot_t2 = temporalKthMomTotPathLen(td_rs, td_rd, 2, td_op);  % mm
td_kth_mom_part_t1 = temporalKthMomPartPathLen(td_rs, td_r_test, td_rd, ...
    combos_dr^3, 1, td_op, ...
    'conv_t', combos_tend, 'conv_dt', combos_tend/combos_ndt, ...
    'usePar', false, 'FFTconv', true);                              % mm
td_kth_mom_part_t2 = temporalKthMomPartPathLen(td_rs, td_r_test, td_rd, ...
    combos_dr^3, 2, td_op, ...
    'conv_t', combos_tend, 'conv_dt', combos_tend/combos_ndt, ...
    'usePar', false, 'FFTconv', true);                              % mm
td_var = temporalVar(td_rs, td_rd, td_op);                          % ps^2
td_var_tot = temporalVarTotPathLen(td_rs, td_rd, td_op);            % mm
td_var_part = temporalVarPartPathLen(td_rs, td_r_test, td_rd, ...
    combos_dr^3, td_op, ...
    'conv_t', combos_tend, 'conv_dt', combos_tend/combos_ndt, ...
    'usePar', false, 'FFTconv', true);                              % mm

save(fullfile(here, 'td_physics_refs.mat'), ...
    'nin', 'nout', 'musp', 'mua', ...
    'td_rs', 'td_rd', 'td_t', 'td_R_t', 'td_PHI_t', ...
    'td_tg', 'td_L_gate', 'td_r_test', 'td_l_gate', ...
    'td_kth_mom_t1', 'td_kth_mom_t2', ...
    'td_kth_mom_tot_t1', 'td_kth_mom_tot_t2', ...
    'td_kth_mom_part_t1', 'td_kth_mom_part_t2', ...
    'td_var', 'td_var_tot', 'td_var_part', ...
    'combos_tend', 'combos_ndt');
fprintf('Wrote td_physics_refs.mat\n');

rmpath(deps);

fprintf('Wrote %s\n', fullfile(here, 'cw_sd_i_example1.mat'));
