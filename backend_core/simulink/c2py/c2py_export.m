function zip_file = c2py_export(model_name)
% C2PY_EXPORT  Prepare a Simulink model for the C_to_Python service.
%
%   HOW TO USE
%     1. Put this file in the folder of your Simulink model (.slx or .mdl).
%        If the model needs variables from a script, put that script next to it as setup.m
%        (it is run first). Scripts the model runs itself (PreLoadFcn / InitFcn) just work.
%     2. In MATLAB, go to that folder and run
%            c2py_export                  % the only model in the folder
%            c2py_export('my_model')      % if the folder holds more than one model
%     3. Upload the file it creates:  <model>_c2py.zip
%
%   WHAT IT DOES
%     - applies the code generation settings below and the fixed-step solver ode8
%       (in memory only: YOUR MODEL FILE IS NOT CHANGED OR SAVED)
%     - generates the C code with Embedded Coder
%     - simulates the model with a few test inputs (step, sine, PRBS)
%     - packs the C files, the simulation results and a manifest into ONE file next to
%       the model: <model>_c2py.zip (the temporary files are deleted)
%
%   Requires MATLAB, Simulink, Simulink Coder and Embedded Coder.

    EXPORT_VERSION = '1.0';

    % ======================= settings (do not change unless you were asked to) ============
    S.solver   = 'ode8';                  % fixed-step solver for the code AND the simulations
    S.profile  = 'readable';              % readable | readable_folded | original
    S.tf       = 10;                      % stop time of the test simulations [s]
    S.amp      = 2;                       % amplitude of the test inputs
    S.signals  = {'step', 'sin', 'prbs'}; % test inputs
    S.seed     = 0;                       % PRBS seed
    S.setup    = 'setup.m';               % optional script next to the model, run first
    % =====================================================================================

    here = fileparts(mfilename('fullpath'));
    old_dir = pwd;
    back = onCleanup(@() cd(old_dir)); %#ok<NASGU>
    cd(here);

    % ---- 1. which model
    if nargin < 1 || isempty(model_name)
        cands = [dir(fullfile(here, '*.slx')); dir(fullfile(here, '*.mdl'))];
        names = unique(regexprep({cands.name}, '\.(slx|mdl)$', ''));
        if isempty(names)
            error('c2py_export:nomodel', 'No .slx / .mdl file next to c2py_export.m (%s).', here);
        elseif numel(names) > 1
            error('c2py_export:many', ['More than one model in %s: %s\n' ...
                  'Run c2py_export(''<model name>'') with the one you want.'], here, strjoin(names, ', '));
        end
        model_name = names{1};
    end
    model_name = char(regexprep(char(model_name), '\.(slx|mdl)$', ''));
    if ~exist(fullfile(here, [model_name '.slx']), 'file') && ~exist(fullfile(here, [model_name '.mdl']), 'file')
        error('c2py_export:nomodel', 'Model "%s" not found in %s.', model_name, here);
    end
    fprintf('\n=== c2py_export %s : model %s ===\n', EXPORT_VERSION, model_name);

    % ---- 2. optional setup script
    if ~isempty(S.setup) && exist(fullfile(here, S.setup), 'file')
        fprintf('[setup] running %s\n', S.setup);
        evalin('base', sprintf('run(''%s'');', strrep(fullfile(here, S.setup), '''', '''''')));
    end

    % ---- 3. temporary folder: build + package. Only the zip ends up next to the model, so
    %         there is exactly one file to upload (and nothing half-made if something fails).
    pkg_name = [model_name '_c2py'];
    zip_file = fullfile(here, [pkg_name '.zip']);
    if exist(zip_file, 'file'), delete(zip_file); end
    if exist(fullfile(here, pkg_name), 'dir')               % left over by an older version of this script
        rmdir(fullfile(here, pkg_name), 's');
    end
    build_dir = tempname;
    mkdir(build_dir);
    old_cfg = Simulink.fileGenControl('getConfig');
    restore_cfg = onCleanup(@() local_restore(old_cfg, build_dir)); %#ok<NASGU>
    pkg_root = fullfile(build_dir, 'package');
    pkg_dir  = fullfile(pkg_root, pkg_name);
    mkdir(fullfile(pkg_dir, 'code'));
    mkdir(fullfile(pkg_dir, 'simulink'));
    Simulink.fileGenControl('set', 'CodeGenFolder', build_dir, 'CacheFolder', build_dir, ...
        'CodeGenFolderStructure', Simulink.filegen.CodeGenFolderStructure.ModelSpecific, ...
        'createDir', true);

    if bdIsLoaded(model_name), close_system(model_name, 0); end
    load_system(model_name);
    close_model = onCleanup(@() local_close(model_name)); %#ok<NASGU>
    solver_in_model = {get_param(model_name, 'Solver'), get_param(model_name, 'SolverType')};

    % ---- 4. settings + code generation
    not_applied = local_settings(model_name, S.profile, S.solver);
    fprintf('[codegen] Embedded Coder ...\n');
    slbuild(model_name);
    rtw = fullfile(build_dir, [model_name '_ert_rtw']);
    if ~exist(rtw, 'dir')
        error('c2py_export:nocode', 'Code generation did not create %s.', rtw);
    end

    % ---- 5. the C / H files
    src = [dir(fullfile(rtw, '*.c')); dir(fullfile(rtw, '*.h'))];
    shared = fullfile(build_dir, 'slprj', 'ert', '_sharedutils');
    if exist(shared, 'dir')
        src = [src; dir(fullfile(shared, '*.c')); dir(fullfile(shared, '*.h'))];
    end
    files = struct('name', {}, 'bytes', {});
    for i = 1:numel(src)
        nm = src(i).name;
        if ~isempty(regexpi(nm, 'main\w*\.c$', 'once')), continue; end          % sample main: not needed
        if any(strcmp({files.name}, nm)), continue; end
        copyfile(fullfile(src(i).folder, nm), fullfile(pkg_dir, 'code', nm));
        files(end+1) = struct('name', nm, 'bytes', src(i).bytes); %#ok<AGROW>
    end
    fprintf('[codegen] %d C/H files collected\n', numel(files));

    % step size of the generated code (the simulations use exactly this one)
    ctext = fileread(fullfile(rtw, [model_name '.c']));
    tok = regexp(ctext, 'Timing\.stepSize0\s*=\s*([-+0-9.eE]+)\s*;', 'tokens', 'once');
    if isempty(tok)
        error('c2py_export:step', 'Could not find the step size in %s.c (is the solver fixed-step?).', model_name);
    end
    dt = str2double(tok{1});

    % ---- 6. test simulations
    inports = find_system(model_name, 'SearchDepth', 1, 'BlockType', 'Inport');
    outports = find_system(model_name, 'SearchDepth', 1, 'BlockType', 'Outport');
    interp = cell(1, numel(inports));
    for i = 1:numel(inports), interp{i} = get_param(inports{i}, 'Interpolate'); end
    n_in = numel(inports);
    t = (0:round(S.tf / dt))' * dt;
    sigs = S.signals;
    if n_in == 0, sigs = {'free'}; end                  % no inputs: one free run
    runs = struct('name', {}, 'input_file', {}, 'simulink_file', {}, 'samples', {}, 't_end', {}, 'error', {});
    n_out = 0;
    for k = 1:numel(sigs)
        sig = sigs{k};
        U = local_inputs(sig, t, n_in, S.amp, S.seed);
        fprintf('[simulink] %-5s %d steps of %g s ...\n', sig, numel(t) - 1, dt);
        [t_o, y_o, err] = local_sim(model_name, t, U, S.solver, dt);
        n_out = size(y_o, 2);
        f_in = sprintf('%s_%s_input.csv', model_name, sig);
        f_sim = sprintf('%s_%s_simulink.csv', model_name, sig);
        local_csv(fullfile(pkg_dir, 'simulink', f_in), [t, U], [{'t'}, arrayfun(@(j) sprintf('u%d', j), 1:n_in, 'UniformOutput', false)]);
        local_csv(fullfile(pkg_dir, 'simulink', f_sim), [t_o, y_o], [{'t'}, arrayfun(@(j) sprintf('y%d', j), 1:n_out, 'UniformOutput', false)]);
        if ~isempty(err)
            fprintf('[simulink]       stopped at t = %g s: %s\n', t_o(end), err);
        end
        runs(end+1) = struct('name', sig, 'input_file', f_in, 'simulink_file', f_sim, ...
                             'samples', numel(t_o), 't_end', t_o(end), 'error', err); %#ok<AGROW>
    end

    % ---- 7. manifest + zip
    m = struct();
    m.format = 'c2py-package';
    m.version = EXPORT_VERSION;
    m.model = model_name;
    m.created = char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss'));
    m.matlab_release = version('-release');
    m.solver = S.solver;
    m.solver_in_model = solver_in_model{1};
    m.solver_type_in_model = solver_in_model{2};
    m.fixed_step = dt;
    m.profile = S.profile;
    m.settings_not_applied = not_applied;
    m.n_inputs = n_in;
    m.n_outputs = n_out;
    m.inport_interpolation = interp;
    m.inports = regexprep(inports, ['^' regexptranslate('escape', model_name) '/'], '');
    m.outports = regexprep(outports, ['^' regexptranslate('escape', model_name) '/'], '');
    m.tf = S.tf;
    m.amp = S.amp;
    m.files = files;
    m.runs = runs;
    fid = fopen(fullfile(pkg_dir, 'c2py_manifest.json'), 'w');
    try
        txt = jsonencode(m, 'PrettyPrint', true);       % R2021a and later
    catch
        txt = jsonencode(m);
    end
    fwrite(fid, txt);
    fclose(fid);
    zip(zip_file, pkg_name, pkg_root);

    fprintf('\n=== done ===\n');
    fprintf('  %d C/H files, %d test simulation(s), step size %g s, solver %s\n', numel(files), numel(runs), dt, S.solver);
    fprintf('  UPLOAD THIS FILE:  %s\n\n', zip_file);
end


% =====================================================================================
function not_applied = local_settings(model, profile, solver)
% same settings as tools/codegen_settings.m of the C_to_Python project
    slv = {'SolverType', 'Fixed-step'; 'Solver', solver};
    base = {
        'SystemTargetFile',         'ert.tlc'
        'TargetLang',               'C'
        'GenCodeOnly',              'on'
        'GenerateReport',           'off'
        'LaunchReport',             'off'
        'GenerateSampleERTMain',    'off'
        'GenerateComments',         'on'
        'CodeInterfacePackaging',   'Nonreusable function'
        'SupportContinuousTime',    'on'
        'OptimizationLevel',        'level0'
        'InstructionSetExtensions', 'None'
        'CombineOutputUpdateFcns',  'off'
    };
    readable = {
        'DefaultParameterBehavior', 'Tunable'
        'OptimizeBlockIOStorage',   'off'
        'LocalBlockOutputs',        'off'
        'BufferReuse',              'off'
        'InlineInvariantSignals',   'off'
        'RollThreshold',            '2'
        'MaxIdLength',              '128'
        'SimulinkBlockComments',    'on'
        'ShowEliminatedStatement',  'on'
        'ForceParamTrailComments',  'on'
        'MATLABSourceComments',     'on'
        'MATLABFcnDesc',            'on'
        'InsertBlockDesc',          'on'
    };
    switch lower(profile)
        case 'original',        extra = cell(0, 2);
        case 'readable',        extra = [readable; {'ExpressionFolding', 'off'}];
        case 'readable_folded', extra = [readable; {'ExpressionFolding', 'on'}];
        otherwise, error('c2py_export:profile', 'Unknown profile "%s".', profile);
    end
    params = [slv; base; extra];
    not_applied = {};
    for i = 1:size(params, 1)
        try
            set_param(model, params{i, 1}, params{i, 2});
        catch err
            not_applied{end+1} = params{i, 1}; %#ok<AGROW>
            warning('c2py_export:param', 'Could not set %s: %s', params{i, 1}, err.message);
        end
    end
    fprintf('[codegen] profile %s, solver %s, %d settings applied\n', profile, solver, ...
            size(params, 1) - numel(not_applied));
end


function U = local_inputs(kind, t, n_in, amp, seed)
% the test inputs of the C_to_Python project (src/validation.py make_inputs)
    prev = rng;                                         % the random generator is given back as it was
    restore_rng = onCleanup(@() rng(prev)); %#ok<NASGU>
    tf = t(end) - t(1); if tf == 0, tf = 1; end
    if numel(t) > 1, dt = t(2) - t(1); else, dt = 1; end
    U = zeros(numel(t), n_in);
    for ch = 1:n_in
        c = ch - 1;
        switch kind
            case 'step'
                U(:, ch) = amp * (t >= t(1) + 0.1 * tf * (1 + c));
            case 'sin'
                U(:, ch) = amp * sin(2 * pi * (2.0 / tf) * (1 + 0.5 * c) * (t - t(1)));
            case 'prbs'
                hold_t = max(dt, tf / 100);
                rng(seed + c, 'twister');
                nblk = ceil(tf / hold_t) + 1;
                bits = 2 * (rand(nblk, 1) > 0.5) - 1;
                idx = min(floor((t - t(1)) / hold_t), nblk - 1) + 1;
                U(:, ch) = amp * bits(idx);
            otherwise
                error('c2py_export:signal', 'Unknown test input "%s".', kind);
        end
    end
end


function [t_out, y_out, err_msg] = local_sim(model, t, U, solver, dt)
% one simulation with the given inputs (same as tools/extract_plant_data.m)
    simIn = Simulink.SimulationInput(model);
    simIn = simIn.setModelParameter('StartTime', num2str(t(1), 17));
    simIn = simIn.setModelParameter('StopTime', num2str(t(end), 17));
    simIn = simIn.setModelParameter('SolverType', 'Fixed-step');
    simIn = simIn.setModelParameter('Solver', solver);
    simIn = simIn.setModelParameter('FixedStep', num2str(dt, 17));
    simIn = simIn.setModelParameter('SaveTime', 'on');
    simIn = simIn.setModelParameter('SaveOutput', 'on');
    simIn = simIn.setModelParameter('OutputSaveName', 'yout');
    simIn = simIn.setModelParameter('SaveFormat', 'Dataset');
    simIn = simIn.setModelParameter('LimitDataPoints', 'off');
    simIn = simIn.setModelParameter('CaptureErrors', 'on');
    if size(U, 2) > 0
        simIn = simIn.setExternalInput([t, U]);
    end
    simOut = sim(simIn);
    err_msg = '';
    try
        err_msg = strtrim(char(simOut.ErrorMessage));
    catch
    end
    if ~isempty(err_msg) && (~ismember('tout', simOut.who) || numel(simOut.get('tout')) < 2 || ~ismember('yout', simOut.who))
        error('c2py_export:sim', 'The simulation failed: %s', err_msg);
    end
    t_out = simOut.get('tout');
    t_out = t_out(:);
    if ~ismember('yout', simOut.who)
        error('c2py_export:noout', 'The model has no root Outport (nothing to compare).');
    end
    y_raw = simOut.get('yout');
    y_out = [];
    if isa(y_raw, 'Simulink.SimulationData.Dataset')
        for i = 1:y_raw.numElements
            v = y_raw.getElement(i).Values;
            if isa(v, 'timeseries'), v = v.Data; end
            y_out = [y_out, local_cols(v, numel(t_out))]; %#ok<AGROW>
        end
    elseif isstruct(y_raw)
        for i = 1:numel(y_raw.signals)
            y_out = [y_out, local_cols(y_raw.signals(i).values, numel(t_out))]; %#ok<AGROW>
        end
    else
        y_out = local_cols(y_raw, numel(t_out));
    end
end


function v = local_cols(v, n)
    v = double(v);
    if ndims(v) == 3, v = squeeze(v); end
    if size(v, 1) ~= n && size(v, 2) == n
        v = v';
    elseif size(v, 1) ~= n
        v = reshape(v, n, []);
    end
end


function local_csv(file, data, header)
    fid = fopen(file, 'w');
    fprintf(fid, '%s\n', strjoin(header, ','));
    fmt = [strjoin(repmat({'%.17g'}, 1, size(data, 2)), ','), '\n'];
    fprintf(fid, fmt, data.');
    fclose(fid);
end


function local_close(name)
    if bdIsLoaded(name), close_system(name, 0); end      % 0 = without saving: the file stays as it was
end


function local_restore(cfg, build_dir)
    try
        Simulink.fileGenControl('setConfig', 'config', cfg, 'createDir', true);
    catch
    end
    try
        rmdir(build_dir, 's');
    catch
    end
end
