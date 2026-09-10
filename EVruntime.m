% EVruntime.m
% Runtime implications of the EV matmul (see EVtest.m for the EV-block-only
% comparison) for the value function iteration AS A WHOLE, across grid sizes:
% loops over n_a in {101,201,501} x n_z in {9,21,51} and reports a table.
%
% Model: a copy of Life-Cycle Model 10 (Intro to Life-Cycle Models; exogenous
% labor supply, one asset, AR(1) z, warm glow), everything after the
% ValueFnIter call removed, with n_a and n_z looped over. Model 10 uses
% divideandconquer=1 and gridinterplayer=1, so the toolkit path is
% ValueFnIter_FHorz_DC1_GI1_nod_raw.
%
% Solvers compared:
%   ValueFnIter_OLDEV — hardcode of exactly the toolkit code path this model
%                       takes, with the current broadcast EV
%   ValueFnIter_NEWEV — same file with ONLY the EV block changed to the matmul
%                       pipeline with exact -Inf restoration
% Both take a timeEV flag: totals are measured with timeEV=0 (no internal
% syncs); the EV-only share is measured in separate timeEV=1 runs (which add
% wait(gpuDevice) syncs around the EV block each age).
%
% Checks per (n_a,n_z): OLDEV must reproduce the toolkit's
% ValueFnIter_Case1_FHorz output exactly (it is the same ops); NEWEV must
% match Policy exactly and V to ULP (summation-order only).
%
% Requires GPU. Run from the Guvenen2007 folder.

clear;
if exist('EVruntime_diary.txt','file'); delete('EVruntime_diary.txt'); end
diary('EVruntime_diary.txt');
fprintf('=== EVruntime.m run %s ===\n',char(datetime('now')));
g=gpuDevice;
fprintf('GPU: %s, total memory %.1f GB, available %.1f GB\n',g.Name,g.TotalMemory/1e9,g.AvailableMemory/1e9);
npass=0; nfail=0;
nrep=5; % timing repetitions (after warmup)

navec=[101,201,501];
nzvec=[9,21,51];

%% Life-Cycle Model 10 parameters (verbatim copy; n_a and n_z are looped below)
Params.agejshifter=19; % Age 20 minus one. Makes keeping track of actual age easy in terms of model age
Params.J=100-Params.agejshifter; % =81, Number of period in life-cycle

n_d=0; % None
N_j=Params.J; % Number of periods in finite horizon

% Discount rate
Params.beta = 0.96;
% Preferences
Params.sigma = 2; % Coeff of relative risk aversion (curvature of consumption)

% Prices
Params.w=1; % Wage
Params.r=0.05; % Interest rate (0.05 is 5%)

% Demographics
Params.agej=1:1:Params.J; % Is a vector of all the agej: 1,2,3,...,J
Params.Jr=46;

% Pensions
Params.pension=0.3;

% Age-dependent labor productivity units
Params.kappa_j=[linspace(0.5,2,Params.Jr-15),linspace(2,1,14),zeros(1,Params.J-Params.Jr+1)];
% Exogenous shock process: AR1 on endowment income
Params.rho_z=0.9;
Params.sigma_epsilon_z=0.03;

% Conditional death probabilities
Params.dj=[0.006879, 0.000463, 0.000307, 0.000220, 0.000184, 0.000172, 0.000160, 0.000149, 0.000133, 0.000114, 0.000100, 0.000105, 0.000143, 0.000221, 0.000329, 0.000449, 0.000563, 0.000667, 0.000753, 0.000823,...
    0.000894, 0.000962, 0.001005, 0.001016, 0.001003, 0.000983, 0.000967, 0.000960, 0.000970, 0.000994, 0.001027, 0.001065, 0.001115, 0.001154, 0.001209, 0.001271, 0.001351, 0.001460, 0.001603, 0.001769, 0.001943, 0.002120, 0.002311, 0.002520, 0.002747, 0.002989, 0.003242, 0.003512, 0.003803, 0.004118, 0.004464, 0.004837, 0.005217, 0.005591, 0.005963, 0.006346, 0.006768, 0.007261, 0.007866, 0.008596, 0.009473, 0.010450, 0.011456, 0.012407, 0.013320, 0.014299, 0.015323,...
    0.016558, 0.018029, 0.019723, 0.021607, 0.023723, 0.026143, 0.028892, 0.031988, 0.035476, 0.039238, 0.043382, 0.047941, 0.052953, 0.058457, 0.064494,...
    0.071107, 0.078342, 0.086244, 0.094861, 0.104242, 0.114432, 0.125479, 0.137427, 0.150317, 0.164187, 0.179066, 0.194979, 0.211941, 0.229957, 0.249020, 0.269112, 0.290198, 0.312231, 1.000000];
% dj covers Ages 0 to 100
Params.sj=1-Params.dj(21:101); % Conditional survival probabilities
Params.sj(end)=0; % In the present model the last period (j=J) value of sj is actually irrelevant

% Warm glow of bequest
Params.wg1=0.3; % (relative) importance of bequests
Params.wg2=3; % degree to which bequests are a luxury good (>=1; =1 would be a normal good)
Params.wg3=Params.sigma; % By using the same curvature as the utility of consumption it makes it much easier to guess appropriate parameter values for the warm glow

d_grid=[]; % No decision variables

DiscountFactorParamNames={'beta','sj'};

ReturnFn=@(aprime,a,z,w,sigma,agej,Jr,pension,r,kappa_j,wg1,wg2,wg3,beta,sj) ...
    LifeCycleModel10_ReturnFn(aprime,a,z,w,sigma,agej,Jr,pension,r,kappa_j,wg1,wg2,wg3,beta,sj);

vfoptions.divideandconquer=1;
vfoptions.gridinterplayer=1;
vfoptions.ngridinterp=20; % 20 evenly-spaced points between each pair of consecutive a_grid points
vfoptions.lowmemory=0;
vfoptions.verbose=0;

%% Loop over grid sizes
ncomb=length(navec)*length(nzvec);
R=struct('n_a',cell(1,ncomb)); % results table rows
cc=0;
for n_a=navec
    for n_z=nzvec
        cc=cc+1;
        fprintf('\n--- n_a=%d, n_z=%d (EV broadcast transient %.2f MB) ---\n',n_a,n_z,n_a*n_z^2*9/1e6);

        % Grids (Model 10's constructions, at this size)
        a_grid=10*(linspace(0,1,n_a).^3)'; % The ^3 means most points are near zero, which is where the derivative of the value fn changes most.
        [z_grid,pi_z]=discretizeAR1_FarmerToda(0,Params.rho_z,Params.sigma_epsilon_z,n_z);
        z_grid=exp(z_grid); % Take exponential of the grid
        [mean_z,~,~,~]=MarkovChainMoments(z_grid,pi_z); % Calculate the mean of the grid so as can normalise it
        z_grid=z_grid./mean_z; % Normalise the grid on z (so that the mean of z is 1)

        % Hardcode of the dispatcher's grid setup for this input (age-independent z)
        a_grid=gpuArray(a_grid);
        z_grid=gpuArray(z_grid);
        pi_z=gpuArray(pi_z);
        z_gridvals_J=z_grid.*ones(1,1,N_j,'gpuArray');   % [N_z,1,N_j]
        pi_z_J=pi_z.*ones(1,1,N_j-1,'gpuArray');         % [N_z,N_z,N_j-1]

        % Correctness: OLDEV must reproduce the toolkit exactly; NEWEV to ULP
        % (these three solves also serve as the timing warmup)
        [V_tk,Policy_tk]=ValueFnIter_Case1_FHorz(n_d,n_a,n_z,N_j,d_grid,gather(a_grid),gather(z_grid),gather(pi_z),ReturnFn,Params,DiscountFactorParamNames,[],vfoptions);
        [V_old,Policy_old,~]=ValueFnIter_OLDEV(n_a,n_z,N_j,a_grid,z_gridvals_J,pi_z_J,ReturnFn,Params,DiscountFactorParamNames,vfoptions,0);
        [V_new,Policy_new,~]=ValueFnIter_NEWEV(n_a,n_z,N_j,a_grid,z_gridvals_J,pi_z_J,ReturnFn,Params,DiscountFactorParamNames,vfoptions,0);

        maxdiff_tk=gather(max(abs(V_old-V_tk),[],'all'));
        polsame_tk=gather(isequal(Policy_old,Policy_tk));
        fprintf('OLDEV vs toolkit: max |V diff| = %.3e (expect exactly 0), Policy identical = %d\n',maxdiff_tk,polsame_tk);
        if maxdiff_tk==0 && polsame_tk; npass=npass+1; else; nfail=nfail+1; fprintf('FAIL: OLDEV is not a faithful hardcode of the toolkit path at n_a=%d,n_z=%d\n',n_a,n_z); end

        maxreldiff_new=gather(max(abs(V_new-V_old)./max(1,abs(V_old)),[],'all'));
        polsame_new=gather(isequal(Policy_new,Policy_old));
        fprintf('NEWEV vs OLDEV: max rel |V diff| = %.3e (tol 1e-12, summation-order only), Policy identical = %d\n',maxreldiff_new,polsame_new);
        if maxreldiff_new<1e-12 && polsame_new; npass=npass+1; else; nfail=nfail+1; fprintf('FAIL: NEWEV does not match OLDEV at n_a=%d,n_z=%d\n',n_a,n_z); end
        clear V_tk Policy_tk V_old Policy_old V_new Policy_new

        % Timing: totals (timeEV=0, no internal syncs)
        t_old=zeros(1,nrep); t_new=zeros(1,nrep);
        for rep=1:nrep
            wait(g); tic;
            [V_old,Policy_old,~]=ValueFnIter_OLDEV(n_a,n_z,N_j,a_grid,z_gridvals_J,pi_z_J,ReturnFn,Params,DiscountFactorParamNames,vfoptions,0);
            wait(g); t_old(rep)=toc;
            clear V_old Policy_old
            wait(g); tic;
            [V_new,Policy_new,~]=ValueFnIter_NEWEV(n_a,n_z,N_j,a_grid,z_gridvals_J,pi_z_J,ReturnFn,Params,DiscountFactorParamNames,vfoptions,0);
            wait(g); t_new(rep)=toc;
            clear V_new Policy_new
        end

        % Timing: EV-only share (timeEV=1, syncs around the EV block each age)
        ev_old=zeros(1,nrep); ev_new=zeros(1,nrep);
        tI_old=zeros(1,nrep); tI_new=zeros(1,nrep);
        for rep=1:nrep
            wait(g); tic;
            [~,~,EVtime]=ValueFnIter_OLDEV(n_a,n_z,N_j,a_grid,z_gridvals_J,pi_z_J,ReturnFn,Params,DiscountFactorParamNames,vfoptions,1);
            wait(g); tI_old(rep)=toc; ev_old(rep)=EVtime;
            wait(g); tic;
            [~,~,EVtime]=ValueFnIter_NEWEV(n_a,n_z,N_j,a_grid,z_gridvals_J,pi_z_J,ReturnFn,Params,DiscountFactorParamNames,vfoptions,1);
            wait(g); tI_new(rep)=toc; ev_new(rep)=EVtime;
        end

        R(cc).n_a=n_a; R(cc).n_z=n_z;
        R(cc).tot_old=mean(t_old); R(cc).tot_new=mean(t_new);
        R(cc).ev_old=mean(ev_old); R(cc).ev_new=mean(ev_new);
        R(cc).tI_old=mean(tI_old); R(cc).tI_new=mean(tI_new);
        fprintf('total: OLD %.4fs, NEW %.4fs (%.2fx); EV block: OLD %.4fs, NEW %.4fs (%.2fx)\n', ...
            R(cc).tot_old,R(cc).tot_new,R(cc).tot_old/R(cc).tot_new,R(cc).ev_old,R(cc).ev_new,R(cc).ev_old/R(cc).ev_new);

        clear a_grid z_grid pi_z z_gridvals_J pi_z_J
    end
end

%% Report table
fprintf('\n--- Runtime comparison table (means over %d reps, %d backward steps per solve) ---\n',nrep,N_j-1);
fprintf('totals from clean runs (timeEV=0); EV-only from instrumented runs (timeEV=1, adds per-age syncs)\n');
fprintf('%5s %5s | %10s %10s %8s | %10s %10s %8s | %9s %9s\n', ...
    'n_a','n_z','totOLD(s)','totNEW(s)','OLD/NEW','EVold(s)','EVnew(s)','OLD/NEW','EVshOLD','EVshNEW');
for cc=1:ncomb
    fprintf('%5d %5d | %10.4f %10.4f %7.2fx | %10.4f %10.4f %7.2fx | %8.1f%% %8.1f%%\n', ...
        R(cc).n_a,R(cc).n_z,R(cc).tot_old,R(cc).tot_new,R(cc).tot_old/R(cc).tot_new, ...
        R(cc).ev_old,R(cc).ev_new,R(cc).ev_old/R(cc).ev_new, ...
        100*R(cc).ev_old/R(cc).tI_old,100*R(cc).ev_new/R(cc).tI_new);
end
fprintf('(EV share columns are the EV block as a %% of the instrumented run at the same setting)\n');

fprintf('\n=== EVruntime summary: %d passed, %d failed ===\n',npass,nfail);
diary off;
