% EVpolicydiag.m
% Diagnoses the two Policy mismatches EVruntime.m reported (n_a=201,n_z=21 and
% n_a=501,n_z=51): NEWEV matched OLDEV on V to ~8e-16 but Policy differed.
%
% Hypothesis to test: the mismatches are argmax TIE-BREAK FLIPS. Changing the
% summation order (BLAS dot product instead of sum over dim 2) perturbs EV at
% ULP level; where two aprime candidates are numerically tied at the optimum
% (very common with the grid interpolation layer, whose fine points sit O(h^2)
% apart in objective value near the peak), a 1-ULP perturbation flips which one
% max() returns. If so, the flips should be (i) rare, (ii) between ADJACENT
% fine-grid points, and (iii) invisible in V.
% The alternative — a real defect in the NEWEV pipeline — would show as flips
% that are many fine-grid points apart and/or a visible V difference.
%
% Also records whether V ever contains -Inf in this model. If it does not, then
% EVruntime/this model never exercise the clamp-and-restore path at all, and the
% only thing being compared is summation order (EVtest.m is what covers the
% -Inf semantics).
%
% Requires GPU. Run from the Guvenen2007 folder.

clear;
if exist('EVpolicydiag_diary.txt','file'); delete('EVpolicydiag_diary.txt'); end
diary('EVpolicydiag_diary.txt');
fprintf('=== EVpolicydiag.m run %s ===\n',char(datetime('now')));
g=gpuDevice;
fprintf('GPU: %s\n',g.Name);

navec=[101,201,501];
nzvec=[9,21,51];

%% Life-Cycle Model 10 parameters (as in EVruntime.m)
Params.agejshifter=19;
Params.J=100-Params.agejshifter;
n_d=0;
N_j=Params.J;
Params.beta = 0.96;
Params.sigma = 2;
Params.w=1;
Params.r=0.05;
Params.agej=1:1:Params.J;
Params.Jr=46;
Params.pension=0.3;
Params.kappa_j=[linspace(0.5,2,Params.Jr-15),linspace(2,1,14),zeros(1,Params.J-Params.Jr+1)];
Params.rho_z=0.9;
Params.sigma_epsilon_z=0.03;
Params.dj=[0.006879, 0.000463, 0.000307, 0.000220, 0.000184, 0.000172, 0.000160, 0.000149, 0.000133, 0.000114, 0.000100, 0.000105, 0.000143, 0.000221, 0.000329, 0.000449, 0.000563, 0.000667, 0.000753, 0.000823,...
    0.000894, 0.000962, 0.001005, 0.001016, 0.001003, 0.000983, 0.000967, 0.000960, 0.000970, 0.000994, 0.001027, 0.001065, 0.001115, 0.001154, 0.001209, 0.001271, 0.001351, 0.001460, 0.001603, 0.001769, 0.001943, 0.002120, 0.002311, 0.002520, 0.002747, 0.002989, 0.003242, 0.003512, 0.003803, 0.004118, 0.004464, 0.004837, 0.005217, 0.005591, 0.005963, 0.006346, 0.006768, 0.007261, 0.007866, 0.008596, 0.009473, 0.010450, 0.011456, 0.012407, 0.013320, 0.014299, 0.015323,...
    0.016558, 0.018029, 0.019723, 0.021607, 0.023723, 0.026143, 0.028892, 0.031988, 0.035476, 0.039238, 0.043382, 0.047941, 0.052953, 0.058457, 0.064494,...
    0.071107, 0.078342, 0.086244, 0.094861, 0.104242, 0.114432, 0.125479, 0.137427, 0.150317, 0.164187, 0.179066, 0.194979, 0.211941, 0.229957, 0.249020, 0.269112, 0.290198, 0.312231, 1.000000];
Params.sj=1-Params.dj(21:101);
Params.sj(end)=0;
Params.wg1=0.3;
Params.wg2=3;
Params.wg3=Params.sigma;
d_grid=[];
DiscountFactorParamNames={'beta','sj'};
ReturnFn=@(aprime,a,z,w,sigma,agej,Jr,pension,r,kappa_j,wg1,wg2,wg3,beta,sj) ...
    LifeCycleModel10_ReturnFn(aprime,a,z,w,sigma,agej,Jr,pension,r,kappa_j,wg1,wg2,wg3,beta,sj);
vfoptions.divideandconquer=1;
vfoptions.gridinterplayer=1;
vfoptions.ngridinterp=20;
vfoptions.lowmemory=0;
vfoptions.verbose=0;
n2short=vfoptions.ngridinterp;

fprintf('\n%5s %5s | %10s %8s | %10s %10s %10s | %11s | %9s\n', ...
    'n_a','n_z','Pol diffs','frac','|dfine|=1','|dfine|=2','|dfine|>2','max relV@dif','-Inf in V');
for n_a=navec
    for n_z=nzvec
        a_grid=gpuArray(10*(linspace(0,1,n_a).^3)');
        [z_grid,pi_z]=discretizeAR1_FarmerToda(0,Params.rho_z,Params.sigma_epsilon_z,n_z);
        z_grid=exp(z_grid);
        [mean_z,~,~,~]=MarkovChainMoments(z_grid,pi_z);
        z_grid=gpuArray(z_grid./mean_z);
        pi_z=gpuArray(pi_z);
        z_gridvals_J=z_grid.*ones(1,1,N_j,'gpuArray');
        pi_z_J=pi_z.*ones(1,1,N_j-1,'gpuArray');

        [V_old,Pol_old,~]=ValueFnIter_OLDEV(n_a,n_z,N_j,a_grid,z_gridvals_J,pi_z_J,ReturnFn,Params,DiscountFactorParamNames,vfoptions,0);
        [V_new,Pol_new,~]=ValueFnIter_NEWEV(n_a,n_z,N_j,a_grid,z_gridvals_J,pi_z_J,ReturnFn,Params,DiscountFactorParamNames,vfoptions,0);

        V_old=gather(V_old); V_new=gather(V_new);
        Pol_old=gather(Pol_old); Pol_new=gather(Pol_new);

        % Policy rows: 1=lower coarse grid point, 2=L2 index (1..n2short+2 up from it), 3=L2 flag
        dany=squeeze(any(Pol_old~=Pol_new,1)); % [n_a,n_z,N_j]
        ndif=nnz(dany); ntot=numel(dany);

        % Position on the fine aprime grid implied by rows 1 and 2
        fi_old=squeeze((Pol_old(1,:,:,:)-1)*(n2short+1)+Pol_old(2,:,:,:));
        fi_new=squeeze((Pol_new(1,:,:,:)-1)*(n2short+1)+Pol_new(2,:,:,:));
        dfi=abs(fi_old(dany)-fi_new(dany));

        relV=abs(V_old(dany)-V_new(dany))./max(1,abs(V_old(dany)));
        ninf=nnz(~isfinite(V_old));

        if ndif>0
            fprintf('%5d %5d | %10d %7.1e | %10d %10d %10d | %11.3e | %9d\n', ...
                n_a,n_z,ndif,ndif/ntot,nnz(dfi==1),nnz(dfi==2),nnz(dfi>2),max(relV),ninf);
            % Detail: which ages, and the worst case
            jlist=unique(ceil(find(dany)/(n_a*n_z)));
            fprintf('        ages with flips: %s\n',mat2str(jlist(:)'));
            [~,iw]=max(dfi);
            idx=find(dany); iw=idx(iw);
            [ia,iz,ij]=ind2sub([n_a,n_z,N_j],iw);
            fprintf('        largest fine-grid gap: %d step(s) at (a=%d,z=%d,j=%d); V_old=%.15g V_new=%.15g\n', ...
                max(dfi),ia,iz,ij,V_old(iw),V_new(iw));
        else
            fprintf('%5d %5d | %10d %7.1e | %10s %10s %10s | %11s | %9d\n', ...
                n_a,n_z,0,0,'-','-','-','-',ninf);
        end
        clear V_old V_new Pol_old Pol_new a_grid z_grid pi_z z_gridvals_J pi_z_J
    end
end

fprintf('\nReading: flips that are 1 fine-grid step apart with relV at machine epsilon are\n');
fprintf('numerically tied optima (summation order decides the tie), not a defect. Flips many\n');
fprintf('steps apart, or a visible relV, would indicate a real problem in the NEWEV pipeline.\n');
fprintf('If "-Inf in V" is 0 everywhere, this model never exercises the clamp/restore path.\n');
diary off;
