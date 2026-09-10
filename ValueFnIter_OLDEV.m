function [V,Policy,EVtime]=ValueFnIter_OLDEV(n_a,n_z,N_j,a_grid,z_gridvals_J,pi_z_J,ReturnFn,Parameters,DiscountFactorParamNames,vfoptions,timeEV)
% Hardcode of exactly the code path that ValueFnIter_Case1_FHorz takes when
% solving Life-Cycle Model 10 (EVruntime.m): n_d=0, one endogenous state, one
% markov z, no e, lowmemory=0, divideandconquer=1, gridinterplayer=1, no
% V_Jplus1. That path is the ValueFnIter_FHorz_DC1_GI1_nod_raw solver plus the
% DC_GI dispatcher's default level1n and its post-processing (UnKron + reshape).
% Branches of the toolkit code that Model 10 never reaches are omitted.
%
% EV computation: the CURRENT toolkit broadcast,
%   EV=EV.*shiftdim(pi_z_J(:,:,jj)',-1); EV(isnan(EV))=0; EV=sum(EV,2);
%
% timeEV=1: accumulate the runtime of just the EV-contraction block into EVtime,
% with wait(gpuDevice) syncs around it (the syncs perturb the total slightly, so
% totals should be measured with timeEV=0). timeEV=0: EVtime returned as NaN.

N_a=prod(n_a);
N_z=prod(n_z);

gdev=gpuDevice;
EVtime=0;
if timeEV==0
    EVtime=NaN;
end

% DC_GI dispatcher default (vfoptions.level1n not set by Model 10)
level1n=floor(sqrt(n_a));

% Hardcode of ReturnFnParamNamesFn for Model 10's ReturnFn (inputs after aprime,a,z)
ReturnFnParamNames={'w','sigma','agej','Jr','pension','r','kappa_j','wg1','wg2','wg3','beta','sj'};

V=zeros(N_a,N_z,N_j,'gpuArray');
Policy=zeros(2,N_a,N_z,N_j,'gpuArray'); % first dim indexes the optimal choice for aprime and aprime2 (in GI layer)
PolicyL2flag=2*ones(1,N_a,N_z,N_j,'gpuArray'); % 1=all weight to lower coarse pt, 2=usual linear weights, 3=all weight to upper coarse pt

% Preallocate (lowmemory==0)
midpoints_jj=zeros(1,N_a,N_z,'gpuArray');

zind=shiftdim(gpuArray(0:1:N_z-1),-1);

% n-Monotonicity
level1ii=round(linspace(1,n_a,level1n));

% Grid interpolation
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n2long=vfoptions.ngridinterp*2+3; % total number of aprime points we end up looking at in second layer
aprime_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*n2short));
n2aprime=length(aprime_grid);

%% j=N_j (no V_Jplus1, lowmemory==0)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);

% n-Monotonicity
ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod(ReturnFn, n_z, a_grid, a_grid(level1ii), z_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);

%Calc the max and it's index
[~,maxindex1]=max(ReturnMatrix_ii,[],1);

% Just keep the 'midpoint' version of maxindex1 [as GI]
midpoints_jj(1,level1ii,:)=maxindex1;

% Attempt for improved version
maxgap=max(maxindex1(1,2:end,:)-maxindex1(1,1:end-1,:),[],3);
for ii=1:(level1n-1)
    curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
    if maxgap(ii)>0
        loweredge=min(maxindex1(1,ii,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
        % loweredge is 1-by-1-by-n_z
        aprimeindexes=loweredge+(0:1:maxgap(ii))';
        % aprime possibilities are maxgap(ii)+1-by-1-by-n_z
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod(ReturnFn, n_z, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
        [~,maxindex]=max(ReturnMatrix_ii,[],1);
        midpoints_jj(1,curraindex,:)=maxindex+(loweredge-1);
    else
        loweredge=maxindex1(1,ii,:);
        midpoints_jj(1,curraindex,:)=repelem(loweredge,1,length(curraindex),1);
    end
end

% Turn this into the 'midpoint'
midpoints_jj=max(min(midpoints_jj,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
% midpoint is 1-by-n_a-by-n_z
aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short)'; % aprime points either side of midpoint
% aprime possibilities are n_d-by-n2long-by-n_a-by-n_z
ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod(ReturnFn,n_z,aprime_grid(aprimeindexes),a_grid,z_gridvals_J(:,:,N_j),ReturnFnParamsVec,2);
[Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
V(:,:,N_j)=shiftdim(Vtempii,1);
Policy(1,:,:,N_j)=shiftdim(squeeze(midpoints_jj),-1); % midpoint
Policy(2,:,:,N_j)=shiftdim(maxindexL2,-1); % aprimeL2ind
% L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
isInfLower    = (ReturnMatrix_ii(1,     :,:) == -Inf);
isInfUpper    = (ReturnMatrix_ii(n2long,:,:) == -Inf);
inLowerStrict = (maxindexL2 >= 2)         & (maxindexL2 <= n2short+1);
inUpperStrict = (maxindexL2 >= n2short+3) & (maxindexL2 <= n2long-1);
PolicyL2flag(1,:,:,N_j) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

%% Iterate backwards through j. (lowmemory==0)
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    % Create a vector containing all the return function parameters (in order)
    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    % --- EV contraction block (the OLD broadcast; this is what NEWEV changes) ---
    if timeEV==1; wait(gdev); tEV=tic; end
    EV=V(:,:,jj+1);

    EV=EV.*shiftdim(pi_z_J(:,:,jj)',-1);
    EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
    EV=sum(EV,2); % sum over z', leaving a singular second dimension
    if timeEV==1; wait(gdev); EVtime=EVtime+toc(tEV); end
    % --- end EV contraction block ---

    % Interpolate EV over aprime_grid
    EVinterp=interp1(a_grid,EV,aprime_grid);

    % n-Monotonicity
    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod(ReturnFn, n_z, a_grid, a_grid(level1ii), z_gridvals_J(:,:,jj), ReturnFnParamsVec,1);

    entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*EV;

    %Calc the max and it's index
    [~,maxindex1]=max(entireRHS_ii,[],1);

    % Just keep the 'midpoint' version of maxindex1 [as GI]
    midpoints_jj(1,level1ii,:)=maxindex1;

    % Attempt for improved version
    maxgap=max(maxindex1(1,2:end,:)-maxindex1(1,1:end-1,:),[],3);
    for ii=1:(level1n-1)
        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
        if maxgap(ii)>0
            loweredge=min(maxindex1(1,ii,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
            % loweredge is 1-by-1-by-n_z
            aprimeindexes=loweredge+(0:1:maxgap(ii))';
            % aprime possibilities are maxgap(ii)+1-by-1-by-n_z
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod(ReturnFn, n_z, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
            aprimez=aprimeindexes+N_a*zind;
            entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*EV(reshape(aprimez,[(maxgap(ii)+1),1,N_z])); % autoexpand the level1iidiff(ii) in 2nd-dim
            [~,maxindex]=max(entireRHS_ii,[],1);
            midpoints_jj(1,curraindex,:)=maxindex+(loweredge-1);
        else
            loweredge=maxindex1(1,ii,:);
            midpoints_jj(1,curraindex,:)=repelem(loweredge,1,length(curraindex),1);
        end
    end

    % Turn this into the 'midpoint'
    midpoints_jj=max(min(midpoints_jj,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
    % midpoint is 1-by-n_a-by-n_z
    aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short)'; % aprime points either side of midpoint
    % aprime possibilities are n_d-by-n2long-by-n_a-by-n_z
    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod(ReturnFn,n_z,aprime_grid(aprimeindexes),a_grid,z_gridvals_J(:,:,jj),ReturnFnParamsVec,2);
    aprimez=aprimeindexes+n2aprime*zind;
    entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*reshape(EVinterp(aprimez(:)),[n2long,N_a,N_z]);
    [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
    V(:,:,jj)=shiftdim(Vtempii,1);
    Policy(1,:,:,jj)=shiftdim(squeeze(midpoints_jj),-1); % midpoint
    Policy(2,:,:,jj)=shiftdim(maxindexL2,-1); % aprimeL2ind
    % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
    isInfLower    = (ReturnMatrix_ii(1,     :,:) == -Inf);
    isInfUpper    = (ReturnMatrix_ii(n2long,:,:) == -Inf);
    inLowerStrict = (maxindexL2 >= 2)         & (maxindexL2 <= n2short+1);
    inUpperStrict = (maxindexL2 >= n2short+3) & (maxindexL2 <= n2long-1);
    PolicyL2flag(1,:,:,jj) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

end

% Currently Policy(1,:) is the midpoint, and Policy(2,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(1,:) to 'lower grid point' and then have Policy(2,:)
% counting 0:nshort+1 up from this.
adjust=(Policy(2,:,:,:)<1+n2short+1); % if second layer is choosing below midpoint
Policy(1,:,:,:)=Policy(1,:,:,:)-adjust; % lower grid point
Policy(2,:,:,:)=adjust.*Policy(2,:,:,:)+(1-adjust).*(Policy(2,:,:,:)-n2short-1); % from 1 (lower grid point) to 1+n2short+1 (upper grid point)

Policy=[Policy; PolicyL2flag];

%% DC_GI dispatcher post-processing (UnKron + reshape)
Policy=UnKronPolicyIndexes1_FHorz_z(Policy,n_a,n_a,n_z,N_j,vfoptions);
V=reshape(V,[n_a,n_z,N_j]);

end
