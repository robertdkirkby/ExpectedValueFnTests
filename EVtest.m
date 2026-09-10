% EVtest.m
% Tests the toolkit's CURRENT EV computation (broadcast) against the PROPOSED
% matrix-multiplication path that was added under a size guard to
% ValueFnIter_FHorz_DC1_nod_raw (both EV sites), motivated by the Guvenen (2007)
% belief chain where N_z=3965 makes the broadcast a 63GB array.
%
% Current (toolkit):   EV=V.*shiftdim(pi_z',-1); EV(isnan(EV))=0; EV=sum(EV,2);
%                      transient: N_a*N_z^2 doubles + N_a*N_z^2 logicals (~9 bytes/elem)
% Proposed (guarded):  Vc=V; Vc(Vc==-Inf)=-1e250; EV=Vc*pi_z';
%                      EV((V==-Inf)*(pi_z'>0)>0)=-Inf;  % exact -Inf restoration
%                      EV=reshape(EV,[N_a,1,N_z]);
%                      transient: a few N_a*N_z doubles + one N_z*N_z indicator
% The clamp stops -Inf*0 producing NaN inside the product; the indicator matmul then
% restores EXACT -Inf wherever some positive-probability transition leads to a -Inf
% continuation (counts of 0/1 doubles are exact integers, so this is not a threshold
% on magnitudes). Result: NO behavioural difference vs the broadcast -- identical
% -Inf pattern, identical argmax behaviour even at fully-infeasible states; finite
% entries differ only by floating-point summation order.
%
% Test 1: same answer on all-finite rand() inputs of the relevant sizes. Sizes where
%         the broadcast cannot fit are still ATTEMPTED, inside try-catch, to
%         demonstrate that it OOMs while the matmul at the same size succeeds.
% Test 2: same answer with -Inf entries in V and structural zeros in pi_z
%         (the -Inf * 0 -> 0 semantics that the isnan-fix implements): wherever the
%         broadcast gives a finite value the matmul must match; wherever the
%         broadcast gives -Inf the matmul must be EXACTLY -Inf (strict equality of the
%         infeasibility pattern, no behavioural difference).
% Test 3: runtimes and memory at the relevant sizes, including the Guvenen size
%         N_z=3965; the broadcast is again attempted inside try-catch at every size,
%         so the OOM sizes are demonstrated rather than assumed.
% Test 4: crossover sweep. Tests 1-3 run only at sizes where the matmul wins by
%         9-28x, but EVruntime.m (Life-Cycle Model 10) found the matmul 1.2-1.7x
%         SLOWER at n_z<=51, where both paths are kernel-launch-bound and the matmul
%         runs more kernels. Both are points on one curve; this sweep locates where
%         it crosses. The crossover is expected at a roughly constant transient SIZE
%         (N_a*N_z^2 elements) rather than at a particular N_z -- that size is the
%         quantity a toolkit size guard would test -- so the sweep runs at three N_a
%         and reports whether the three crossover element-counts agree.
%
% Note on N_a: Tests 1-3 use N_a=501 throughout (the Guvenen baseline asset grid
% size); Test 4 sweeps its own N_a and does not use that value.
%
% Requires GPU. Run from the Guvenen2007 folder.

clear;
if exist('EVtest_diary.txt','file'); delete('EVtest_diary.txt'); end
diary('EVtest_diary.txt');
fprintf('=== EVtest.m run %s ===\n',char(datetime('now')));
g=gpuDevice;
fprintf('GPU: %s, total memory %.1f GB, available %.1f GB\n',g.Name,g.TotalMemory/1e9,g.AvailableMemory/1e9);
npass=0; nfail=0;
N_a=501;
nrep=5;

%% Test 1: all-finite correctness at the relevant sizes
fprintf('\n--- Test 1: correctness, all-finite rand() inputs ---\n');
for N_z=[525, 1000, 3965] % RIP size, mid size, Guvenen HIP size
    rng(1);
    V=rand(N_a,N_z,'gpuArray');
    pi_z=rand(N_z,N_z,'gpuArray');
    pi_z=pi_z./sum(pi_z,2); % row-stochastic
    % proposed matmul path (clamp and -Inf restoration are no-ops here, included for faithfulness)
    Vc=V; Vc(Vc==-Inf)=-1e250;
    EV2=Vc*pi_z';
    EV2((V==-Inf)*(pi_z'>0)>0)=-Inf;
    EV2=reshape(EV2,[N_a,1,N_z]);
    matmulok=gather(~any(isnan(EV2),'all')) && isequal(size(EV2),[N_a,1,N_z]);
    % current broadcast path: actually attempt it, so sizes that OOM demonstrate the OOM
    projGB=(N_a*N_z^2*9)/1e9;
    try
        EV1=V.*shiftdim(pi_z',-1);
        EV1(isnan(EV1))=0;
        EV1=sum(EV1,2);
        maxreldiff=gather(max(abs(EV1-EV2)./max(1,abs(EV1)),[],'all'));
        fprintf('N_z=%4d: max rel diff broadcast vs matmul = %.3e (tol 1e-12)\n',N_z,maxreldiff);
        if maxreldiff<1e-12; npass=npass+1; else; nfail=nfail+1; fprintf('FAIL: EV mismatch at N_z=%d\n',N_z); end
        clear EV1
    catch ME
        fprintf('N_z=%4d: broadcast ERRORED as projected (%.1f GB transient): %s\n',N_z,projGB,ME.identifier);
        fprintf('         matmul at the same size succeeded (valid [%d,1,%d] output, no NaN): %d\n',N_a,N_z,matmulok);
        if matmulok; npass=npass+1; else; nfail=nfail+1; fprintf('FAIL: matmul did not produce valid output at N_z=%d\n',N_z); end
    end
    clear V pi_z Vc EV2
end

%% Test 2: -Inf and zero-probability semantics
fprintf('\n--- Test 2: correctness with -Inf in V and structural zeros in pi_z ---\n');
for N_z=[525, 1000]
    rng(2);
    V=rand(N_a,N_z,'gpuArray');
    % -Inf pattern like the real model: fully-infeasible low-asset rows, plus scattered
    % (a,z') infeasibilities at a rate tuned so that (with ~0.3*N_z nonzero transitions
    % per row) about half of the remaining EV entries stay finite. The first run of this
    % test used a flat 15% rate, under which the finite set is empty w.p. ~1-1e-11 per
    % entry and the comparison was vacuous.
    V(1:20,:)=-Inf;
    pscat=1-0.5^(1/(0.3*N_z));
    V(rand(N_a,N_z,'gpuArray')<pscat)=-Inf;
    pi_z=rand(N_z,N_z,'gpuArray').*(rand(N_z,N_z,'gpuArray')<0.3); % ~70% structural zeros
    pi_z(:,1)=0.1; % keep every row with at least one nonzero
    pi_z=pi_z./sum(pi_z,2);
    % count the (-Inf x zero-probability) co-occurrences: the case the isnan-fix handles
    co=gather(sum(sum(V==-Inf,1).*sum(pi_z==0,1)));
    % current broadcast path (the reference semantics: -Inf*0 -> NaN -> 0)
    EV1=V.*shiftdim(pi_z',-1);
    EV1(isnan(EV1))=0;
    EV1=sum(EV1,2);
    % proposed matmul path with exact -Inf restoration
    Vc=V; Vc(Vc==-Inf)=-1e250;
    EV2=Vc*pi_z';
    EV2((V==-Inf)*(pi_z'>0)>0)=-Inf;
    EV2=reshape(EV2,[N_a,1,N_z]);
    fin=isfinite(EV1);
    n_inf=gather(nnz(~fin)); n_fin=gather(nnz(fin));
    if n_fin>0
        maxreldiff=gather(max(abs(EV1(fin)-EV2(fin))./max(1,abs(EV1(fin)))));
    else
        maxreldiff=NaN; % vacuous -- caught by the n_fin>0 requirement below
    end
    saminfpattern=gather(isequal(~isfinite(EV1),~isfinite(EV2))); % identical infeasibility pattern
    okinf=gather(all(EV2(~fin)==-Inf)); % and exactly -Inf, not just huge-negative
    anynan=gather(any(isnan(EV1),'all'))||gather(any(isnan(EV2),'all'));
    fprintf('N_z=%4d: finite EV entries %d, -Inf EV entries %d, (-Inf x zero-prob) cases %d\n',N_z,n_fin,n_inf,co);
    fprintf('         finite: max rel diff = %.3e (tol 1e-12); -Inf pattern identical = %d; -Inf exactly -Inf = %d; any NaN = %d\n',maxreldiff,saminfpattern,okinf,anynan);
    if n_fin>0 && n_inf>0 && co>0 && maxreldiff<1e-12 && saminfpattern && okinf && ~anynan; npass=npass+1;
    else; nfail=nfail+1; fprintf('FAIL: -Inf/zero semantics at N_z=%d\n',N_z); end
    clear V pi_z Vc EV1 EV2 fin
end

%% Test 3: runtimes and memory
fprintf('\n--- Test 3: runtimes (mean over %d reps, after warmup) and memory ---\n',nrep);
fprintf('%8s %14s %14s %16s %16s\n','N_z','broadcast (s)','matmul (s)','broadcast mem','matmul mem');
for N_z=[525, 1000, 1500, 2000, 3965]
    rng(3);
    V=rand(N_a,N_z,'gpuArray');
    V(rand(N_a,N_z,'gpuArray')<0.15)=-Inf;
    pi_z=rand(N_z,N_z,'gpuArray');
    pi_z=pi_z./sum(pi_z,2);
    projGB=(N_a*N_z^2*9)/1e9;      % broadcast transient: double result + logical mask
    matmulGB=(3*N_a*N_z*8+N_z*N_z*8)/1e9; % matmul transient: clamped copy + output + indicators
    % matmul timing (full cleaned pipeline incl. the exact -Inf restoration)
    Vc=V; Vc(Vc==-Inf)=-1e250; EV2=Vc*pi_z'; EV2((V==-Inf)*(pi_z'>0)>0)=-Inf; EV2=reshape(EV2,[N_a,1,N_z]); wait(g); %#ok<NASGU> % warmup
    tic;
    for rep=1:nrep
        Vc=V; Vc(Vc==-Inf)=-1e250;
        EV2=Vc*pi_z';
        EV2((V==-Inf)*(pi_z'>0)>0)=-Inf;
        EV2=reshape(EV2,[N_a,1,N_z]); %#ok<NASGU>
    end
    wait(g); t_mat=toc/nrep;
    clear Vc EV2
    % broadcast timing: actually attempt it, so sizes that OOM demonstrate the OOM
    try
        EV1=V.*shiftdim(pi_z',-1); EV1(isnan(EV1))=0; EV1=sum(EV1,2); wait(g); clear EV1 % warmup
        tic;
        for rep=1:nrep
            EV1=V.*shiftdim(pi_z',-1);
            EV1(isnan(EV1))=0;
            EV1=sum(EV1,2);
        end
        wait(g); t_bro=toc/nrep;
        clear EV1
        fprintf('%8d %14.4f %14.4f %13.2f GB %13.4f GB\n',N_z,t_bro,t_mat,projGB,matmulGB);
    catch ME
        fprintf('%8d %14s %14.4f %13.2f GB %13.4f GB   (broadcast OOM: %s)\n',N_z,'OOM',t_mat,projGB,matmulGB,ME.identifier);
    end
    clear V pi_z
end

%% Test 4: crossover sweep -- where does the matmul overtake the broadcast?
% The -Inf rate here differs from Test 3's flat 15%: with a dense pi_z, a single
% -Inf anywhere in V(a,:) makes the whole EV row -Inf, so at 15% every EV entry
% would be -Inf and the correctness spot-check below would be vacuous. Instead
% scatter at a rate that leaves about half the EV entries finite. The timing
% difference is confined to how many elements the two indexed assignments write.
fprintf('\n--- Test 4: crossover sweep (per-call EV block, mean over %d reps after warmup) ---\n',nrep);
navec4=[101,201,501];
nzvec4=[9,21,41,61,81,101,151,251,401,525,1000];
Xstar_all=nan(1,length(navec4));
maxreldiff4=0; ncomp4=0; nfinpts4=0; ninfpts4=0; patternok4=true;
for ia=1:length(navec4)
    N_a4=navec4(ia);
    fprintf('\nN_a=%d\n',N_a4);
    fprintf('%8s %15s %15s %10s %14s %12s\n','N_z','broadcast (s)','matmul (s)','bro/mat','elements','bro mem');
    t_bro4=nan(1,length(nzvec4)); t_mat4=nan(1,length(nzvec4));
    for iz=1:length(nzvec4)
        N_z=nzvec4(iz);
        rng(4);
        pscat=1-0.5^(1/N_z); % leaves ~half the EV entries finite (see note above)
        V=rand(N_a4,N_z,'gpuArray');
        V(rand(N_a4,N_z,'gpuArray')<pscat)=-Inf;
        pi_z=rand(N_z,N_z,'gpuArray');
        pi_z=pi_z./sum(pi_z,2);
        nelem=N_a4*N_z^2;
        projGB=(nelem*9)/1e9; % broadcast transient: double result + logical mask
        % matmul timing (full cleaned pipeline incl. the exact -Inf restoration)
        Vc=V; Vc(Vc==-Inf)=-1e250; EV2=Vc*pi_z'; EV2((V==-Inf)*(pi_z'>0)>0)=-Inf; EV2=reshape(EV2,[N_a4,1,N_z]); wait(g); % warmup
        tic;
        for rep=1:nrep
            Vc=V; Vc(Vc==-Inf)=-1e250;
            EV2=Vc*pi_z';
            EV2((V==-Inf)*(pi_z'>0)>0)=-Inf;
            EV2=reshape(EV2,[N_a4,1,N_z]);
        end
        wait(g); t_mat4(iz)=toc/nrep;
        clear Vc
        % broadcast timing: attempted, so any OOM at the top sizes is demonstrated
        try
            EV1=V.*shiftdim(pi_z',-1); EV1(isnan(EV1))=0; EV1=sum(EV1,2); wait(g); % warmup
            tic;
            for rep=1:nrep
                EV1=V.*shiftdim(pi_z',-1);
                EV1(isnan(EV1))=0;
                EV1=sum(EV1,2);
            end
            wait(g); t_bro4(iz)=toc/nrep;
            % spot-check that the pipeline being timed is still the correct one
            fin=isfinite(EV1);
            nf=gather(nnz(fin)); ni=gather(nnz(~fin));
            if nf>0
                maxreldiff4=max(maxreldiff4,gather(max(abs(EV1(fin)-EV2(fin))./max(1,abs(EV1(fin))))));
                nfinpts4=nfinpts4+1;
            end
            if ni>0; ninfpts4=ninfpts4+1; end
            if ~isequal(gather(~isfinite(EV1)),gather(~isfinite(EV2))); patternok4=false; end
            ncomp4=ncomp4+1;
            clear EV1 fin
            fprintf('%8d %15.5f %15.5f %9.2fx %14d %9.2f GB\n',N_z,t_bro4(iz),t_mat4(iz),t_bro4(iz)/t_mat4(iz),nelem,projGB);
        catch ME
            fprintf('%8d %15s %15.5f %10s %14d %9.2f GB   (broadcast OOM: %s)\n',N_z,'OOM',t_mat4(iz),'inf',nelem,projGB,ME.identifier);
        end
        clear V pi_z EV2
    end
    % Locate the crossover: the first swept size at which the broadcast stops winning
    r4=t_bro4./t_mat4;
    r4(isnan(t_bro4))=Inf; % an OOM counts as the matmul winning
    ic=find(r4>=1,1,'first');
    if isempty(ic)
        fprintf('N_a=%d: no crossover within the swept range (broadcast faster throughout)\n',N_a4);
    elseif ic==1
        fprintf('N_a=%d: matmul already faster at the smallest swept N_z=%d (crossover is below the range)\n',N_a4,nzvec4(1));
    else
        X1=N_a4*nzvec4(ic-1)^2; X2=N_a4*nzvec4(ic)^2;
        if isfinite(r4(ic))
            % log-log interpolation on the ratio to estimate where it equals 1
            Xstar_all(ia)=exp(log(X1)+(0-log(r4(ic-1)))*(log(X2)-log(X1))/(log(r4(ic))-log(r4(ic-1))));
        else
            Xstar_all(ia)=sqrt(X1*X2); % broadcast OOMed at the upper bracket; take the geometric midpoint
        end
        fprintf('N_a=%d: crossover between N_z=%d and N_z=%d; estimated at %.3g elements (N_z~%.0f at this N_a)\n', ...
            N_a4,nzvec4(ic-1),nzvec4(ic),Xstar_all(ia),sqrt(Xstar_all(ia)/N_a4));
    end
end

fprintf('\ncrossover summary (a toolkit size guard would test N_a*N_z^2 elements):\n');
for ia=1:length(navec4)
    if isfinite(Xstar_all(ia))
        fprintf('  N_a=%4d: ~%.3g elements (%.3f GB broadcast transient), i.e. N_z~%.0f at this N_a\n', ...
            navec4(ia),Xstar_all(ia),Xstar_all(ia)*9/1e9,sqrt(Xstar_all(ia)/navec4(ia)));
    else
        fprintf('  N_a=%4d: crossover not bracketed within the swept range\n',navec4(ia));
    end
end
if all(isfinite(Xstar_all))
    fprintf('  spread across N_a: %.2fx (close to 1 means the crossover is a transient-size threshold and transfers across grids)\n', ...
        max(Xstar_all)/min(Xstar_all));
end
if ncomp4>0 && nfinpts4>0 && ninfpts4>0 && patternok4 && maxreldiff4<1e-12
    npass=npass+1;
    fprintf('sweep spot-check: %d points compared (%d with finite EV entries, %d with -Inf EV entries), max rel diff %.3e, -Inf patterns identical\n', ...
        ncomp4,nfinpts4,ninfpts4,maxreldiff4);
else
    nfail=nfail+1;
    fprintf('FAIL: sweep spot-check (%d points, %d finite, %d -Inf, patterns ok=%d, max rel diff %.3e)\n', ...
        ncomp4,nfinpts4,ninfpts4,patternok4,maxreldiff4);
end

fprintf('\n=== EVtest summary: %d passed, %d failed ===\n',npass,nfail);
diary off;
