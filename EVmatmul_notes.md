# Computing the expected value function with a matrix product

Notes accompanying the tests in
[ExpectedValueFnTests](https://github.com/robertdkirkby/ExpectedValueFnTests).

## The problem

VFI Toolkit computed the expectation over next period's exogenous state by
broadcasting the transition matrix against the value function:

```matlab
EV=EV.*shiftdim(pi_z_J(:,:,jj)',-1);
EV(isnan(EV))=0; % -Inf*0 gives NaN; the zeros come from the transition probabilities
EV=sum(EV,2);    % sum over z', leaving a singular second dimension
```

`EV` enters as `[N_a,N_z]`, so the middle line materialises an
`[N_a,N_z,N_z]` double plus an equally sized logical mask — about 9 bytes per
element. That is quadratic in `N_z`, and it is a *transient*: it exists only to
be summed away.

At `N_a=501` this is 1.2GB at `N_z=525`, 4.5GB at `N_z=1000`, and 70.9GB at
`N_z=3965`. On a 21GB GPU it runs out of memory somewhere around `N_z=1500-2000`
(the exact point depends on what else is resident) and exceeds MATLAB's
`gpuArray` maximum array size near `N_z=4000`.

This surfaced in a replication of Guvenen (2007), where agents learn about their
own income profile. The belief state is a three-dimensional chain
(`betahat`, `zhat`, `v`) of `61x13x5 = 3965` points, which made the model
unsolvable rather than merely slow.

## The replacement

```matlab
EVinf=(EV==-Inf);
EV(EVinf)=-1e250;                            % stop -Inf*0 -> NaN inside the product
EV=EV*pi_z_J(:,:,jj)';                       % sum over z'
EV(EVinf*(pi_z_J(:,:,jj)'>0)>0)=-Inf;        % exact -Inf restoration
EV=reshape(EV,[N_a,1,N_z]);
```

The transient is now a few `N_a`-by-`N_z` arrays plus one `N_z`-by-`N_z`
indicator — megabytes instead of gigabytes.

### Why the `isnan` fix cannot simply be kept

This is the crux, and it is easy to miss.

Under `.*` every product `V(a,z')*pi(z,z')` exists as its own array element, so
a `-Inf x 0` lands in its own slot and can be repaired individually:

```
broadcast terms: 0.5  NaN  1      <- the NaN sits in its OWN slot
after isnan fix: 0.5   0   1      sum = 1.5
```

Under `*` the multiply and the add are fused inside BLAS. The individual
products never exist as addressable values, so a single NaN propagates through
the summation and poisons the entire entry:

```
matmul, no clamp : NaN            <- one NaN poisons the whole dot product
```

By the time the result is visible there is nothing left to repair: you cannot
tell which term was bad, nor recover the sum of the remaining terms. So the
matmul path cannot *fix* the NaN afterwards — it has to *prevent* it. That is
what the clamp is for. It is the matmul-compatible substitute for the `isnan`
overwrite: same intent, opposite timing.

### Why this is exact, not approximate

Split the entries of the output by whether any infeasible continuation is
actually reachable.

- **All `-Inf` continuations have probability exactly 0.** The clamped value
  enters as `finite x 0 = 0`, contributing exactly zero — which is precisely
  what the `isnan` overwrite arranged. No clamp artifact can survive.
- **Some `-Inf` continuation has positive probability.** The clamped value
  produces a huge negative number, and the restoration line replaces the entry
  with exactly `-Inf`, matching the broadcast.

The restoration is an indicator matrix product: `(EV==-Inf)*(pi'>0)` counts, for
each `(a,z)`, how many positive-probability transitions lead to an infeasible
continuation. These are counts of 0/1 values in double precision, so they are
exact integers — `>0` is a genuine test, not a threshold on magnitudes.

Consequently the clamp *value* is irrelevant among finite numbers: `-1e250`,
`-7` and `+42` all give identical results. `-1e250` is chosen defensively, so
that any hypothetical leakage would be an absurd number rather than a plausible
one, while staying far from overflow.

The only remaining difference from the broadcast is floating-point summation
order — MATLAB's `sum` along a dimension versus BLAS dot products.

## The tests

| file | what it establishes |
|---|---|
| `EVtest.m` | correctness and cost of the EV block in isolation |
| `EVruntime.m` | effect on a whole value function iteration |
| `ValueFnIter_OLDEV.m` / `ValueFnIter_NEWEV.m` | hardcoded solvers differing *only* in the EV block |
| `EVpolicydiag.m` | whether the ULP differences move any policy choice |

`EVtest.m` has four parts. Tests 1 and 2 check correctness on all-finite inputs
and on inputs carrying `-Inf` with structural zeros in `pi`. Test 2 is the
important one: it demands an *identical* infeasibility pattern (`isequal`),
every non-finite entry exactly `-Inf`, finite entries agreeing to 1e-12, and no
NaN — and it asserts that the finite, `-Inf`, and `-Inf`-times-zero-probability
populations are all non-empty, so a pass cannot be vacuous. Test 3 measures
runtime and memory, attempting the broadcast inside `try/catch` so the
out-of-memory sizes are demonstrated rather than assumed. Test 4 sweeps
`N_z` at three `N_a` to locate where the matmul overtakes the broadcast.

`EVruntime.m` embeds Life-Cycle Model 10 from the Intro to Life-Cycle Models
(exogenous labour supply, one asset, AR(1) shock, warm glow) and solves it with
two hardcoded copies of the toolkit's code path that differ only in the EV
block, timing both the whole iteration and the EV block alone.

## Results

Measured on an NVIDIA RTX 4000 Ada (21GB).

**Correctness.** All checks pass. Maximum relative difference against the
broadcast is 3.3e-16, with identical `-Inf` patterns and no NaN on either path.
At `N_z=525` the non-vacuous populations were 121,601 finite entries, 141,424
`-Inf` entries, and 4.27 million `-Inf`-times-zero-probability co-occurrences.

**The EV block.**

| `N_z` | broadcast | matmul | speedup | broadcast memory | matmul memory |
|---|---|---|---|---|---|
| 525 | 0.0081s | 0.0017s | 4.8x | 1.24 GB | 0.009 GB |
| 1000 | 0.0872s | 0.0061s | 14.3x | 4.51 GB | 0.020 GB |
| 1500 | 0.3431s | 0.0123s | 27.9x | 10.15 GB | 0.036 GB |
| 2000 | out of memory | 0.0218s | — | 18.04 GB | 0.056 GB |
| 3965 | exceeds max array size | 0.0849s | — | 70.89 GB | 0.173 GB |

**Where the crossover lies.** Below a certain size the matmul is *slower*,
because both paths are dominated by kernel-launch overhead and the matmul runs
more kernels. The crossover is a roughly constant *transient size*, not a
particular `N_z`:

| `N_a` | crossover (elements of `N_a*N_z^2`) | equivalent `N_z` |
|---|---|---|
| 101 | 2.94e6 | ~170 |
| 201 | 2.12e6 | ~103 |
| 501 | 1.69e6 | ~58 |

The equivalent `N_z` moves by a factor of three across this range while the
element count moves by 1.74x, so anyone tempted to gate this on `N_z` alone
should gate on `N_a*N_z^2` instead.

**Cost below the crossover.** A fixed ~30 microseconds per EV call, which is
about 3ms per 80-period solve, or 0.3-1.3% of total value-function-iteration
runtime for Life-Cycle Model 10. It is a fixed cost rather than a proportional
one, so its share shrinks as models grow. Against that, the largest case above
saves 6.5 seconds per solve — and runs at all.

**Policy.** Value function differences at ULP can in principle flip a tied
argmax. Measured across nine grid sizes, this happened twice: one policy entry
in 341,901 and one in 2,069,631. Both moved to the *adjacent* fine-grid point,
with the value function identical to fifteen significant digits — numerically
tied optima where the summation order broke the tie, not a behavioural change.

## Limits

**Mixed infinities.** If a state has both `+Inf` and `-Inf` continuations at
positive probability, the broadcast's terms are `Inf` and `-Inf`, *neither of
which is NaN*, so its `isnan` fix never fires and the sum is NaN. The matmul
with a `-Inf`-only clamp forms `+Inf x 0 = NaN` inside the product and also
gives NaN. Both are wrong, differently. Value functions carrying `+Inf` therefore
need their own analysis before this substitution is applied to them — in VFI
Toolkit this is why the Epstein-Zin families, whose transformed continuation can
carry `+Inf`, were left on the broadcast.

**Genuine NaN.** The `isnan` overwrite is indiscriminate: it zeroes any NaN,
including one arriving from a bug in a user's return function. The matmul
propagates such a NaN instead. On already-broken input the old code silently
repairs and the new code fails loudly.

**Row-slice paths.** Low-memory code paths that loop over the current exogenous
state broadcast only one row of `pi` at a time. Their transient is already
linear in `N_z`, so there is nothing to fix and they were left alone.

## Applied to VFI Toolkit

The substitution replaced the broadcast at 128 sites across 48 finite-horizon
value-function-iteration files: the plain, divide-and-conquer, grid-interpolation
and combined tiers, together with their semi-exogenous counterparts. Every one of
those sites is the same two-dimensional shape, which is what made a single
recipe sufficient.

A size guard was written first — using the matmul only above a threshold — and
then removed. It cost essentially nothing at runtime (0.25ns per branch test),
but it would have doubled the code at every site, and, more importantly, every
test-bank grid sits about 150x below the crossover, so the guarded matmul would
never have been executed by any test. Its first real run would have been the
largest model in production. Unguarded, every test run exercises it. The price is
the ~1% on small models documented above.

Regression testing compared a full run of the toolkit's core finite-horizon test
bank before and after. Of 1,558 exact-zero checks, 152 moved off zero, none by
more than 2.1e-14, and every one belonged to a comparison the change explains:
low-memory tiers against the default (those paths keep the broadcast), value
functions recomputed from policy by code outside the converted set, and
cross-tier tests. No policy index changed in that bank.
