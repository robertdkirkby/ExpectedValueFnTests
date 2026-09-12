Essentially this is a repo containing a bunch of tests comparing whether to compute the expected value fn as
V.*pi_z
OR
V*pi_z
(so .* or *)

Takeaway was that .* was almost twice as fast for small models, but * was 10x or more faster for big models (say N_z>1000, depends on N_a as well).
And * was able to solve large models where .* gave out-of-memory errors.

Viewed in the context of the finite horizon value fn, taking twice as long to compute expectations was only a 1% slowdown for the VFI as a whole.
Whereas, pushing out where out-of-memory errors occur is a meaningful improvement.

So I decided to go with * ; replacing the .* which is what VFI Toolkit used to use.

One option would have been an if-else statement on the size of N_z (and N_a) to determine which of .* and * to use. 
But since the slowdown of * for small models is only 1% in the context of VFI (even if near doubling runtime for the EV itself)
I figured it was not worth the code complexity of putting if-else everywhere.
