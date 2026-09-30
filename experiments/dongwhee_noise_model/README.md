# Dongwhee noise-model compatibility review

Source: Dongwhee's `01_Baseline.zip`, Slack file `F0C56JBPRGS`, shared in
[this message](https://pncel.slack.com/archives/D0BNPSPPZ27/p1790662621419269).
The retained review and original source snapshot are under
`results/dongwhee_noise_model_review_20260929/`.

The source archive is preserved in scratch at
`/data2/s2chitni/.tmp/dante/runs/dongwhee-noise-review-r_zap2pe/01_Baseline.zip`.
SHA256: `511c2f0e5a763899313610eb45e8ab60398f8b0ff11fa406ff5b180ec9edf94e`.

## What requires a Stim checkout

`01_Baseline/Makefile` includes `../Stim/src` and compiles Stim C++ sources into
the author's custom sampler. For that executable, `01_Baseline/` and `Stim/`
must be siblings. The upstream source is https://github.com/quantumlib/Stim.
A local source checkout is sufficient in principle; the archive contains no
Stim patch or requirement for a fork. Its C++ build has not been verified here.
The bundled executable is Linux ARM64; this host is x86_64 and would require
a rebuild. Pin the source revision if reproducing that path, since Stim does
not promise C++ API compatibility across releases.

## Integration route checked here

The author's `run.py` already builds a noise-annotated Stim circuit to supply
the MWPM decoder. That builder uses standard `DEPOLARIZE1`, `DEPOLARIZE2`, and
`PAULI_CHANNEL_1` instructions and works with our existing pip-installed
Stim 1.16.0. That circuit can supply both sampling and the decoder DEM.
Our seven decoders accept that DEM and its sampled detector arrays.

`probe.py` extracts only the reviewed parameter definitions and circuit-builder
block; it never invokes the package's experiment launcher, task-state handling,
or C++ executable. All six presets were checked in X and Z memory at d=3, r=3,
128 shots each. A Willow d=7, r=28 X-memory circuit was then decoded for 32
shots by weighted UF, correlated UF, BP1/2/5/10 + UF, and correlated MWPM.
The 192 native physical corrections were reconstructed and checked. These
are interface checks, not logical-error-rate measurements.

## Proposed integration

1. Extract a clean reusable circuit/noise builder and explicit device/rate
   configuration from the reviewed Python implementation. Keep the original
   source snapshot as a reference and require circuit/DEM equivalence tests.
2. Feed one saved circuit, DEM and detector/truth sample into our existing
   seven-decoder harness. Freeze device parameters and sampling seeds.
3. To isolate the noise-model change against SI1000, port the injection points
   to our CZ circuit, with explicit data/ancilla roles, round boundaries,
   immune EPR qubits and ideal preparation/readout. The reference builder's
   R/MR-based role/round inference is not a drop-in replacement for this circuit.
4. Reproduce the author's ordinary memory experiment separately when needed.
   Its X and Z memory circuits each have one logical observable, r=d and noisy
   preparation/readout. Our previous study used two jointly tracked observables,
   r=4d and ideal boundaries. Do not mix those failure-rate definitions.

The model combines gate, measurement/reset, Pauli-twirled T1/T2, spectator
idling, and device-dependent DD pulse errors. Its provided baseline rates are
p1=0.0001, p2=0.001, p_reset=0.001, with p_meas overridden by the device preset
(currently 0.001). Coherence times and gate/readout durations add independent
parameters; changing a single p does not scale the entire model.

No Stim checkout, package replacement, production noise-adapter change or
large Monte Carlo experiment was performed by this compatibility review.
The original C++ sampler has not been statistically cross-validated against
the Python sampling route. The exact reproduction caveats are in the report.

Subsequent setup: [stim_dante_setup](../stim_dante_setup/README.md) created the
`polarbearsrock/stim-dante` fork and built the unchanged reference C++ sampler
against its pinned Stim 1.16.0 source. Its build and OpenMP smoke checks are
recorded separately in `results/stim_dante_setup_20260929/`.
