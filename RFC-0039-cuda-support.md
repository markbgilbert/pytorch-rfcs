
# [CUDA version for PyTorch CI/CD]

**Authors:**
* @atalman @malfet @tinglvv @nWEIdia @ptrblck


## **Motivation**

The proposal is to provide two main benefits

- Aligned decision policy provides transparent decision making;
- Incorporate decision points in the early release process to stagger CUDA migration from the RC release integration work.

This RFC defines **policy only**: when we add a CUDA version, when we remove one, and what a version must satisfy before it can ship. The mechanical step-by-step upgrade instructions are intentionally not part of this document, see [Relationship to RELEASE.md and the upgrade runbook](#relationship-to-releasemd-and-the-upgrade-runbook).

## **Relationship to RELEASE.md and the upgrade runbook**

This RFC is the counterpart of [RFC-0038 (CPython support)](https://github.com/pytorch/rfcs/blob/master/RFC-0038-cpython-support.md), which is referenced from the `Python` section of [`pytorch/pytorch` RELEASE.md](https://github.com/pytorch/pytorch/blob/main/RELEASE.md). The `Accelerator Software` section of RELEASE.md should likewise link to this RFC.

Division of ownership:

- **This RFC** — the policy: introduction criteria, deprecation criteria, the Legacy / Stable / Experimental categories, and the gates a version must pass before it is released.
- **[RELEASE.md](https://github.com/pytorch/pytorch/blob/main/RELEASE.md)** — the per-release record and the release mechanics. In particular [Release Compatibility Matrix](https://github.com/pytorch/pytorch/blob/main/RELEASE.md#release-compatibility-matrix) and [PyTorch CUDA Support Matrix](https://github.com/pytorch/pytorch/blob/main/RELEASE.md#pytorch-cuda-support-matrix) record the Stable and Experimental CUDA versions and supported architectures for each release; [Accelerator Software](https://github.com/pytorch/pytorch/blob/main/RELEASE.md#accelerator-software) states the support window; [Drafting RCs](https://github.com/pytorch/pytorch/blob/main/RELEASE.md#drafting-rcs-release-candidates-for-pytorch-and-domain-libraries) and [Modify release matrix](https://github.com/pytorch/pytorch/blob/main/RELEASE.md#modify-release-matrix) cover the RC and promotion steps. Whenever the matrix changes under this policy, RELEASE.md is updated in the same release.
- **CUDA/cuDNN upgrade runbook** — the mechanical instructions (docker images, magma, Windows AMI, nightly matrix, CI jobs, domain libraries), currently the archived [`pytorch/builder` CUDA_UPGRADE_GUIDE.MD](https://github.com/pytorch/builder/blob/main/CUDA_UPGRADE_GUIDE.MD). These reference specific PRs, scripts and versions and change on every upgrade, so they are kept as a runbook alongside the CI/CD scripts they describe, not in this RFC.

## **Proposed Policy and Process**

### We should introduce new version of CUDA when

- It enables new important GPU architecture (For example Blackwell with CUDA-12.8)
- It brings significant performance improvements
- It fixes significant correctness issues
- It significantly reduces binary/memory footprint
- It adds desired functionality or features

### We would deprecate version of CUDA when

As soon as we introduce a new Experimental Version we should consider moving the previous Experimental Version to Stable (see [Detailed Process of Transitioning CUDA version from Experimental to Stable](#detailed-process-of-transitioning-cuda-version-from-experimental-to-stable)), and decommission the previous Stable version. Typically we want to support at least 2 versions of CUDA with an optional exception for Legacy Version (see below). This matches the support window stated in [RELEASE.md, Accelerator Software](https://github.com/pytorch/pytorch/blob/main/RELEASE.md#accelerator-software).

- Optional Legacy Version: If we need to have 1 version for backend compatibility or to work around the current limitation. For example: CUDA older driver is incompatible with newer CUDA version. We should keep this version as static as possible (i.e. no cuDNN, NCCL, or other libraries) to avoid mixing the legacy stack with latest libs which can lead to unexpected behavior.
- Stable Version: This is a stable CUDA version that is used most of the time. This is the version we want to upload to PyPI.
- Latest Experimental Version: This is the latest version of CUDA that we want to support. Minimal requirement to qualify to be included in a PyTorch OSS Release is to have it **built in CD and covered by CI build and test jobs**, with all failures triaged before branch cut. Availability in nightly (CD) alone does not qualify a version for release.

### Release gating principles

These two rules take precedence over any individual step below:

1. **We do not release anything that is not tested in CI.** CD-only enablement produces binaries nobody has validated. A CUDA version is eligible for a release (including as Experimental) only once it has CI build *and* test coverage and its failures are known and tracked.
2. **We do not cut an RC for anything we are not planning to promote.** The release matrix (Legacy / Stable / Experimental) is decided before branch cut, and only versions in that matrix are built in the RC. An RC binary is a promise to ship; if we would not promote it to the final release, we do not build it in the RC.

### Detailed Process of Introducing new CUDA version

1. Evaluate CUDA update necessity. Please see section above: [We should introduce new version of CUDA when](#we-should-introduce-new-version-of-cuda-when)

2. Evaluate if we have all packages for update
  When: As soon as Update determined to be necessary. Start by creating RFC (see [example](https://github.com/pytorch/pytorch/issues/145544)) with possible CUDA matrix to support for next release.
  Goal: Make sure everything is available to perform complete upgrade of CUDA and dependencies

3. Update CUDA in CD (prerequisite for CI enablement, **not** a qualification for release)
  When: Evaluate if we have all packages for update is complete
  Goal: Make sure all Linux and Windows wheel and libtorch binaries are produced on nightly. Nightly binaries on their own do not qualify the version for a release; they exist so that CI can be enabled and so users can test early.

4. Update CUDA in CI (**this is the necessary condition to qualify for a CUDA version to be released as Experimental**)
  When: Update CUDA in CD is complete and nightly binaries are green
  Goal: Make sure the new CUDA version has both build and test jobs running in CI. All failing tests are identified, tracked with issues, and either fixed or explicitly accepted before branch cut. A version with no CI test coverage is not eligible for the release matrix.

5. Run benchmarks to compare new experimental CUDA version to current stable CUDA version
  When: Update CUDA in CI is complete
  Goal: Make sure we identify all performance regressions

6. Decide the release matrix before branch cut
  When: Before the release branch is cut
  Goal: Fix the Legacy / Stable / Experimental matrix for the release based on CI signal and benchmark results. Only versions in this matrix are built in the RC — we do not produce RC binaries for a CUDA version we are not planning to promote to the final release. Record the outcome in [RELEASE.md](https://github.com/pytorch/pytorch/blob/main/RELEASE.md#release-compatibility-matrix).


### Detailed Process of Transitioning CUDA version from Experimental to Stable

A CUDA version is promoted from Experimental to Stable only when all of the following conditions hold:

- **It is already in the Experimental state**, i.e. it has been part of at least one release as the Latest Experimental Version.
- **Full PyTorch CI and CD are running this version.** Not a subset: the complete CI build and test matrix (including the jobs that only run on the Stable version today), and all CD binaries — wheels and libtorch, Linux x86 and aarch64, Windows — produced on nightly. Benchmarks show no unresolved regressions against the current Stable version.
- **Downstream projects consuming PyTorch are ready to switch to this version and have tested it.** Domain libraries (torchvision, torchaudio) build and test against it, and the ecosystem consumers tracked in the update RFC issue have confirmed readiness.

As a consequence of promotion, this becomes the CUDA version we publish to PyPI.

1. Confirm full CI/CD coverage
  When: The version has shipped at least one release as Experimental
  Goal: Move the version to the full CI matrix and confirm it is green and not flaky over a sustained period. Any job still running only on the current Stable version is either enabled on the candidate or explicitly waived.

2. Confirm downstream readiness
  When: Full CI/CD coverage is confirmed
  Goal: Domain libraries and the downstream consumers tracked in the update RFC issue have built and tested against this version and confirmed they can switch. Promotion is not started while a required downstream consumer is still blocked.

3. Promote to Stable
  When: Steps 1 and 2 are complete, before the release branch is cut
  Goal: The version becomes the Stable version in the release matrix and the version uploaded to PyPI — validate that its wheels fit within the PyPI size limits before committing to this. Update the matrix in [RELEASE.md](https://github.com/pytorch/pytorch/blob/main/RELEASE.md#release-compatibility-matrix) and the `latest` tag handling for released images. The previous Stable version then becomes a candidate for deprecation, see the section below.

### Detailed Process of Deprecating CUDA version

1. Evaluate deprecation of legacy CUDA version from CI/CD
When: We completed CUDA update and previous experimental CUDA version is qualified to be stable and we have 3 supported versions (legacy, stable and experimental). Start by creating RFC [issue](https://github.com/pytorch/pytorch/issues/147383) to deprecate legacy CUDA version from CI/CD.
Goal: Make sure we support 2 versions of CUDA, supporting 3 versions can be an exception for certain release where we need to keep legacy version

2. Deprecate legacy CUDA version from CI/CD
When: Evaluate deprecation of legacy CUDA version from CI/CD is complete
Goal: Support for legacy CUDA versions is dropped, starting from PyTorch Domain Libraries and then in PyTorch core. First we drop CD support and then CI support, so that we never ship a version we have stopped testing. Update the matrix in [RELEASE.md](https://github.com/pytorch/pytorch/blob/main/RELEASE.md#release-compatibility-matrix) accordingly.
