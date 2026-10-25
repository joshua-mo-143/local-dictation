# Service hardening

`local-dictation.service` runs the speech model with as much of systemd's
sandboxing as CUDA and the first model download allow. It's built for a
single-user workstation: it limits accidental exposure and how far a compromised
model process can reach. It doesn't make the model safe to run on a machine
shared with people you don't trust.

## What the service can reach

| Area | Policy |
| --- | --- |
| Files | Everything is read-only (`ProtectSystem=strict`, `ProtectHome=read-only`) except the user runtime directory `%t`, which holds the socket and `%t/local-dictation`, and the Hugging Face cache (`CacheDirectory=huggingface`), so a fresh install can download the model. `/tmp` is private |
| Network | Unix sockets for `dictate`, IPv4 and IPv6 for model downloads, and netlink for CUDA device discovery. Nothing else (`RestrictAddressFamilies=`) |
| Devices | All visible, because CUDA needs the NVIDIA device nodes |
| Privileges | No capabilities, no privilege gain through `execve`, no creating setuid or setgid files, no new namespaces, no realtime scheduling, and native system calls only |
| Kernel and processes | Kernel tunables, modules, and logs, the clock, the hostname, and control groups are protected, and other users' processes are hidden in `/proc` |

The runtime directory `%t/local-dictation` is owner-only (`0700`) and survives
service restarts (`RuntimeDirectoryPreserve=yes`), so recordings kept for a retry
aren't lost when the service restarts. Like the rest of `%t`, it's removed when
you log out. Files the service creates are owner-only (`UMask=0077`).

## Resources

- Memory: throttled above 4 GB and stopped above 6 GB (`MemoryHigh=`,
  `MemoryMax=`). Swap is unlimited, to avoid an unmeasured failure while CUDA
  starts up. On an out-of-memory kill, systemd stops the whole service
  (`OOMPolicy=stop`), and `OOMScoreAdjust=200` makes the kernel more likely to
  pick it than your desktop.
- At most 512 tasks and 4096 open files.
- No core dumps (`LimitCORE=0`), so a crash doesn't write process memory, which
  can hold audio and transcripts, to disk.
- Restarts 5 seconds after a failure, up to five starts a minute, and gets 15
  seconds to stop.

These are safety ceilings, not measured minimums.

## Deliberately left out

`MemoryDenyWriteExecute=`, `SystemCallFilter=`, `IPAddressDeny=`, and device
restrictions (`PrivateDevices=`, `DevicePolicy=`, `DeviceAllow=`,
`DeviceDeny=`) aren't used. Device restrictions break CUDA, and the others
haven't been shown to work with CUDA, ONNX Runtime, and an unprivileged user
manager. The checks reject them so they don't creep in by accident.

## Environment

The unit sets only `CUDA_CACHE_PATH`, `HF_HOME`, and `TMPDIR`, and loads no
`EnvironmentFile=`. The launcher forces the same values, along with `HOME`,
`XDG_CACHE_HOME`, and `XDG_RUNTIME_DIR`, so the service's paths can't be
redirected through the environment. It also forces `ORT_DISABLE_TELEMETRY=1`,
which keeps ONNX Runtime from collecting telemetry. Engine, device, and audio
choices live in `config.toml`, never in the unit.

## How it's enforced

- `python -m local_dictation.service_units`, which runs in CI, checks the shipped
  unit. Every controlled directive must appear exactly once with the expected
  value, the left-out directives must be absent, and the environment and
  `ExecStart=` must match. It also checks that the checkout and wheel ship the
  same unit, and that the unit agrees with the package and the launcher.
- When `systemd-analyze` is installed, it also runs `systemd-analyze verify` on
  the unit in a staged root; `--require-systemd-analyze` makes a missing
  `systemd-analyze` an error. Some systemd versions can't verify a user unit
  there, so the check falls back to system context and says so in its output.
  `--require-user-context` makes that fallback an error.
- `dictate services audit` checks the installed unit as systemd sees it,
  including drop-ins. See [Service management](service-management.md#what-the-audit-checks).

## Residual risks

- Processes running as you are trusted. The socket is owner-only, but any of
  them can connect to it. The service can write anywhere in your runtime
  directory, which also holds other programs' sockets, because its own socket
  lives there.
- The model cache and network access are open so a fresh install can download
  the model. Pre-provisioning the cache would allow a narrower policy.
- CI doesn't exercise CUDA, ONNX Runtime, faster-whisper, or NVIDIA driver
  behaviour. `ProtectProc=invisible`, the address-family list, and the resource
  ceilings need checking on real hardware.

## Checking a host

From the checkout, verify the shipped unit, then the installed one:

```fish
uv run python -m local_dictation.service_units --require-systemd-analyze
dictate services audit
```

To confirm that your user manager actually enforces the file-system sandbox,
run a throwaway unit with the same home protection. It should fail with
`Read-only file system`:

```fish
systemd-run --user --wait --collect --pipe \
    --property=ProtectHome=read-only \
    python3 -c 'from pathlib import Path; Path.home().joinpath("ld-probe").touch()'
```

Watch memory, tasks, and restarts while dictating:

```fish
systemctl --user show local-dictation.service \
    --property=MemoryCurrent --property=MemoryPeak --property=TasksCurrent \
    --property=NRestarts --property=Result
```

The [hardware smoke checks](hardware-smoke.md) cover CUDA model loading,
capture, and text insertion.
