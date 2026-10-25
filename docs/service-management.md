# Service management

`local-dictation.service` is a systemd user unit. systemd supervises the speech
server, restarts it when it fails, and keeps its logs. `dictate services` wraps
`systemctl --user` and `journalctl --user` for that one unit, and never touches
any other unit.

```fish
dictate services install
dictate services start
dictate services status
dictate services restart
dictate services stop
dictate services audit
dictate services logs --follow
dictate services uninstall
```

| Command | What it does |
| --- | --- |
| `install [--force]` | Copies the shipped unit to `~/.config/systemd/user/`, reloads systemd, and enables the unit so it starts when you log in. If a different unit is already installed, it stops and asks for `--force` |
| `start`, `stop`, `restart` | Starts, stops, or restarts the service. `stop` fails if the unit isn't installed |
| `status` | Prints whether the service is active and enabled, such as `local-dictation.service: active (enabled)` |
| `audit` | Checks the installed unit and the service's state without changing anything, and exits non-zero on any problem |
| `logs [-f] [-n LINES]` | Shows the last 100 journal lines, or `LINES`, and keeps following with `-f` |
| `uninstall` | Stops and disables the service, removes the unit file if it still matches the shipped one, and reloads systemd |

`uninstall` keeps a unit file you changed, and says so, so you can review it
first. It also leaves your settings, vocabulary, service environment, and model
cache in place.

## What the audit checks

`dictate services audit` reads the unit as systemd sees it, including any
drop-ins, and checks that:

- It runs `~/local-dictation/.venv-service/bin/local-dictation-server`, and that
  file exists and is executable.
- Its sandboxing, resource, and restart settings match the shipped values
  exactly. Drop-ins that loosen or tighten them both fail; see
  [Service hardening](service-hardening.md) for why.
- It loads no `EnvironmentFile=`, and sets only the shipped environment
  variables, with their shipped values.
- It leaves devices unrestricted, because CUDA needs the NVIDIA device nodes,
  and adds no `IPAddressDeny=` or `SystemCallFilter=`.
- The service hasn't failed and isn't stuck in a restart loop.

It also prints the device, backend, and model from your configuration, and the
service's state, memory, and task count.

## Restarts

The service restarts 5 seconds after a failure, up to five starts a minute. Past
that, systemd gives up and marks it failed until you start it again. If it
refuses because the service started too often, clear the failure first:

```fish
systemctl --user reset-failed local-dictation.service
dictate services start
```

[Troubleshooting](troubleshooting.md#the-service-wont-start-or-keeps-restarting)
covers finding the cause.

## Configuration changes

The service reads `config.toml` once, when it starts. After changing an `stt.*`
setting, `dictation.prompt`, or `dictation.replacements`, run
`dictate services restart`; `dictate config set` tells you when that's needed.
See [When changes apply](configuration.md#when-changes-apply).

The unit itself holds no engine, device, or audio choices, only the paths the
service owns. Make those choices in `config.toml`, not in a drop-in: the
audit rejects drop-ins that add environment variables.

## Updates and drift

When the shipped unit changes, `install` refuses to overwrite the old one until
you've reviewed it. [Updating](installation.md#updating) covers the update
steps, and
[Troubleshooting](troubleshooting.md#the-installed-unit-drifted) covers
replacing a unit that drifted.
