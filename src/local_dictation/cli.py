"""The ``dictate`` command.

Recording commands run from desktop key bindings, so this module imports only
the standard library at load time; each command imports what it needs.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

RECORDING_COMMANDS = {
    "begin": "start recording",
    "end": "stop recording, then transcribe and type the text",
    "toggle": "end when recording, otherwise begin",
    "cancel": "stop recording and discard the audio",
    "transcribe": "transcribe and type a finished recording that was not delivered",
    "vad": "start or stop hands-free voice-activated dictation",
}
CONFIG_SECTIONS = ("stt", "dictation")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="dictate",
        description="Local push-to-talk and voice-activated dictation.",
    )
    commands = root.add_subparsers(dest="command", required=True)
    for name, help_text in RECORDING_COMMANDS.items():
        commands.add_parser(name, help=help_text)
    retry = commands.add_parser(
        "retry", help="transcribe and type a recording kept after a failed request"
    )
    retry.add_argument("path", type=Path, help="recording path named in the error")
    _add_vocabulary_parser(commands)
    _add_config_parser(commands)
    _add_services_parser(commands)
    doctor = commands.add_parser(
        "doctor", help="check dictation health and suggest safe fixes"
    )
    doctor.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="emit machine-readable diagnostics instead of the summary",
    )
    doctor.add_argument(
        "--fix",
        action="store_true",
        help="offer confirmation-gated repairs for issues that support them",
    )
    return root


def _add_vocabulary_parser(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    vocabulary = commands.add_parser(
        "vocabulary", help="manage personal transcript corrections"
    )
    actions = vocabulary.add_subparsers(dest="vocabulary_command", required=True)
    actions.add_parser("list", help="show stored replacements")
    add = actions.add_parser(
        "add", help="replace a misheard phrase with a written form"
    )
    add.add_argument("spoken", help="phrase as the recognizer writes it")
    add.add_argument("written", help="text to type instead")
    add.add_argument(
        "--force", action="store_true", help="overwrite a different stored form"
    )
    remove = actions.add_parser("remove", help="delete a replacement")
    remove.add_argument("spoken")
    export = actions.add_parser("export", help="print or write a JSON backup")
    export.add_argument("--output", type=Path)
    importer = actions.add_parser("import", help="merge a JSON backup")
    importer.add_argument("path", type=Path)
    importer.add_argument(
        "--replace", action="store_true", help="discard existing replacements first"
    )


def _add_config_parser(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    config = commands.add_parser("config", help="inspect or update configuration")
    actions = config.add_subparsers(dest="config_command", required=True)
    show = actions.add_parser("show", help="print stored configuration")
    show.add_argument("key", nargs="?", help="optional dotted key, such as stt.device")
    show.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="emit machine-readable output",
    )
    setter = actions.add_parser("set", help="validate and store one setting")
    setter.add_argument("key", help="dotted key, such as stt.device")
    setter.add_argument(
        "value",
        nargs="?",
        help="new value; omit for dictation.source to pick a PipeWire capture source",
    )
    reset = actions.add_parser("reset", help="restore defaults")
    reset.add_argument(
        "--section", choices=CONFIG_SECTIONS, help="reset only one section"
    )


def _add_services_parser(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    services = commands.add_parser(
        "services", help="manage the local-dictation user service"
    )
    actions = services.add_subparsers(dest="service_command", required=True)
    install = actions.add_parser("install", help="install and enable the unit")
    install.add_argument(
        "--force", action="store_true", help="replace a modified unit file"
    )
    actions.add_parser("start", help="start the service")
    actions.add_parser("stop", help="stop the service")
    actions.add_parser("restart", help="restart the service")
    actions.add_parser("status", help="show service status")
    actions.add_parser("audit", help="validate the installed unit and its drop-ins")
    service_logs = actions.add_parser("logs", help="show service logs")
    service_logs.add_argument("-f", "--follow", action="store_true")
    service_logs.add_argument("-n", "--lines", type=int, default=100)
    actions.add_parser("uninstall", help="stop, disable, and remove the unit")


def _dispatch_vocabulary(args: argparse.Namespace) -> None:
    from . import vocabulary

    if args.vocabulary_command == "list":
        vocabulary.list_entries()
    elif args.vocabulary_command == "add":
        vocabulary.add_replacement(args.spoken, args.written, force=args.force)
    elif args.vocabulary_command == "remove":
        vocabulary.remove_replacement(args.spoken)
    elif args.vocabulary_command == "export":
        vocabulary.export_entries(args.output)
    else:
        vocabulary.import_entries(args.path, replace=args.replace)


def _dispatch_config(args: argparse.Namespace) -> None:
    from . import config_management

    if args.config_command == "show":
        print(config_management.show_config(key=args.key, json_output=args.json_output))
        return
    if args.config_command == "set":
        result = config_management.commit_config_setting(args.key, args.value)
    else:
        result = config_management.reset_config(section=args.section)
    print(config_management.format_restart_notice(result.restart_services))


def _dispatch_services(args: argparse.Namespace) -> None:
    from . import service_manager

    command = args.service_command
    if command == "install":
        service_manager.install_service(force=args.force)
    elif command == "start":
        service_manager.start_service()
    elif command == "stop":
        service_manager.stop_service()
    elif command == "restart":
        service_manager.restart_service()
    elif command == "status":
        service_manager.status()
    elif command == "audit":
        raise SystemExit(service_manager.audit_service())
    elif command == "logs":
        service_manager.logs(follow=args.follow, lines=max(args.lines, 1))
    else:
        service_manager.uninstall_service()


def dispatch(args: argparse.Namespace) -> None:
    if args.command in RECORDING_COMMANDS:
        from .dictation import run
        from .user_config import load_user_config

        run(args.command, load_user_config())
    elif args.command == "retry":
        from .dictation import retry
        from .user_config import load_user_config

        retry(args.path.absolute(), load_user_config())
    elif args.command == "vocabulary":
        _dispatch_vocabulary(args)
    elif args.command == "config":
        _dispatch_config(args)
    elif args.command == "services":
        _dispatch_services(args)
    else:
        from .diagnostics import doctor

        raise SystemExit(doctor(json_output=args.json_output, fix=args.fix))


def main(arguments: Sequence[str] | None = None) -> None:
    args = parser().parse_args(arguments)
    try:
        dispatch(args)
    except Exception as exc:
        from .diagnostic_safety import COMMAND_FAILURE, log_diagnostic
        from .notifications import notify

        log_diagnostic("cli", "command_failed", f"{type(exc).__name__}: {exc}")
        print(f"dictate: {COMMAND_FAILURE}", file=sys.stderr)
        notify(COMMAND_FAILURE, error=True)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
