"""Two isolated GPU workers, 32-fit barrier and source-only readout freeze.

The caller owns the global nohup/run lock and the final report/export step.
This coordinator marks only ALL_GPU_WORK_COMPLETED, never COMPLETED.
"""
import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
from spec import QUEUES, STUDY, ORDER, READOUTS
TARGETS = ("ped1", "ped2", "avenue")
WORKER_PHASES = ("preflight", "fit", "probe", "evaluate")
SOURCE_ARTIFACTS = ("source_probe.json", "source_probe.npz", "calibration.json")
COORDINATOR_PHASES = ("prepare", "selection", "freeze")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + str(os.getpid()) + "." + str(time.time_ns()) + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def release_identity():
    """Use exactly the original common.release_identity manifest contract."""
    manifest = read_json(ROOT / "release_manifest.json")
    for name, expected in manifest["files"].items():
        if sha(ROOT / name) != expected:
            raise RuntimeError("Release modified: " + name)
    return digest(manifest)


def status(phase, **extra):
    write_json(OUT / "state.json", {"phase": phase, "updated_unix": time.time(), **extra})


def lane_path(lane):
    return OUT / "workers" / ("gpu" + lane)


class SignalStop(BaseException):
    def __init__(self, number):
        self.number = number
        super().__init__("Received signal " + str(number))


def on_signal(number, _frame):
    raise SignalStop(number)


def install_signals():
    previous = {}
    for number in (signal.SIGINT, signal.SIGTERM):
        previous[number] = signal.signal(number, on_signal)
    return previous


def restore_signals(previous):
    for number, handler in previous.items():
        signal.signal(number, handler)


def stop_owned(process, number=signal.SIGTERM, group=False, cleanup_timeout=15):
    """Signal only a Popen child, or the session created for that child."""
    if process.poll() is not None:
        return
    if group and os.name == "posix":
        try:
            os.killpg(process.pid, number)
        except ProcessLookupError:
            pass
    else:
        process.send_signal(number)
    # Cleanup timeouts never limit a research fit or GPU-idle wait.
    try:
        process.wait(timeout=cleanup_timeout)
    except subprocess.TimeoutExpired:
        if group and os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            process.kill()
        process.wait()


def wait_gpu(lane, poll_seconds=30):
    """Read-only polling without creating a CUDA context or a deadline."""
    while True:
        raw = subprocess.check_output(
            ["nvidia-smi", "--id=" + lane, "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
        used, utilization = [int(part.strip()) for part in raw.split(",")]
        processes = subprocess.check_output(
            ["nvidia-smi", "--id=" + lane, "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
        if used < 300 and utilization <= 2 and not processes:
            print("Physical GPU" + lane + " idle; proceeding.", flush=True)
            return
        message = f"Physical GPU{lane} occupied: {used}MiB,{utilization}%. Waiting without CUDA context."
        write_json(lane_path(lane) / "state.json", {"phase": "WAITING_FOR_GPU", "updated_unix": time.time(), "message": message})
        print(message, flush=True)
        time.sleep(poll_seconds)


def lane_artifacts_complete(lane, phase):
    """Read-only skip for already committed fits/scores; preflights rerun guards."""
    if phase == "preflight":
        return False
    prepared_hash = sha(OUT / "prepared.json")
    preflight_hash = sha(lane_path(lane) / "preflight.json")
    preflight = read_json(lane_path(lane) / "preflight.json")
    release = release_identity()
    if (preflight.get("status") != "PASS" or preflight.get("prepared_sha256") != prepared_hash
            or preflight.get("release") != release or preflight.get("physical_gpu") != int(lane)
            or preflight.get("queue") != QUEUES[lane]):
        raise RuntimeError("Preflight identity changed: GPU" + lane)
    for candidate in QUEUES[lane]:
        destination = OUT / candidate
        needed = ("config.json", "completed.json", "last.pt")
        if not all((destination / name).is_file() for name in needed):
            return False
        config = read_json(destination / "config.json")
        done = read_json(destination / "completed.json")
        expected_config = {"study": STUDY, "candidate": candidate, "recipe": candidate.split("_s")[0],
                           "prepared_sha256": prepared_hash, "preflight_sha256": preflight_hash,
                           "physical_gpu": int(lane), "hardware_identity": preflight["environment"]["hardware_identity"],
                           "seed": 17, "iterations": 5000, "batch_size": 8, "release": release}
        if (
            config != expected_config
            or done.get("step") != 5000
            or done.get("config_hash") != digest(config)
            or done.get("checkpoint_sha256") != sha(destination / "last.pt")
        ):
            raise RuntimeError("Completed fit changed: " + candidate)
        if phase in ("probe", "evaluate"):
            if not all((destination / name).is_file() for name in SOURCE_ARTIFACTS):
                return False
            import probes
            binding = probes.source_binding(candidate)
            summary = read_json(destination / "source_probe.json")
            calibration = read_json(destination / "calibration.json")
            if (summary.get("binding") != binding
                    or summary.get("scores_sha256") != sha(destination / "source_probe.npz")
                    or summary.get("calibration_sha256") != sha(destination / "calibration.json")
                    or summary.get("normalcy_bn_unchanged") is not True
                    or calibration.get("binding") != binding
                    or calibration.get("normalcy_bn_unchanged") is not True):
                raise RuntimeError("Completed source probe changed: " + candidate)
        if phase == "evaluate":
            frozen_hash = sha(OUT / "freeze.json")
            freeze = read_json(OUT / "freeze.json")
            if (freeze.get("study") != STUDY or freeze.get("release") != release
                    or freeze.get("prepared_sha256") != prepared_hash
                    or freeze.get("worker_preflights", {}).get(lane) != preflight_hash
                    or freeze.get("evaluation_batch") != 16
                    or freeze.get("source_validation_sha256") != sha(OUT / "source_validation.json")
                    or freeze.get("source_selection_sha256") != sha(OUT / "source_selection.json")
                    or freeze.get("models", {}).get(candidate) != done):
                raise RuntimeError("Evaluation freeze identity changed: " + candidate)
            validate_source_validation()
            for target in TARGETS:
                scores = destination / (target + ".npz")
                metadata = scores.with_suffix(".json")
                if not scores.is_file() or not metadata.is_file():
                    return False
                meta = read_json(metadata)
                if (meta.get("freeze_sha256") != frozen_hash or meta.get("scores_sha256") != sha(scores)
                        or meta.get("calibration_sha256") != sha(destination / "calibration.json")
                        or meta.get("source_selection_sha256") != sha(OUT / "source_selection.json")
                        or meta.get("normalcy_eval") is not True
                        or meta.get("normalcy_bn_unchanged") is not True):
                    raise RuntimeError("Completed scores changed: " + candidate + "/" + target)
    return True


def phase_command(phase, lane, args):
    command = [sys.executable, str(ROOT / "crosscheck.py"), "--phase", phase,
               "--workers", str(args.workers), "--evaluation-batch", str(args.evaluation_batch)]
    if lane is not None:
        command += ["--lane", lane]
    return command


def worker(args):
    """Hold one lane lock and record every attempt, including signal exits."""
    import fcntl
    import study_support

    lane, phase = args.lane, args.phase
    destination = lane_path(lane)
    destination.mkdir(parents=True, exist_ok=True)
    lock = (destination / "worker.lock").open("a")
    # A duplicate supervisor never overwrites the active supervisor's receipt.
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (destination / "worker.pid").write_text(str(os.getpid()) + "\n", encoding="utf-8")
    os.environ["CUDA_VISIBLE_DEVICES"] = lane
    started = time.time()
    process, code, skipped, received_signal, child_exit = None, 1, False, None, None
    previous = install_signals()
    command = phase_command(phase, lane, args)
    try:
        study_support.validate_parent()
        study_support.validate_prepared()
        if phase != "preflight":
            study_support.validate_preflight(lane)
        skipped = lane_artifacts_complete(lane, phase)
        if skipped:
            code = 0
            complete_state = {"phase": {"fit": "ALL_LANE_FITS_COMPLETED", "probe": "ALL_LANE_PROBES_COMPLETED",
                                        "evaluate": "ALL_LANE_EVALUATIONS_COMPLETED"}[phase],
                              "updated_unix": time.time(), "already_committed": True}
            key = {"fit": "completed_fits", "probe": "completed_source_probes", "evaluate": "primary_AUROC_rows"}[phase]
            complete_state[key] = len(QUEUES[lane]) if phase != "evaluate" else 3 * len(QUEUES[lane])
            if phase == "evaluate":
                complete_state["readout_AUROC_rows"] = 3 * len(QUEUES[lane]) * len(READOUTS)
            write_json(destination / "state.json", complete_state)
            print(f"GPU{lane}: {phase} already committed; validated and skipped.", flush=True)
        else:
            print(f"GPU{lane}: starting phase {phase}", flush=True)
            wait_gpu(lane)
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, start_new_session=True)
            child_exit = process.wait()
            code = child_exit if child_exit >= 0 else 128 - child_exit
            if child_exit < 0:
                received_signal = -child_exit
        study_support.validate_parent()
        study_support.validate_prepared()
    except SignalStop as error:
        received_signal = error.number
        code = 128 + error.number
        if process is not None:
            stop_owned(process, error.number, group=True)
        print(f"GPU{lane}: {phase} interrupted by signal {error.number}", flush=True)
    except BaseException as error:
        if process is not None:
            stop_owned(process, group=True)
        code = 1
        print(f"GPU{lane}: {phase} failed: {error}", flush=True)
    finally:
        restore_signals(previous)
        receipt = {"physical_gpu": int(lane), "phase": phase, "exit": code,
                   "attempt_id": args.attempt_id, "started_unix": started,
                   "finished_unix": time.time(), "supervisor_pid": os.getpid(),
                   "child_pid": process.pid if process is not None else None,
                   "child_exit": child_exit,
                   "signal": received_signal if received_signal is not None else (-code if code < 0 else None),
                   "already_committed": skipped, "command": command}
        write_json(destination / "attempts" / (args.attempt_id + "_exit.json"), receipt)
        write_json(destination / (phase + "_exit.json"), receipt)
        if code:
            write_json(destination / "state.json", {"phase": "FAILED", "failed_stage": phase,
                       "exit": code, "signal": receipt["signal"], "updated_unix": time.time()})
        lock.close()
    return code


def parallel_phase(phase, args):
    status("BOTH_" + phase.upper())
    jobs, exits = [], {}
    try:
        for lane in QUEUES:
            destination = lane_path(lane)
            destination.mkdir(parents=True, exist_ok=True)
            log = (destination / "execution.log").open("a", encoding="utf-8")
            attempt = phase + "-" + lane + "-" + str(time.time_ns())
            command = [sys.executable, str(ROOT / "coordinate.py"), "--worker", "--lane", lane,
                       "--phase", phase, "--attempt-id", attempt, "--workers", str(args.workers),
                       "--evaluation-batch", str(args.evaluation_batch)]
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = lane
            try:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                           stdin=subprocess.DEVNULL, env=environment, start_new_session=True)
            except BaseException:
                log.close()
                raise
            jobs.append((lane, process, log, attempt))
        # A failed lane does not discard the other lane's healthy progress.
        exits = {lane: process.wait() for lane, process, _, _ in jobs}
    except BaseException:
        for _, process, _, _ in jobs:
            # A supervisor has to finish its own 15-second child-session cleanup
            # and write the receipt before the coordinator may escalate it.
            stop_owned(process, cleanup_timeout=35)
        raise
    finally:
        for _, _, log, _ in jobs:
            log.close()
    receipts = {}
    for lane, _, _, attempt in jobs:
        receipt = read_json(lane_path(lane) / (phase + "_exit.json"))
        if (receipt.get("attempt_id") != attempt or receipt.get("exit") != exits[lane]
                or receipt.get("physical_gpu") != int(lane) or receipt.get("phase") != phase):
            raise RuntimeError("Invalid current lane completion receipt: " + lane)
        receipts[lane] = sha(lane_path(lane) / (phase + "_exit.json"))
    if any(exits.values()):
        raise RuntimeError(f"{phase} worker exits={exits}; inspect worker logs and resume the same launch")
    return exits, receipts


def coordinator_phase(phase, args):
    started = time.time()
    command = phase_command(phase, None, args)
    process, code, received_signal, child_exit = None, 1, None, None
    try:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, start_new_session=True)
        child_exit = process.wait()
        code = child_exit if child_exit >= 0 else 128 - child_exit
        if child_exit < 0:
            received_signal = -child_exit
    except SignalStop as error:
        received_signal, code = error.number, 128 + error.number
        if process is not None:
            stop_owned(process, error.number, group=True)
        raise
    except BaseException:
        if process is not None:
            stop_owned(process, group=True)
        raise
    finally:
        receipt = {"phase": phase, "exit": code, "physical_gpu": None, "started_unix": started,
                   "finished_unix": time.time(), "signal": received_signal if received_signal is not None else (-code if code < 0 else None),
                   "child_exit": child_exit,
                   "child_pid": process.pid if process is not None else None, "command": command}
        write_json(OUT / "coordinator" / (phase + "_exit.json"), receipt)
        write_json(OUT / "coordinator" / "attempts" / (phase + "-" + str(time.time_ns()) + "_exit.json"), receipt)
    if code:
        raise RuntimeError(f"Coordinator {phase} exited {code}")
    return sha(OUT / "coordinator" / (phase + "_exit.json"))


def cross_card_gate():
    import study_support
    for lane in QUEUES: study_support.validate_preflight(lane)
    records = {lane: read_json(lane_path(lane) / "preflight.json") for lane in QUEUES}
    binding = {lane: sha(lane_path(lane) / "preflight.json") for lane in QUEUES}
    prepared_hash = sha(OUT / "prepared.json")
    for lane, record in records.items():
        if (record.get("status") != "PASS" or record.get("prepared_sha256") != prepared_hash
                or record.get("queue") != QUEUES[lane] or record.get("physical_gpu") != int(lane)
                or record.get("seed17_original_updater_parity", {}).get("status") != "PASS"):
            raise RuntimeError("Invalid source preflight: " + lane)
    same_initial = records["0"]["shared_original_initial_hashes"] == records["1"]["shared_original_initial_hashes"]
    same_update = records["0"]["seed17_original_updater_parity"]["state_sha256"] == records["1"]["seed17_original_updater_parity"]["state_sha256"]
    value = {"status": "PASS" if same_initial and same_update else "FAILED",
             "parent_binding_sha256": sha(OUT / "parent_binding.json"),
             "shared_initial_tensors_exact": same_initial,
             "first_source_baseline_model_Adam_RNG_exact": same_update,
             "worker_preflights": binding,
             "scope": "Disposable first source batch only; no assertion of identical complete5000-step cross-GPU training trajectories"}
    path = OUT / "cross_card_parity.json"
    if path.exists() and read_json(path) != value:
        raise RuntimeError("Cross-card parity receipt changed")
    write_json(path, value)
    if value["status"] != "PASS":
        raise RuntimeError("Cross-card source parity failed; no formal fits started")
    return sha(path)


def validate_source_validation():
    record = read_json(OUT / "source_validation.json")
    if (record.get("study") != STUDY or record.get("prepared_sha256") != sha(OUT / "prepared.json")
            or record.get("source_split_sha256") != sha(OUT / "source_split.json")
            or record.get("source_probe_plan_sha256") != sha(OUT / "source_probe_plan.npy")
            or set(record.get("artifacts", {})) != set(ORDER)):
        raise RuntimeError("Source validation manifest identity changed")
    for candidate in ORDER:
        values = record["artifacts"][candidate]
        if set(values) != set(SOURCE_ARTIFACTS + ("completed.json",)):
            raise RuntimeError("Source validation artifact set changed: " + candidate)
        for name, expected in values.items():
            if sha(OUT / candidate / name) != expected:
                raise RuntimeError("Frozen source validation artifact changed: " + candidate + "/" + name)
    selection = read_json(OUT / "source_selection.json")
    if selection.get("source_validation_sha256") != sha(OUT / "source_validation.json"):
        raise RuntimeError("Source selection is not bound to source validation")


def completion_artifacts():
    validate_source_validation()
    names = ["parent_binding.json", "prepared.json", "source_plan.npy", "source_manifest.json", "cross_card_parity.json", "freeze.json",
             "source_split.json", "source_probe_plan.npy", "source_selection.json", "source_validation.json"]
    names += ["workers/gpu" + lane + "/" + name for lane in QUEUES for name in ("preflight.json", "optimizer_regression.json")]
    import study_support
    for lane in QUEUES:
        study_support.validate_preflight(lane)
        if not lane_artifacts_complete(lane, "evaluate"):
            raise RuntimeError("Missing final frozen score artifacts: GPU" + lane)
        for candidate in QUEUES[lane]:
            names += [candidate + "/" + name for name in ("config.json", "completed.json", "last.pt", "train.jsonl")]
            names += [candidate + "/" + name for name in SOURCE_ARTIFACTS]
            names += [candidate + "/" + target + suffix for target in TARGETS for suffix in (".npz", ".json")]
    return {name: sha(OUT / name) for name in names}


def completed_noop():
    path = OUT / "run_completion.json"
    if not path.exists():
        return False
    import study_support
    study_support.validate_parent()
    study_support.validate_prepared()
    record = read_json(path)
    if (record.get("status") != "ALL_GPU_WORK_COMPLETED" or record.get("queues") != QUEUES
            or record.get("freeze_sha256") != sha(OUT / "freeze.json")
            or record.get("artifacts") != completion_artifacts()):
        raise RuntimeError("Completed cross-check evidence changed")
    for phase in WORKER_PHASES:
        if record.get("worker_exits", {}).get(phase) != {lane: 0 for lane in QUEUES}:
            raise RuntimeError("Completed worker exits changed")
        receipts = record.get("worker_exit_receipts", {}).get(phase, {})
        if set(receipts) != set(QUEUES):
            raise RuntimeError("Completed worker receipt map is incomplete")
        for lane, expected in receipts.items():
            path = lane_path(lane) / (phase + "_exit.json")
            receipt = read_json(path)
            attempt = receipt.get("attempt_id")
            if (sha(path) != expected or receipt.get("phase") != phase
                    or receipt.get("physical_gpu") != int(lane) or receipt.get("exit") != 0
                    or not isinstance(attempt, str) or not attempt or Path(attempt).name != attempt
                    or sha(lane_path(lane) / "attempts" / (attempt + "_exit.json")) != expected):
                raise RuntimeError("Completed worker receipt changed")
    for phase in COORDINATOR_PHASES:
        path = OUT / "coordinator" / (phase + "_exit.json")
        receipt = read_json(path)
        if (sha(path) != record.get("coordinator_exit_receipts", {}).get(phase)
                or receipt.get("phase") != phase or receipt.get("exit") != 0
                or receipt.get("physical_gpu") is not None):
            raise RuntimeError("Completed coordinator receipt changed")
    print("All GPU work already completed; validated committed artifacts.", flush=True)
    return True


def coordinate(args):
    # Check committed completion before updating any receipt or global state.
    if completed_noop():
        return
    prepare_receipt = coordinator_phase("prepare", args)
    preflight, preflight_receipts = parallel_phase("preflight", args)
    parity_hash = cross_card_gate()
    fits, fit_receipts = parallel_phase("fit", args)
    if not all(lane_artifacts_complete(lane, "fit") for lane in QUEUES):
        raise RuntimeError("The 32-fit final5000 barrier is incomplete; no source probes started")
    probes, probe_receipts = parallel_phase("probe", args)
    if not all(lane_artifacts_complete(lane, "probe") for lane in QUEUES):
        raise RuntimeError("The 32-arm source probe barrier is incomplete; no selection/evaluation started")
    status("SELECTING_FROM_HELDOUT_SOURCE_ONLY")
    selection_receipt = coordinator_phase("selection", args)
    validate_source_validation()
    status("FREEZING_ALL32_FINAL_CHECKPOINTS_AND_SOURCE_READOUTS")
    freeze_receipt = coordinator_phase("freeze", args)
    evaluations, evaluation_receipts = parallel_phase("evaluate", args)
    import study_support
    study_support.validate_parent()
    study_support.validate_prepared()
    write_json(OUT / "run_completion.json", {
        "status": "ALL_GPU_WORK_COMPLETED", "completed_fits": len(ORDER), "completed_source_probes": len(ORDER),
        "AUROC_rows": len(ORDER)*3, "readout_AUROC_rows": len(ORDER)*3*len(READOUTS),
        "worker_exits": {"preflight": preflight, "fit": fits, "probe": probes, "evaluate": evaluations},
        "worker_exit_receipts": {"preflight": preflight_receipts, "fit": fit_receipts,
                                 "probe": probe_receipts, "evaluate": evaluation_receipts},
        "coordinator_exit_receipts": {"prepare": prepare_receipt, "selection": selection_receipt, "freeze": freeze_receipt},
        "queues": QUEUES, "wall_time_limit": None, "freeze_sha256": sha(OUT / "freeze.json"),
        "cross_card_parity_sha256": parity_hash, "source_validation_sha256": sha(OUT / "source_validation.json"),
        "source_selection_sha256": sha(OUT / "source_selection.json"), "artifacts": completion_artifacts(),
        "finished_unix": time.time(), "report_export_required": True,
    })
    status("ALL_GPU_WORK_COMPLETED", completed_fits=len(ORDER), AUROC_rows=len(ORDER)*3,
           readout_AUROC_rows=len(ORDER)*3*len(READOUTS), report_export_required=True)
    print("32 fits, 32 source probes and 96 target score arrays completed; report/export must validate 672 readout rows.", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--lane", choices=list(QUEUES), help=argparse.SUPPRESS)
    parser.add_argument("--phase", choices=WORKER_PHASES, help=argparse.SUPPRESS)
    parser.add_argument("--attempt-id", help=argparse.SUPPRESS)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--evaluation-batch", type=int, default=16)
    args = parser.parse_args(argv)
    if args.workers < 0 or args.evaluation_batch != 16:
        parser.error("Nonnegative loader workers and frozen evaluation batch16 required")
    if args.worker:
        if args.lane is None or args.phase is None or not args.attempt_id:
            parser.error("Internal worker requires lane, phase and attempt-id")
        if Path(args.attempt_id).name != args.attempt_id:
            parser.error("Internal attempt-id must be a filename")
        return worker(args)
    if any(value is not None for value in (args.lane, args.phase, args.attempt_id)):
        parser.error("Lane and phase arguments are internal worker options")
    previous = install_signals()
    try:
        coordinate(args)
    except SignalStop as error:
        status("INTERRUPTED", signal=error.number, exit=128 + error.number)
        return 128 + error.number
    except Exception as error:
        status("FAILED", message=str(error))
        raise
    finally:
        restore_signals(previous)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
