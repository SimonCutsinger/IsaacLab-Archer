#!/usr/bin/env python3

# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import shutil
import subprocess
from pathlib import Path

from isaaclab_container_utils import apptainer_utils, x11_utils
from isaaclab_container_utils.isaaclab_container_interface import IsaacLabContainerInterface


def main():
    parser = argparse.ArgumentParser(description="Utility for using Docker with Isaac Lab.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # We have to create separate parent parsers for common options to our subparsers
    profile_parser = argparse.ArgumentParser(add_help=False)
    profile_parser.add_argument("profile", nargs="?", default="base", help="Optional container profile specification.")
    file_composition_parser = argparse.ArgumentParser(add_help=False)
    file_composition_parser.add_argument(
        "--add_yamls",
        nargs="*",
        default=None,
        help=(
            "Allows additional .yaml files to be passed to the docker compose command. Files will be merged with"
            " docker-compose.yaml in the order in which they are provided."
        ),
    )
    file_composition_parser.add_argument(
        "--add_envs",
        nargs="*",
        default=None,
        help=(
            "Allows additional .env files to be passed to the docker compose command. Files will be merged with"
            " .env.base in the order in which they are provided."
        ),
    )

    # Actual command definition begins here
    subparsers.add_parser(
        "start",
        help="Build the docker image and create the container in detached mode.",
        parents=[profile_parser, file_composition_parser],
    )
    subparsers.add_parser(
        "enter", help="Begin a new bash process within an existing Isaac Lab container.", parents=[profile_parser]
    )
    subparsers.add_parser(
        "copy", help="Copy build and logs artifacts from the container to the host machine.", parents=[profile_parser]
    )
    subparsers.add_parser(
        "stop", help="Stop the docker container and remove it.", parents=[profile_parser, file_composition_parser]
    )
    subparsers.add_parser("push", help="Push the docker image to the cluster.", parents=[profile_parser])
    config_parser = subparsers.add_parser(
        "config",
        help="Parse, resolve and render compose file in canonical format.",
        parents=[profile_parser, file_composition_parser],
    )
    config_parser.add_argument("--output_dir", help="Path to the file where the config should be stored. Defaults to 'None' and prints to stdin.")
    job_parser = subparsers.add_parser("job", help="Submit a job to the cluster.", parents=[profile_parser])
    job_parser.add_argument(
        "job_args", nargs=argparse.REMAINDER, help="Optional arguments specific to the executed script."
    )

    args = parser.parse_args()

    if not shutil.which("docker"):
        raise RuntimeError("Docker is not installed! Please check the 'Docker Guide' for instruction.")

    # Creating container interface
    ci = IsaacLabContainerInterface(context_dir=Path(__file__).resolve().parent, profile=args.profile, yamls=args.add_yamls, envs=args.add_envs)

    print(f"[INFO] Using container profile: {ci.profile}")
    if args.command == "start":
        print(f"[INFO] Building the docker image and starting the container {ci.container_name} in the background...")
        x11_yaml, x11_envar = x11_utils.x11_check(ci.statefile)
        ci.add_yamls += x11_yaml
        ci.environ.update(x11_envar)
        ci.start()
    elif args.command == "enter":
        print(f"[INFO] Entering the existing {ci.container_name} container in a bash session...")
        ci.enter()
    elif args.command == "copy":
        print(f"[INFO] Copying artifacts from the 'isaac-lab-{ci.container_name}' container...")
        ci.copy()
        print("\n[INFO] Finished copying the artifacts from the container.")
    elif args.command == "stop":
        print(f"[INFO] Stopping the launched docker container {ci.container_name}...")
        ci.stop()
        x11_utils.x11_cleanup(ci.statefile)
    elif args.command == "config":
        print(f"[INFO] Configuring the passed options into a compose yaml...")
        if not args.output_dir is None:
            output = str(Path(args.output_dir).resolve())
        else:
            output = None
        ci.config(output_dir=output)
    elif args.command == "push":
        if not shutil.which("apptainer"):
            apptainer_utils.install_apptainer()
        if not ci.does_image_exist():
            raise RuntimeError(f"The image '{ci.image_name}' does not exist!")
        apptainer_utils.check_docker_version_compatible()
        cluster_login = ci.dot_vars["CLUSTER_LOGIN"]
        cluster_isaaclab_dir = ci.dot_vars["CLUSTER_ISAACLAB_DIR"]
        cluster_sif_path = ci.dot_vars["CLUSTER_SIF_PATH"]
        exports_dir = ci.context_dir / "exports"
        exports_dir.mkdir(parents=True, exist_ok=True)
        for file in exports_dir.glob(f"{ci.container_name}*"):
            file.unlink()
        subprocess.run(
            [
                "APPTAINER_NOHTTPS=1",
                "apptainer",
                "build",
                "--sandbox",
                "--fakeroot",
                f"{ci.container_name}.sif",
                f"docker-daemon://{ci.image_name}",
            ],
            check=True,
            shell=True,
            cwd=exports_dir,
        )
        subprocess.run(
            ["tar", "-cvf", f"{ci.container_name}.tar", f"{ci.container_name}.sif"],
            check=True,
            cwd=exports_dir,
        )
        subprocess.run(["ssh", cluster_login, f"mkdir -p {cluster_sif_path}"], check=True, cwd=exports_dir)
        subprocess.run(
            [
                "scp",
                f"{ci.container_name}.tar",
                f"{cluster_login}:{cluster_sif_path}/{ci.container_name}.tar",
            ],
            check=True,
            cwd=exports_dir,
        )
    elif args.command == "job":
        cluster_login = ci.dot_vars["CLUSTER_LOGIN"]
        cluster_isaaclab_dir = ci.dot_vars["CLUSTER_ISAACLAB_DIR"]
        apptainer_utils.check_singularity_image_exists(ci)
        subprocess.run(["ssh", cluster_login, f"mkdir -p {cluster_isaaclab_dir}"], check=True)
        print("[INFO] Syncing Isaac Lab code...")
        subprocess.run(
            [
                "rsync",
                "-rh",
                "--exclude",
                "*.git*",
                "--filter=:- .dockerignore",
                f"/{ci.context_dir}/..",
                f"{cluster_login}:{cluster_isaaclab_dir}",
            ],
            check=True,
        )
        print("[INFO] Executing job script...")
        subprocess.run(
            [
                "ssh",
                cluster_login,
                f"cd {cluster_isaaclab_dir} && sbatch {cluster_isaaclab_dir}/docker/cluster/submit_job.sh",
                cluster_isaaclab_dir,
                f"{ci.container_name}",
            ]
            + args.job_args,
            check=True,
        )
    else:
        raise RuntimeError(f"Invalid command provided: {args.command}")


if __name__ == "__main__":
    main()
