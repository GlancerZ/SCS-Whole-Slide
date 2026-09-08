#!/usr/bin/env bash
# Run from tmux on an allocated compute node; never request additional jobs.
set -euo pipefail
scs_target_job="${1:?job id required}"
[[ "$scs_target_job" =~ ^[0-9]+$ ]] || exit 2
cd /home/glancerz/labwork/codex/TLS
while true; do
    scs_target_state="$(squeue -h -j "$scs_target_job" -o %T)"
    case "$scs_target_state" in
        RUNNING)
            python /home/glancerz/.codex/skills/slurm-remote-exec/scripts/slurm_job_exec.py \
                --jobid "$scs_target_job" \
                --env 'source /home/glancerz/labwork/codex/TLS/.venv-scs/bin/activate' \
                --command 'hostname && python --version && bash scripts/scs_parallel_tmux.sh'
            exit $?
            ;;
        PENDING|CONFIGURING) sleep 30 ;;
        *) echo "Target job ${scs_target_job} no longer usable: ${scs_target_state}"; exit 1 ;;
    esac
done
