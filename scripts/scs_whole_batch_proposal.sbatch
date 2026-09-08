#!/usr/bin/env bash
# Draft for review. Not submitted: the user's allocation-only convention
# requires authorization before this batch entrypoint is used.
#SBATCH --account=rrg-junding
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bowen.zhao@mail.mcgill.ca
#SBATCH --job-name=ST19-SCS

set -euo pipefail
workspace=/home/glancerz/labwork/codex/TLS
cd "$workspace"
source "$workspace/.venv-scs/bin/activate"
bash "$workspace/scripts/scs_whole_tmux.sh"
