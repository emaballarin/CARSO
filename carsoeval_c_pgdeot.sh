#!/usr/bin/bash -li
#SBATCH --job-name=eval_carso_sc_c_pgdeot
#SBATCH --mail-type=FAIL,END
#SBATCH --partition=DGX
#SBATCH --time=1-05:00:00
#SBATCH --nodes=1                   # Nodes
#SBATCH --ntasks-per-node=7         # GPUs per node
#SBATCH --cpus-per-task=24          # Cores per node / GPUs per node
#SBATCH --mem=256G                  # 4 * Cores per node
#SBATCH --gres=gpu:7                # GPUs per node
################################################################################
#
sleep 3
#
export CODEHOME="$HOME/Downloads/"
export MYPYTHON="$HOME/micromamba/envs/nightorch/bin/python"
#
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
#
echo " "
echo "hostname="$(hostname)
echo "WORLD_SIZE="$WORLD_SIZE
echo "OMP_NUM_THREADS="$OMP_NUM_THREADS
echo "MASTER_ADDR="$MASTER_ADDR
echo "MASTER_PORT="$MASTER_PORT
echo " "
#
################################################################################
cd "$CODEHOME/CARSO/src/"
#
echo "-----------------------------------------------------------------------------------------------------------------"
echo " "
echo "START TIME "$(date +'%Y_%m_%d-%H_%M_%S')
echo " "
echo "-----------------------------------------------------------------------------------------------------------------"
srun "$MYPYTHON" -O "$CODEHOME/CARSO/src/eval_c.py" --dist --e2e --pgdeot --batchsize 70
echo "-----------------------------------------------------------------------------------------------------------------"
echo " "
echo "STOP TIME "$(date +'%Y_%m_%d-%H_%M_%S')
echo " "
echo "-----------------------------------------------------------------------------------------------------------------"
#
