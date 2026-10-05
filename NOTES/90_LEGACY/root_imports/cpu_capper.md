

# cancel all your jobs in any QoS
squeue -u $USER -h -o %i | xargs -r scancel
# allocate mem and cpus
srun -p main --qos=asad_base -c 56 --mem=126G --pty bash -lc '
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK \
       MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK NUMEXPR_MAX_THREADS=$SLURM_CPUS_PER_TASK
source ~/anaconda3/etc/profile.d/conda.sh
conda activate fermionic_env
exec bash'