SLURM-GEN
=========

Small Python script that generates SLURM batch scripts to run job arrays easily.

Usage example:

```bash
./slurm-gen.py -s time=00:00:01 -e OMP_NUM_THREADS=1 -o run.sh -u echorun -- echo [id] [0-9][0-9]
```

This will for example generate the following sbatch script:

```bash
#!/bin/bash -l
#SBATCH --time=00:00:01
#SBATCH --array=0-99%10
#SBATCH --output=echorun_%a.out

# sbatch script generate by sbatch_gen using arguments:
# echo ${v0} ${v1}${v2}

varray0=(${SLURM_ARRAY_TASK_ID})
varray1=(0 1 2 3 4 5 6 7 8 9)
varray2=(0 1 2 3 4 5 6 7 8 9)

r=${SLURM_ARRAY_TASK_ID}
d=$(($r/1))
i0=$(($r - $d*1))
r=$d
v0=${varray0[${i0}]}

d=$(($r/10))
i1=$(($r - $d*10))
r=$d
v1=${varray1[${i1}]}

d=$(($r/10))
i2=$(($r - $d*10))
r=$d
v2=${varray2[${i2}]}

if [ ! -s "echorun_${SLURM_ARRAY_TASK_ID}.out" ] || [ -n "$(grep -l 'srun: error' "echorun_${SLURM_ARRAY_TASK_ID}.out")" ]
then
    OMP_NUM_THREADS=1 srun echo ${v0} ${v1}${v2}
fi
```