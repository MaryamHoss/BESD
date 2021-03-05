#!/bin/bash
#SBATCH --time=15:00:00
#SBATCH --account=def-eplourde
#SBATCH --mem 32G
#SBATCH --cpus-per-task 4
#SBATCH --gres=gpu:p100:1
#SBATCH --mail-user=seyedeh.maryam.hosseini.telgerdi@usherbrooke.ca
#SBATCH --mail-type=END

source ~/projects/def-eplourde/hoss3301/denv2/bin/activate
cd ~/projects/def-eplourde/hoss3301/work/TrialsOfNeuralVocalRecon

module load python/3.6
python main_conv_prediction.py with fusion_type='_FiLM_v1' sound_len=87552 spike_len=256 input_type='denoising_eeg_' epochs=60 batch_size=16