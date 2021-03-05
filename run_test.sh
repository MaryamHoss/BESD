#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --account=def-eplourde
#SBATCH --mem 32G
#SBATCH --cpus-per-task 2
#SBATCH --gres=gpu:p100:1
#SBATCH --mail-user=seyedeh.maryam.hosseini.telgerdi@usherbrooke.ca
#SBATCH --mail-type=END

module load python/3.6
source ~/projects/def-eplourde/hoss3301/denv3/bin/activate
cd ~/projects/def-eplourde/hoss3301/work/TrialsOfNeuralVocalRecon


python main_conv_test.py with sound_len=87552 spike_len=256 fusion_type='_FiLM_v1_orthogonal_skip_unet' input_type='denoising_eeg_FBC' test_type='speaker_independent' exp_folder='2021-02-19--17-44-01--9021-mcp_' testing=True
