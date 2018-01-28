# ######################################################
#
# Acoustic fingerprinting
# 1) Creates unique acoustic fingerprints of all songs in a database
# 2) Matches a short song sample to the complete particular song in the database
#
#
# By Valentin Todorov
#
# ######################################################


from scipy.fftpack import fft, dct
from scipy.io import wavfile
from scipy.signal import blackman, hanning, hamming
import numpy as np
from scipy import signal
import os
import timeit

# Number of points to use in the FFT
n_fft = 512

# Frame size and overlap between frames for the FFT
frame_size = 0.025
frame_overlap = 0.015

# Create a list with songs. Remove non-wav elements from the list
songs_dir = "/Users/valentin/GoogleDrive/MusicEngine/wav/"
songs_list = os.listdir(songs_dir)

try:
    songs_list.remove(".DS_Store")
except:
    songs_list

# Location of sample
sample_dir = "/Users/valentin/GoogleDrive/MusicEngine/sample/"
sample_name = "sampleDt16bars102rap_middle.wav"


# ######################################################
# Create acoustic fingerprints for all songs in the database
# ######################################################

# Create empty lists for song names with FFT frame and the frequency bands
song_fft_window = []

band_0_25 = []
band_26_50 = []
band_51_75 = []
band_76_100 = []
band_101_125 = []
band_126_150 = []
band_151_175 = []
band_176_200 = []
band_201_225 = []
band_226_250 = []

# start_fingerprinting = timeit.default_timer()

### Loop through all songs in the database for which I want to create a fingerprint
for s in range(0, len(songs_list)):

    print("Creating fingerprint for %s \n" % songs_list[s])

    #### Read in the raw audio data and get the sample rate (in samples/sec)
    sample_rate, soundtrack_data = wavfile.read(songs_dir + songs_list[s])

    # If the audio is stereo (len(soundtrack_data.shape) == 2), take only one of the channels, otherwise if mono use as is
    if len(soundtrack_data.shape) == 2:
        audio_signal = soundtrack_data.T[0]
        # audio_signal = audio_signal[0:int(10 * sample_rate)]       # Just for testing purposes, keep only the first n seconds
    else:
        audio_signal = soundtrack_data
        # audio_signal = audio_signal[0:int(10 * sample_rate)]

    time = np.arange(0, float(audio_signal.shape[0]), 1) / sample_rate

    #### Split the audio data into frames. Fourier Transform needs to be applied over short chunks of the raw audio data
    # Calculate the length of each frame and the step for moving forward the FFT
    frame_stride = round(frame_size - frame_overlap, 3)
    frame_length, frame_step = int(round(frame_size * sample_rate)), int(round(frame_stride * sample_rate))

    # Calculate the total number of frames
    signal_length = len(audio_signal)
    number_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))

    # Pad the raw signal data with zeros to make sure that all frames have equal number of samples
    pad_audio_length = number_frames * frame_step + frame_length  # This number should be very close to the audio signal length. The difference is caused by the rounding in the calculation of number_frames
    zeros_vector = np.zeros((pad_audio_length + signal_length))
    pad_signal = np.append(audio_signal, zeros_vector)

    frames_idx = np.tile(np.arange(0, frame_length), (number_frames, 1)) + \
                 np.tile(np.arange(0, number_frames * frame_step, frame_step), (frame_length, 1)).T
    signal_frames = pad_signal[frames_idx.astype(np.int32, copy=False)]

    #### Create a window function for the Fourier transform. The windows are used for smoothing values of the raw signal in each time frame
    signal_frames *= np.hamming(frame_length)

    #### Calculate FFT (FFT is the implementation of the Discrete Fourier Transformation)
    # The FFT is symmetrical, and by using "np.fft.rfft" we only take the first half automatically. Otherwise, if we use "np.fft.fft" we'll need to take the first half only
    signal_fft_transform = np.fft.rfft(signal_frames, n=n_fft)
    signal_fft_transform_abs = np.absolute(signal_fft_transform)

    # Calculate the power for each frame
    signal_power = ((signal_fft_transform_abs ** 2) / n_fft)

    #### Define the length of each frequency bin by deciding how many bins I need and then chunk the output from FFT by each bin
    # When NFFT = 512, the result has 257 datapoints from which I subtract 7 to round the bins
    npoints = signal_fft_transform_abs[0].shape[0] - 7
    frequency_bins = 10
    points_per_bin = int(npoints / frequency_bins)

    #### Create frequency bins from the indices of all frequencies in the range 0 - 250 (for NFFT = 512) in step of 25
    frames_idx = np.tile(np.arange(0, points_per_bin, 1), (frequency_bins, 1)) + np.tile(
        np.arange(0, npoints, points_per_bin), (points_per_bin, 1)).T

    # Loop through all the frames/windows to which Fourier transform was applied
    for i in range(0, signal_fft_transform_abs.shape[0]):
        # Limit the output from the Fourier transform only to the first 250 points (for NFFT = 512)
        fft_results = signal_fft_transform_abs[i, 0:npoints]

        # Split the results from the Fourier transform into the frequency bins created above
        fft_results_tiled = fft_results[frames_idx]

        # Calculate the maximum power in each bin. This returns a list with the maximum power for each frequency bin in the window frames of the audio signal
        max_power = [max(fft_results_tiled[x]) for x in range(0, frames_idx.shape[0])]

        # Append the maximum power from each frequency band to the appropriate frequency band lists
        band_0_25.append(max_power[0])
        band_26_50.append(max_power[1])
        band_51_75.append(max_power[2])
        band_76_100.append(max_power[3])
        band_101_125.append(max_power[4])
        band_126_150.append(max_power[5])
        band_151_175.append(max_power[6])
        band_176_200.append(max_power[7])
        band_201_225.append(max_power[8])
        band_226_250.append(max_power[9])

        # Create an index which is a combination of song name and Fourier transform frame. This index tracks songs and frame sequence
        # The number of records in this list should equal the number of records in the lists with frquency bands
        fft_window = i  # A sequential number of the Fourier transform windows for each song
        song_fft_window.append(songs_list[s].split(".wav")[0] + "_" + str(fft_window))

# end_fingerprinting = timeit.default_timer()
# print(end_fingerprinting - start_fingerprinting)


# ######################################################
# Create an acoustic fingerprint for the short song sample
# ######################################################

print("Creating fingerprint for the sample song - %s \n" % sample_name)

# Create empty lists for song names with FFT frame and the frequency bands
sample_0_25 = []
sample_26_50 = []
sample_51_75 = []
sample_76_100 = []
sample_101_125 = []
sample_126_150 = []
sample_151_175 = []
sample_176_200 = []
sample_201_225 = []
sample_226_250 = []

#### Read in the raw audio for the sample
sample_rate, soundtrack_data = wavfile.read(sample_dir + sample_name)

# If the audio is stereo (len(soundtrack_data.shape) == 2), take only one of the channels, otherwise if mono use as is
if len(soundtrack_data.shape) == 2:
    audio_signal = soundtrack_data.T[0]
    audio_signal = audio_signal[0:int(2 * sample_rate)]  # keep only the first n seconds
else:
    audio_signal = soundtrack_data
    audio_signal = audio_signal[0:int(2 * sample_rate)]

time = np.arange(0, float(audio_signal.shape[0]), 1) / sample_rate

#### Split the audio data into frames. Fourier Transform needs to be applied over short chunks of the raw audio data
# Calculate the length of each frame and the step for moving forward the FFT
frame_stride = round(frame_size - frame_overlap, 3)
frame_length, frame_step = int(round(frame_size * sample_rate)), int(round(frame_stride * sample_rate))

# Calculate the total number of frames
signal_length = len(audio_signal)
number_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))

# Pad the raw signal data with zeros to make sure that all frames have equal number of samples
pad_audio_length = number_frames * frame_step + frame_length  # This number should be very close to the audio signal length. The difference is caused by the rounding in the calculation of number_frames
zeros_vector = np.zeros((pad_audio_length + signal_length))
pad_signal = np.append(audio_signal, zeros_vector)

frames_idx = np.tile(np.arange(0, frame_length), (number_frames, 1)) + np.tile(
    np.arange(0, number_frames * frame_step, frame_step), (frame_length, 1)).T
signal_frames = pad_signal[frames_idx.astype(np.int32, copy=False)]

#### Create a window function for the Fourier transform. The windows are used for smoothing values of the raw signal in each time frame
signal_frames *= np.hamming(frame_length)

#### Calculate FFT (FFT is the implementation of the Discrete Fourier Transformation)
# The FFT is symmetrical, and by using "np.fft.rfft" we only take the first half automatically. Otherwise, if we use "np.fft.fft" we'll need to take the first half only
signal_fft_transform = np.fft.rfft(signal_frames, n=n_fft)
signal_fft_transform_abs = np.absolute(signal_fft_transform)

# Calculate the power for each frame
signal_power = ((signal_fft_transform_abs ** 2) / n_fft)

#### Define the length of each frequency bin by deciding how many bins I need and then chunk the output from FFT by each bin
# When NFFT = 512, the result has 257 datapoints from which I subtract 7 to round the bins
npoints = signal_fft_transform_abs[0].shape[0] - 7
frequency_bins = 10
points_per_bin = int(npoints / frequency_bins)

#### Create frequency bins from the indices of all frequencies in the range 0 - 250 (for NFFT = 512) in step of 25
frames_idx = np.tile(np.arange(0, points_per_bin, 1), (frequency_bins, 1)) + \
             np.tile(np.arange(0, npoints, points_per_bin), (points_per_bin, 1)).T

# Loop through all the frames/windows to which Fourier transform was applied
for i in range(0, signal_fft_transform_abs.shape[0]):
    # Limit the output from the Fourier transform only to the first 250 points (for NFFT = 512)
    fft_results = signal_fft_transform_abs[i, 0:npoints]

    # Split the results from the Fourier transform into the frequency bins created above
    fft_results_tiled = fft_results[frames_idx]

    # Calculate the maximum power in each bin. This returns a list with the maximum power for each frequency bin in the window frames of the audio signal
    max_power = [max(fft_results_tiled[x]) for x in range(0, frames_idx.shape[0])]

    # Append the maximum power from each frequency band to the appropriate frequency band lists
    sample_0_25.append(max_power[0])
    sample_26_50.append(max_power[1])
    sample_51_75.append(max_power[2])
    sample_76_100.append(max_power[3])
    sample_101_125.append(max_power[4])
    sample_126_150.append(max_power[5])
    sample_151_175.append(max_power[6])
    sample_176_200.append(max_power[7])
    sample_201_225.append(max_power[8])
    sample_226_250.append(max_power[9])

# ######################################################
# Match the fingerprint of the sample to the database and return the results
# Determine which match is the actual
# ######################################################

print("Match the sample song (%s) to the database with fingerprints \n" % sample_name)

import operator
from collections import Counter
import pandas as pd

database_songs_list = [x.split(".wav")[0] for x in os.listdir(songs_dir)]
final_match_df = pd.DataFrame(data=database_songs_list, columns=["song"])

sample_0_25 = set(sample_0_25)
sample_26_50 = set(sample_26_50)
sample_51_75 = set(sample_51_75)
sample_76_100 = set(sample_76_100)
sample_101_125 = set(sample_101_125)
sample_126_150 = set(sample_126_150)
sample_151_175 = set(sample_151_175)
sample_176_200 = set(sample_176_200)
sample_201_225 = set(sample_201_225)
sample_226_250 = set(sample_226_250)

sample_freq_bands_idx = [sample_0_25, sample_26_50, sample_51_75, sample_76_100, sample_101_125, sample_126_150,
                         sample_151_175, sample_176_200, sample_201_225, sample_226_250]
database_freq_bands_idx = [band_0_25, band_26_50, band_51_75, band_76_100, band_101_125, band_126_150, band_151_175,
                           band_176_200, band_201_225, band_226_250]
database_freq_bands_list = ["band_0_25", "band_26_50", "band_51_75", "band_76_100", "band_101_125", "band_126_150",
                            "band_151_175", "band_176_200", "band_201_225", "band_226_250"]

for j in range(0, len(sample_freq_bands_idx)):
    match_idx = [i for i, item in enumerate(database_freq_bands_idx[j]) if item in sample_freq_bands_idx[j]]
    song_match_list = [song_fft_window[i].split("_")[0] for i in match_idx]

    count_match_occurences = dict(Counter(song_match_list))
    sorted_match_occurences = sorted(count_match_occurences.items(), key=lambda x: x[1], reverse=True)

    summary_match_df = pd.DataFrame.from_records(sorted_match_occurences, columns=["song", database_freq_bands_list[j]])
    final_match_df = pd.merge(final_match_df, summary_match_df, on="song")

# Print out the index of songs that are a match
print((final_match_df.sum(axis=1) / 1980) * 100)
final_match_df
