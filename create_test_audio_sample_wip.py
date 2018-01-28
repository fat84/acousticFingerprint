# ######################################################
#
# This program extracts a sample from a song.
# This is used to test the acoustic fingerprinting and matching algorithm.
#
#
# By Valentin Todorov
#
# ######################################################


from scipy.io import wavfile

# Location of the sample
song_dir = "/Users/valentin/GoogleDrive/MusicEngine/wav/"
song_dir_output = "/Users/valentin/GoogleDrive/MusicEngine/sample/"

song = "dt16bars102rap.wav"
part_of_song = "middle"


# Starting position of sample (seconds) and length of the sample (in seconds)
sample_start = 5
sample_end = 5
sample_length = sample_start + sample_end

sample_rate, soundtrack_data = wavfile.read(song_dir + song)

wavfile.write(data = soundtrack_data[int(sample_start * sample_rate):int(sample_length * sample_rate)],
              filename = song_dir_output + "sample_" + song + "_" + part_of_song + ".wav",
              rate = sample_rate)
