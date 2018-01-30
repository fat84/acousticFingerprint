# ######################################################
#
# MP3 to WAV
# This program converts MP3 files to to WAV format
#
#
# By Valentin Todorov
#
# ######################################################


import os
import time
import pydub

# Location of songs to convert and output directory
files_location_in = "/Volumes/EXTERNAL/emoMusic/full_songs/mp3/"
files_location_out = "/Volumes/EXTERNAL/emoMusic/full_songs/wav/"
converted_mp3_files = []

# Create a list with all the MP3 files in the directory and convert from MP3 to WAV
# This will convert only the MP3 files that are not yet converted
mp3_songs = [x.split(".mp3")[0] for x in os.listdir(files_location_in)]
converted_mp3_files_idx = [x.split(".wav")[0] for x in os.listdir(files_location_out)]
song_id_list = [(x + ".mp3") for x in (list(set(mp3_songs) - set(converted_mp3_files_idx)))]

song_counter = len(song_id_list) - 1

print("Starting the conversion from MP3 to WAV...\n")
time.sleep(1)


# Convert MP3 to WAV
for song_id in song_id_list:
    if song_id.find("mp3") > 0:
        try:
            print("\nConvert song: %s" % song_id)
            sound = pydub.AudioSegment.from_mp3(files_location_in + song_id)
            sound.export(files_location_out + song_id.split(".mp3")[0] + ".wav", format = "wav", bitrate = None)
            converted_mp3_files.append(song_id)
        except:
            print("Cannot convert %s to WAV format" % song_id)
        
    print("%s more songs to convert" % song_counter)
    song_counter -= 1

print("\nThe conversion for all songs has completed")
