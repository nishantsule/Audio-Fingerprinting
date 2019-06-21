import numpy as np
import pydub
import scipy.signal 
import skimage.feature 
import datasketch 
from bokeh.io import show
from bokeh.plotting import figure 
import pyaudio
import warnings
import sys
import pickle 

# This class defines an AudioFP object that stores the name a song and its fingerprint. 
# It also contains all the functions used to read/record audio, generate spectrogram, find peaks,
# generate fingerprint, and saving the object to file.

class AudioFP():
    
    # Initializing AudioFP object.
    
    # Creating the AudioFP class object will prompt the user to chose whether they would like to read audio
    # from a file or to record using a microphone, or to open an already saved object. 
    # Enter 'f' to read an audio file, 'r' to record, or 's' to open a saved object. 
    # Entering any other character will cause the program to throw an exception and exit.
    # The user is also prompted to choose whether they want to generate plots. 
    # Enter 'y' to generate plots or 'n' to skip plotting.
    # After these user selections are made, the program automatically reads/records audio, generate
    # a sprectrogram, finds the local peaks in the spectrogram, and generates a fingerprint 
    # or simply reads an existing AudioFP object from file.
    # Finally, if the user chose to read audio from file or record it, they are prompted to choose
    # whether they want to save the object to file. Enter 'y' to save or 'n' to skip.

    def __init__(self):
        self.songname = ''
        self.fingerprint = datasketch.MinHash(num_perm=256)
        self.framerate = []
        if input('Do you want to proceed manually? Enter "y" or "n": ') == 'n':
            self.ask_user()
        else:
            pass
        
    def ask_user(self):
        audio_type = input('Enter "f" to read from audio file, "r" to record audio, or "s" to open saved fingerprint: ')
        if audio_type == 'f':
            filename = input('Enter the filename you want to read (excluding the extension): ')
            self.songname = filename
            if input('Do you want to show all plots? Enter "y" or "n": ') == 'y':
                plot = True
            else:
                plot = False
            channels, self.framerate = self.read_audiofile(plot, filename)
            f, t, sgram = self.generate_spectrogram(plot, channels, self.framerate)
            fp, tp, peaks = self.find_peaks(plot, f, t, sgram)
            self.generate_fingerprint(plot, fp, tp, peaks)
            if input('Do you want to save the fingerprint to file for later use? Enter "y" or "n": ') == 'y':
                print('Saving the fingerprint')
                self.save_fingerprint()
            else:
                print('Not saving anything')
        elif audio_type == 'r':
            filename = input('Enter a name for the recording:')
            self.songname = filename  
            if input('Do you want to show all plots? Enter "y" or "n": ') == 'y':
                plot = True
            else:
                plot = False
            channels, self.framerate = self.record_audiofile(plot)
            f, t, sgram = self.generate_spectrogram(plot, channels, self.framerate)
            fp, tp, peaks = self.find_peaks(plot, f, t, sgram)
            self.generate_fingerprint(plot, fp, tp, peaks)
            if input('Do you want to save the fingerprint to file for later use? Enter "y" or "n": ') == 'y':
                print('Saving the fingerprint')
                self.save_fingerprint()
            else:
                print('Not saving anything')
        elif audio_type == 's':
            objname = input('Enter the filename (excluding the extention) where the fingerprint is saved: ')
            objname = objname + '.pkl'
            with open(objname, 'rb') as inputobj:
                data = pickle.load(inputobj)
                self.songname = data['songname']
                self.fingerprint = data['fingerprint']
                self.framerate = data['framerate']
            if input('Do you want to see the details of the file? Enter "y" or "n": ') == 'y':
                plot = True
                print('Songname: ', self.songname)
                print('Framerate: ', self.framerate)
                print('Audio-fingerprint:')
                print(self.fingerprint.digest())
            else:
                plot = False
        else:
            sys.exit('''Error: Incorrect entry. Enter "f" to read an audio file, 
                     "r" to record, or "s" to open a saved object''')
        
    # Read audio file using pydub and plot signal.
    # The audio file has to be .mp3 format
    def read_audiofile(self, plot, filename):
        songdata = []  # Empty list for holding audio data
        channels = []  # Empty list to hold data from separate channels
        audiofile = pydub.AudioSegment.from_file(filename + '.mp3')
        songdata = np.frombuffer(audiofile._data, np.int16)
        for chn in range(audiofile.channels):
            channels.append(songdata[chn::audiofile.channels])  # separate signal from channels
        framerate = audiofile.frame_rate
        channels = np.sum(channels, axis=0) / len(channels)  # Averaging signal over all channels
        # Plot time signal
        if plot:
            p1 = figure(plot_width=900, plot_height=500, title='Audio Signal', 
                        x_axis_label='Time (s)', y_axis_label='Amplitude (arb. units)')
            time = np.linspace(0, len(channels)/framerate, len(channels))
            p1.line(time[0::1000], channels[0::1000])
            show(p1)
        return channels, framerate
            
    # Record audio file using pyaudio and plot signal
    def record_audiofile(self, plot):
        rec_time = int(input('How long do you want to record? Enter time in seconds: '))
        start_rec = input('Do you want to start recoding? Enter "y" to start:')
        if start_rec=='y':
            chk_size = 8192  # chunk size
            fmt = pyaudio.paInt16  # format of audio 
            chan = 2  # Number of channels 
            samp_rate = 44100  # sampling rate
            framerate = samp_rate
            p = pyaudio.PyAudio()  # Initializing pyaudio object to open audio stream
            astream = p.open(format=fmt, channels=chan, rate=samp_rate,
                             input=True, frames_per_buffer=chk_size)
            songdata = []
            channels = []
            channels = [[] for i in range(chan)]
            for i in range(0, np.int(samp_rate / chk_size * rec_time)):
                songdata = astream.read(chk_size)
                nums = np.fromstring(songdata, dtype=np.int16)
                for c in range(chan):
                    channels[c].extend(nums[c::chan])
            # Close audio stream
            astream.stop_stream()
            astream.close()
            p.terminate()
        else:
            sys.exit('Audio recording did not start. Start over again.')
        channels = np.sum(channels, axis=0) / len(channels)  # Averaging signal over all channels
        # Plot time signal
        if plot:
            p1 = figure(plot_width=900, plot_height=500, title='Audio Signal', 
                        x_axis_label='Time (s)', y_axis_label='Amplitude (arb. units)')
            time = np.linspace(0, len(channels[0])/framerate, len(channels[0]))
            p1.line(time[0::100], channels[0][0::100])
            show(p1)
        return channels, framerate
        
    # Generate and plot spectrogram of audio data
    def generate_spectrogram(self, plot, audiosignal, framerate):
        fs = framerate  # sampling rate
        window = 'hamming'  # window function
        noverlap = int(overlap_ratio * nperseg)  # number of points to overlap
        # generate spectrogram from consecutive FFTs over the defined window
        f, t, sgram = scipy.signal.spectrogram(audiosignal, fs, window, nperseg, noverlap)  
        sgram = 10 * np.log10(sgram)  # transmorm linear output to dB scale 
        sgram[sgram == -np.inf] = 0  # replace infs with zeros
        # Plot Spectrogram
        if plot:
            p2 = figure(plot_width=900, plot_height=500, title='Spectrogram',
                        x_axis_label='Time (s)', y_axis_label='Frequency (Hz)',
                        x_range=(min(t), max(t)), y_range=(min(f), max(f)))
            p2.image([sgram[::2, ::2]], x=min(t), y=min(f), 
                     dw=max(t), dh=max(f), palette='Spectral11')
            show(p2)
        return f, t, sgram
        
    # Find peaks in the spectrogram using image processing
    def find_peaks(self, plot, f, t, sgram):
        coordinates = skimage.feature.peak_local_max(sgram, min_distance=min_peak_sep, indices=True,
                                     threshold_abs=min_peak_amp)
        
        peaks = sgram[coordinates[:, 0], coordinates[:, 1]]
        tp = t[coordinates[:, 1]]
        fp = f[coordinates[:, 0]]
        # Plot the peaks detected on the spectrogram
        if plot:
            p3 = figure(plot_width=900, plot_height=500, title='Spectrogram with Peaks',
                        x_axis_label='Time (s)', y_axis_label='Frequency (Hz)',
                        x_range=(min(t), max(t)), y_range=(min(f), max(f)))
            p3.image([sgram[::2, ::2]], x=min(t), y=min(f), 
                     dw=max(t), dh=max(f), palette='Spectral11')
            p3.scatter(tp, fp)
            show(p3)
        return fp, tp, peaks
        
    # Use the peak data from the spectrogram to generate a string with pairs of 
    # peak frequencies and the time delta between them.
    def generate_fingerprint(self, plot, fp, tp, peaks):
        # Create the data to be used for fingerprinting
        # for each frequency (anchor) find the next few frequencies (targets) and calculate their time deltas
        # the anchor-target frequency pairs and their time deltas will be used to generate the fingerprints
        s = []  # Empty list to contain data for fingerprint
        for i in range(len(peaks)):
            for j in range(1, peak_connectivity):
                if (i + j) < len(peaks):
                    f1 = fp[i]
                    f2 = fp[i + j]
                    t1 = tp[i]
                    t2 = tp[i + j]
                    t_delta = t2 - t1
                    if t_delta >= peak_time_delta_min and t_delta <= peak_time_delta_max:
                        s.append(str(np.rint(f1)) + str(np.rint(f2)) + str(np.rint(t_delta))) 
        for data in s:
            self.fingerprint.update(data.encode('utf8'))
        if plot:
            print('{} audio-fingerprint: '.format(self.songname))
            print(self.fingerprint.digest())
    
    # Save the AudioFP object to file for later use
    def save_fingerprint(self):
        filename = self.songname + '.pkl'
        obj_dict = {'songname': self.songname, 'fingerprint': self.fingerprint, 'framerate': self.framerate}
        with open(filename, 'wb') as output:  # Overwrites any existing file.
            pickle.dump(obj_dict, output, pickle.HIGHEST_PROTOCOL)
