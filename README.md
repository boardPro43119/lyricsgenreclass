Benjamin Prud'homme
CMPU 366 Final Project
Prof. Gordon
May 27, 2021

Files:

lyricsgenreclass.py: Working application for performing classification of music based on genre. Trains model on full dataset, weighting genres equally

lyricsgenreclassbalancedweights.py: Version of lyricsgenreclass.py that gives genres weights that are inversely proportionate to the relative numbers of songs per genre

lyricsgenreclassequalgenredist.py: Version of lyricsgenreclass.py that trains model on a random sample containing 900 songs per genre, weighting genres equally

tcc_ceds_music.csv: Original dataset by Moura, Fontelles, Sampaio, Franca: "Music Dataset: Lyrics and Metadata from 1950 to 2019." Includes song metadata, lyrics, audio properties, and scores from topic model used by researchers.

tcc_ceds_music_lyricsonly.csv: Data file containing only lyrics and genre properties of songs from above dataset.

tcc_ceds_musics_<genre>: Data files containing only lyrics/genre pairs for songs of given genre only

/confusion_matrices: Directory containing confusion matrices for each of the three versions of lyricsgenreclass.py

/wordclouds: Directory containing wordclouds visualizing most common words in each genre