# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 21:14:45 2016

@author: Pranav
"""
 
import winsound as ws

def play_note(note_id):
    note_map = {1:'pianoa1',
                2:'pianoab1',
                3:'pianob1',
                4:'pianoc1',
                5:'pianoc2',
                6:'pianocd1',
                7:'pianocd2',
                8:'pianod1',
                9:'pianod2',
                10:'pianode1',
                11:'pianode2',
                12:'pianoe1',
                13:'pianoe2',
                14:'pianof1',
                15:'pianofg1',
                16:'pianog1',
                17:'pianoga1',
                18:'pianoga1',
                19:'pianoga1',
                20:'pianoga1',
                21:'pianoga1',
                22:'pianoga1',
                23:'pianoga1',
                24:'pianoga1',
                }
    note_id += 1
    if note_id < 1 or note_id > len(note_map):
        print('No note found.')        
        return
    note_name = note_map[note_id]
    filename = 'tones/raw/'+str(note_name)+'.wav'
    print(filename)
    ws.PlaySound(filename, ws.SND_FILENAME)

for i in range(0,17):
    play_note(i)