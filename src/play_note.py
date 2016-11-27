# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 21:14:45 2016

@author: Pranav
"""
import pygame

pygame.init()

def get_key_id_map(key_map):
    map_copy = list(key_map)

    map_copy.sort()
    key_id_map = dict()
    for i in range(0,len(map_copy)):
        if map_copy[i] == 0:
            continue
        key_id_map[map_copy[i]] = i+1
    return key_id_map

def play_note(note_id):
    note_map = {
                1:'pianoa1',
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
                18:'pianoa1',
                19:'pianoab1',
                20:'pianob1',
                21:'pianoc1',
                22:'pianoc2',
                23:'pianocd1',
                24:'pianocd2',
    }

    if note_id < 1 or note_id > len(note_map):
        if note_id > len(note_map):
            note_id = note_id % 24
        else:
            print('No note found for id' + str(note_id))
    note_name = note_map[note_id]
    filename = '../tones/raw/'+str(note_name)+'.wav'
#    ws.PlaySound(filename, ws.SND_FILENAME) # Just a test. Winsound works only for Windows and is slower.
    #ps.playsound(filename)
    pygame.mixer.Sound(filename).play()

def play_key(key, key_id_map):
    if key in key_id_map:
        note_id = key_id_map[key]
#        print('Note ID ' + str(note_id))
        play_note(note_id)
    else:
        print('Cannot map key to note. Ignoring key press.')
