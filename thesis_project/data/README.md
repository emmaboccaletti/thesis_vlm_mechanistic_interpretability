# Data
This folder provides access to datasets used in the project.

Currently it contains a symbolic link to the MOMENTS dataset:
MOMENTS -> /scratch-shared/eboccaletti/MOMENTS

Created using: `ln -s /scratch-shared/eboccaletti/MOMENTS MOMENTS`

The dataset itself is stored on the shared scratch filesystem to avoid
duplicating large files in the project repository.

The MOMENTS dataset contains video segments from football matches
annotated with whether they correspond to contextually important moments.

This folder should only contain:
- symbolic links to datasets

Large datasets should **not** be copied into the repository.

## MOMENTS dataset
```
MOMENTS/
   <id>/
       important-moments/
           1/
              Example content:
              IM_1.mp4
              IM_1_v1.json  
              IM_1_v1.wav   
              IM_1_v2.json
              IM_1_v2.wav
              IM_2.mp4
              ...
              IM_8_v2.wav
           2/
              Example content:
              IM_1.mp4
              IM_1_v1.json  
              IM_1_v1.wav   
              IM_1_v2.json
              IM_1_v2.wav
              IM_2.mp4
              ...
              IM_4_v2.wav
       non-important-moments/
           1/
              Example content:
              NIM_1.mp4
              NIM_1_v1.json  
              NIM_1_v1.wav   
              NIM_1_v2.json
              NIM_1_v2.wav
              NIM_2.mp4
              ...
              NIM_9_v2.wav
           2/
              Example content:
              NIM_1.mp4
              NIM_1_v1.json  
              NIM_1_v1.wav   
              NIM_1_v2.json
              NIM_1_v2.wav
              NIM_2.mp4
              ...
              NIM_2_v2.wav
```

Where:

<id> (e.g., 0Glu8uEj) = a specific football game

important-moments = moments that appear in the highlight reel

non-important-moments = moments from the full game that do not appear in highlights

1, 2 = two different video sources of the same game

### (Non)Important Moments
To confirm with Aditya K Surikuchi. What “1” and “2” actually represent?

In the paper we are told: "Mapping the highlights in GOAL with
two publicly available full game video datasets, i.e., SoccerNet and SoccerReplay-1988"

So we know that for the dataset creation, highlight videos are aligned with full match recordings from two datasets. This is done to _localize_ the moment.

So there are two different full-game broadcasts of the same match.
- 1 = full game video source A
- 2 = full game video source B

Both contain the same match, but they are:
- different broadcasts
- different frame rates
- slightly different editing
- slightly different commentary timing

This is why we have folders 1 and 2.

#### Why folder 1 and 2 may contain different numbers of files

Commentary alignment differences

### Different Modalities
The paper explains that, for each moment, they extract:
- video frames
- audio commentary
- transcription text

As such, each moment contains multiple modalities:

| file  | modality |
|-------|-----|
| .mp4 | video clip " |
| .wav	| audio commentary  |
| .json	| transcription / metadata |

The dataset intentionally includes multiple commentary versions.

### What a single moment actually represents

Example: `important-moments/1/IM_3`

means:
- match: 0Glu8uEj
- class: important
- source video: 1
- moment index: 3

Inside you have:

- IM_3.mp4      -> video clip
- IM_3_v1.wav   -> commentary audio
- IM_3_v1.json  -> transcript
- IM_3_v2.wav   -> alternative segment
- IM_3_v2.json