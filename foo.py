#print how many files are in each subfolder in data folder "video"
import os
data_path = "video"
video_1 = [1439, 1060, 987, 3486]
video_2 = [656, 1457, 1583, 3696]
video_3 = [826, 505, 804, 2135]
video_4 = [828, 669, 260, 1757]
video_5 = [527, 176, 808, 1511]
video_6 = [1434, 1225, 872, 3531]
video_7 = [1162, 967, 878, 3007]
video_8 = [1007, 1706, 1492, 4205]
video_9 = [504, 705, 831, 2040]
ref_table = [None, video_1, video_2, video_3, video_4, video_5, video_6, video_7, video_8, video_9]
for subdir in os.listdir(data_path):
    subdir_path = os.path.join(data_path, subdir)
    if os.path.isdir(subdir_path):
        subfiles = os.listdir(subdir_path)
        _sub = {}
        ref = ref_table[int(subdir_path.split("/")[-1][6])]
        for ss in subfiles:
            if ss[0] == ".":
                continue
            i = int(ss[0])
            ss_path = os.path.join(subdir_path, ss)
            if os.path.isdir(ss_path):
                if (ref[i] == len(os.listdir(ss_path))):
                    pass
                else:
                    print(f"Error: {ss_path} {_sub[ss]} has {len(os.listdir(ss_path))} files, but should have {ref[i]} files")
                
        # print(f"{subdir}, files: {_sub}")
