import os, sys      
import json

# with open('/home/admin/mmdetection/submit/submit_filelist.txt') as f:

with open(sys.argv[1], 'r') as f1:
    dict_1 = json.load(f1)
    list_img_1 = dict_1["images"]
    list_anno_1 = dict_1["annotations"]
with open(sys.argv[2], 'r') as f2:
    dict_2 = json.load(f2)
    list_img_2 = dict_2["images"]
    list_anno_2 = dict_2["annotations"]

list_img_3 = list_img_1 + list_img_2
list_anno_3 = list_anno_1 + list_anno_2

dict_3 = {"images":list_img_3, "annotations":list_anno_3}

with open(sys.argv[3], 'w') as f3:
    json.dump(dict_3, f3)
print("Done............")