import time, os
import json
import mmcv 
from mmdet.apis import init_detector, inference_detector

def main():

    config_file = '/root/mmdetection/configs/hrnet/faster_rcnn_hrnetv2p_w18_1x_train-all-1400_test-1400_pos2neg.py' 
    checkpoint_file = '/root/mmdetection/work_dirs/faster_rcnn_hrnetv2p_w18_1x_train-all-1400_test-1400_pos2neg/latest.pth'
    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    with open('/root/mmdetection/tools/T_dir_neg.txt') as f:

        for line in f:

            line = line.strip('\n')

            test_path = '/root/docker_mounts_sata/pap_work/trainset/neg/2000x2000_v20191119/' + line 
            # test_path = '/root/docker_mounts_sata/pap_work/testset/test_jpg/2000x2000/T2019_12'  # test img path
            
            # such as 'T2019_600.json'
            json_name = line + '.json'

            img_list = []
            for img_name in os.listdir(test_path):
                if img_name.endswith('.jpg'):
                    img_list.append(img_name)

            result = []
            #cnt = 0
            
            for i, img_name in enumerate(img_list, 1):          # each image
                full_img = os.path.join(test_path, img_name)
                predict = inference_detector(model, full_img)

                #cnt = cnt + 1
                #print('now proceesing : ', line, str(cnt), end='\n')
                
                for i, bboxes in enumerate(predict, 1):         # 
                    if len(bboxes)>0:
                        
                        image_name = img_name                   #  'T2019_12_3_8.jpg'
                        s0 = image_name.split('.')[0]           #  'T2019_12_3_8'
                        grid_x = int(s0.split('_')[2])
                        grid_y = int(s0.split('_')[3])

                        for bbox in bboxes:                    # each box
                            x1, y1, x2, y2, score = bbox.tolist()
                            x = grid_x * 2000 + int(x1)
                            y = grid_y * 2000 + int(y1)
                            w = int(x2 - x1)
                            h = int(y2 - y1)
                            score = round(score, 5)       # as  0.12345
                            result.append({'x': x, 'y': y, 'w': w, 'h': h, 'p': score})

            with open(json_name,'w') as fp:
                # json.dump(result, fp, indent = 4, separators=(',', ': '))
                json.dump(result, fp)

if __name__ == "__main__":
    main()
