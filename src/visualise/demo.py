import argparse
import os
import sys
import cv2
import numpy as np
import glog as log

# to make run from console for module import
sys.path.append(os.path.abspath('..'))

from main.config import Config
from main.model import Model
from visualise.trimesh_renderer import TrimeshRenderer
from visualise.vis_util import preprocess_image, visualize


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def process(img, do_render, joints_csv, bad_good):
    assert (img is not None)
    original_img, input_img, params = preprocess_image(img, config.ENCODER_INPUT_SHAPE[0])
    result = model.detect(input_img)
    cam = np.squeeze(result['cam'].numpy())[:3]
    vertices = np.squeeze(result['vertices'].numpy())
    joints = np.squeeze(result['kp2d'].numpy())
    joints = ((joints + 1) * 0.5) * params['img_size']
    

    joints = np.reshape(joints, [38, 1])
    
    log.info(joints.shape)

    if do_render == True:
        renderer = TrimeshRenderer()
        visualize(renderer, original_img, params, vertices, cam, joints)

    return joints



# added an argument for a video also, so your can run the model on either a video or an image
# splits the video into frames and runs the model on each frame
# also an argument for whether the video is bad posture or good posture (0 or 1, respectively)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Demo HMR2.0')

    parser.add_argument('--bad0_good1', required = True)
    parser.add_argument('--joints_csv', required = True) 
    parser.add_argument('--render', required=False, default = True)
    parser.add_argument('--image', required=False, default = '')
    parser.add_argument('--video', required=False, default = '') 
    parser.add_argument('--model', required=False, default='base_model', help="model from logs folder")
    parser.add_argument('--setting', required=False, default='paired(joints)', help="setting of the model")
    parser.add_argument('--joint_type', required=False, default='cocoplus', help="<cocoplus|custom>")
    parser.add_argument('--init_toes', required=False, default=False, type=str2bool,
                        help="only set to True when joint_type=cocoplus")




    args = parser.parse_args()
    if args.init_toes:
        assert args.joint_type, "Only init toes when joint type is cocoplus!"



    class DemoConfig(Config):
        BATCH_SIZE = 1
        ENCODER_ONLY = True
        LOG_DIR = os.path.abspath('../../logs/{}/{}'.format(args.setting, args.model))
        INITIALIZE_CUSTOM_REGRESSOR = args.init_toes
        JOINT_TYPE = args.joint_type


    config = DemoConfig()

    # initialize model
    model = Model()


    bad_good = args.bad0_good1

    if args.video != '':
        cap = cv2.VideoCapture(args.video)
        success,image = cap.read()
        joints_video = np.empty([38, 1])
        joints_frame = process(image, args.render, args.joints_csv, bad_good)
        joints_video = np.concatenate((joints_video, joints_frame), axis=0)
        count = 0
        csvfile = open(args.joints_csv, 'a')
        while success:
            success,image = cap.read()
            if success == False:
                break
            #print('Read a new frame: ', success)
            #if count % 10 == 0:
            joints_frame = process(image, args.render, args.joints_csv, bad_good)
            
            row = ''
            for i in range(joints_frame.shape[0]):
                row += '%s'%joints_frame[i, 0] + ','
            row += '%s'%bad_good
            log.info(row)
            csvfile.write(row + '\n')
            count += 1
        csvfile.close()

        #print('---------')
        #print(joints_video)
        #print('==========')
        #print(joints_video.shape)
            
        

    

    

    if args.image != '':
        img = cv2.imread(args.image)
        process(img, args.render, args.joints_csv)