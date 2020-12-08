import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
import torch.autograd
from torchvision import transforms
from PIL import Image
import time
import yaml
from torch.utils.tensorboard import SummaryWriter
from models.FANet import FANet
from utils import AverageMeter
from data.dataset import SOD360Dataset


tensor2img = transforms.ToPILImage()
def inference(test_loader, model, cfg, result_path):
    assert cfg['use_gpu']

    with torch.no_grad():
        for i_iter, batch in enumerate(test_loader):
            input_list, _, name, img_size = batch
            equi_img = input_list[0].type(torch.FloatTensor).to(torch.device('cuda', cfg['device_id']))
            cubemap_B = input_list[1].type(torch.FloatTensor).to(torch.device('cuda', cfg['device_id']))
            cubemap_D = input_list[2].type(torch.FloatTensor).to(torch.device('cuda', cfg['device_id']))
            cubemap_F = input_list[3].type(torch.FloatTensor).to(torch.device('cuda', cfg['device_id']))
            cubemap_L = input_list[4].type(torch.FloatTensor).to(torch.device('cuda', cfg['device_id']))
            cubemap_R = input_list[5].type(torch.FloatTensor).to(torch.device('cuda', cfg['device_id']))
            cubemap_T = input_list[6].type(torch.FloatTensor).to(torch.device('cuda', cfg['device_id']))

            pred_list = model([equi_img, cubemap_B, cubemap_D, cubemap_F, cubemap_L, cubemap_R, cubemap_T])
            equi_pred = nn.functional.interpolate(torch.sigmoid(pred_list[0]), size=(img_size[1], img_size[0]), mode='bilinear', align_corners=True)
            equi_pred = np.array(tensor2img(equi_pred.data.squeeze(0).cpu())).astype(np.uint8)
            equi_pred = Image.fromarray(equi_pred)

            if not os.path.exists(result_path):
                os.makedirs(result_path)
            equi_pred.save(os.path.join(result_path, name[0] + '.png'))


def main():
    with open('./config.yaml') as f:
        config = yaml.safe_load(f)
    for key in config.keys():
        print("\t{} : {}".format(key, config[key]))
    print('---------------------------------------------------------------' + '\n')

    result_path = os.path.join(config['result'], "{0}/{1}".format(config['testset_name'], config['model_name']))
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    load_model_path = os.path.join(config['checkpoints'], "{0}".format(config['model_name']))
    if not os.path.exists(load_model_path):
        os.makedirs(load_model_path)

    model = FANet(num_classes=config['num_classes'])
    if config['use_gpu']:
        model = model.to(torch.device('cuda', config['device_id']))


    test_dataset = SOD360Dataset(config['test_data'], config['test_list'],
                                 (config['equi_input_width'], config['equi_input_height']), train=False)
    test_loader = data.DataLoader(test_dataset, batch_size=1, shuffle=False,
                                  num_workers=config['processes'], pin_memory=True)

    print('Inferring the data on %d-th iteration model ' % (config['model_id']+1))

    saved_state_dict = torch.load(os.path.join(load_model_path,'{0}_{1:02}.pth'.format(config['model_name'], config['model_id'])))
    model.load_state_dict(saved_state_dict)

    model.eval()

    inference(test_loader, model, config, result_path)



if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print(end - start, 'seconds')