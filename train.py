
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
import torch.autograd
from torch.utils.tensorboard import SummaryWriter
import time
import yaml
from models.FANet import FANet
from utils import AverageMeter
from data.dataset import SOD360Dataset


def get_1x_lr_params(model):
    b = []
    b.append(model.basenet.conv1)
    b.append(model.basenet.bn1)
    b.append(model.basenet.layer1)
    b.append(model.basenet.layer2)
    b.append(model.basenet.layer3)
    b.append(model.basenet.layer4)
    for i in range(len(b)):
        for j in b[i].modules():
            jj = 0
            for k in j.parameters():
                jj += 1
                if k.requires_grad:
                    yield k


def get_10x_lr_params(model):
    b = []
    b.append(model.FPN.parameters())
    b.append(model.PFFusion.parameters())
    b.append(model.MLFCombination.parameters())


    for j in range(len(b)):
        for i in b[j]:
            yield i


###############################################################################
def adjust_learning_rate(cfg, optimizer, epoch, i_iter, dataset_lenth):
    lr = cfg['lr']*((1-float(epoch*dataset_lenth+i_iter)/(cfg['epochs']*dataset_lenth))**(cfg['power']))
    print('Epoch [{}] Learning rate: {}'.format(epoch, lr))
    optimizer.param_groups[0]['lr'] = lr
    optimizer.param_groups[1]['lr'] = lr * 10

def train(train_loader, model, criterion1, criterion2, optimizer, epoch, out_model_path, log_dir_path, init_iter, cfg):
    assert cfg['use_gpu']
    model.train()

    batch_time = AverageMeter()
    losses = AverageMeter()
    writer = SummaryWriter(log_dir_path)

    for i_iter, batch in enumerate(train_loader):
        ttime = time.time()
        i_iter += init_iter
        adjust_learning_rate(cfg, optimizer, epoch, i_iter, len(train_loader))
        input_list, label_list, _, _ = batch
        equi_img = input_list[0].type(torch.FloatTensor).to(torch.device('cuda', cfg['device_id']))
        cubemap_B = input_list[1].type(torch.FloatTensor).to(torch.device('cuda', cfg['device_id']))
        cubemap_D = input_list[2].type(torch.FloatTensor).to(torch.device('cuda', cfg['device_id']))
        cubemap_F = input_list[3].type(torch.FloatTensor).to(torch.device('cuda', cfg['device_id']))
        cubemap_L = input_list[4].type(torch.FloatTensor).to(torch.device('cuda', cfg['device_id']))
        cubemap_R = input_list[5].type(torch.FloatTensor).to(torch.device('cuda', cfg['device_id']))
        cubemap_T = input_list[6].type(torch.FloatTensor).to(torch.device('cuda', cfg['device_id']))
        equi_label = label_list[0].type(torch.FloatTensor).to(torch.device('cuda', cfg['device_id']))

        pred_list = model([equi_img, cubemap_B, cubemap_D, cubemap_F, cubemap_L, cubemap_R, cubemap_T])
        equi_pred = nn.functional.interpolate(pred_list[0], size=(cfg['equi_input_height'], cfg['equi_input_width']), mode='bilinear', align_corners=True)
        equi_l1pred = nn.functional.interpolate(pred_list[1], size=(cfg['equi_input_height'], cfg['equi_input_width']), mode='bilinear', align_corners=True)
        equi_l2pred = nn.functional.interpolate(pred_list[2], size=(cfg['equi_input_height'], cfg['equi_input_width']), mode='bilinear', align_corners=True)
        equi_l3pred = nn.functional.interpolate(pred_list[3], size=(cfg['equi_input_height'], cfg['equi_input_width']), mode='bilinear', align_corners=True)
        equi_l4pred = nn.functional.interpolate(pred_list[4], size=(cfg['equi_input_height'], cfg['equi_input_width']), mode='bilinear', align_corners=True)

        equi_loss1 = criterion1(equi_pred, equi_label)
        equi_loss2 = criterion2(torch.sigmoid(equi_pred), equi_label)
        equi_loss = equi_loss1 + equi_loss2

        equi_l1loss1 = criterion1(equi_l1pred, equi_label)
        equi_l1loss2 = criterion2(torch.sigmoid(equi_l1pred), equi_label)
        equi_l1loss = equi_l1loss1 + equi_l1loss2

        equi_l2loss1 = criterion1(equi_l2pred, equi_label)
        equi_l2loss2 = criterion2(torch.sigmoid(equi_l2pred), equi_label)
        equi_l2loss = equi_l2loss1 + equi_l2loss2

        equi_l3loss1 = criterion1(equi_l3pred, equi_label)
        equi_l3loss2 = criterion2(torch.sigmoid(equi_l3pred), equi_label)
        equi_l3loss = equi_l3loss1 + equi_l3loss2

        equi_l4loss1 = criterion1(equi_l4pred, equi_label)
        equi_l4loss2 = criterion2(torch.sigmoid(equi_l4pred), equi_label)
        equi_l4loss = equi_l4loss1 + equi_l4loss2

        loss = equi_loss + equi_l1loss + equi_l2loss + equi_l3loss + equi_l4loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time.update((time.time() - ttime) / cfg['summary_freq'])
        losses.update(loss.data.item(), cfg['batch_size'])

        if i_iter % cfg['summary_freq'] == 0:
            print('Epoch: [{0}/{1}][{2}/{3}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                    epoch, cfg['epochs'], i_iter, len(train_loader), batch_time=batch_time,
                    loss=losses))
        writer.add_scalar('Train/Batch_Loss', losses.val, i_iter+epoch*len(train_loader))
        writer.add_scalar('Train/Epoch_Loss', losses.avg, epoch)

        if i_iter % len(train_loader) == (len(train_loader)-1):
            print( 'saving checkpoint: ', os.path.join(out_model_path, '{0}_{1:02}.pth'.format(cfg['model_name'], epoch)))
            torch.save(model.state_dict(), os.path.join(out_model_path,
                        '{0}_{1:02}.pth'.format(cfg['model_name'], epoch)))


def main():
    with open('./config.yaml') as f:
        config = yaml.safe_load(f)
    for key in config.keys():
        print("\t{} : {}".format(key, config[key]))
    print('---------------------------------------------------------------' + '\n')

    log_dir_path = os.path.join(config['logs'], "{0}/{1}".format(config['model_name'], 'train'))
    if not os.path.exists(log_dir_path):
        os.makedirs(log_dir_path)

    out_model_path = os.path.join(config['checkpoints'], "{0}".format(config['model_name']))
    if not os.path.exists(out_model_path):
        os.makedirs(out_model_path)

    train_dataset = SOD360Dataset(config['train_data'], config['train_list'], (config['equi_input_width'], config['equi_input_height']))
    train_loader = data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['processes'], pin_memory=True)

    model = FANet(num_classes=config['num_classes'])
    if config['use_gpu']:
        model = model.to(torch.device('cuda', config['device_id']))

    model_num = []
    init_model_num=0
    if model_num != []:
        model.load_state_dict(torch.load(os.path.join(out_model_path,
                        '{0}_{1:02}_{2:06}.pth'.format(config['model_name'], 0, max(model_num)))))
        init_model_num = max(model_num)

    if config['use_gpu']:
        criterion1 = nn.BCEWithLogitsLoss().to(torch.device('cuda', config['device_id']))
        criterion2 = nn.L1Loss().to(torch.device('cuda', config['device_id']))
    else:
        criterion1 = nn.BCEWithLogitsLoss()
        criterion2 = nn.L1Loss()

    optimizer = torch.optim.SGD([{'params': get_1x_lr_params(model),'lr': config['lr']},{'params': get_10x_lr_params(model),'lr': config['lr'] * 10}],
                                lr=config['lr'], momentum=config['momentum'], weight_decay=config['weight_decay'])

    for epoch in range(config['epochs']):
        train(train_loader, model, criterion1, criterion2, optimizer, epoch, out_model_path, log_dir_path, init_model_num, config)


if __name__ == '__main__':
    main()