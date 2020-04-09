import time
from random import randint

import torch as pt
from torch.utils.data import DataLoader
import numpy as np
import visdom
from tqdm import tqdm

from option import args
from custom_dataset import MyDataset

from model import edsr.EDSR
from model import other.Fusion
from model import other.Segmentation


DATASET_PATH = '/data/freddie/pytorch/datasets/SALICON/'
USE_GPU = True
EPOCHS = 30
BATCH_SIZE = 24
LR = 1e-3
WEIGHT_DECAY = 0
RESIZE = (256, 320)
TRAINING_DATASET_STAT = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def train_model(epochs):
    if trained:
        print("Loading trained model: {}".format(trained))
        model = Test()
        model.load_state_dict(pt.load(trained).state_dict())
    else:
        model = Test()

    if USE_GPU:
        pt.cuda.set_device(1)
        model.cuda()
    training_data = MyDataset(DATASET_PATH, normalization=TRAINING_DATASET_STAT, validation=False,
                              annotation_type='float', fixation_map=True, resize=RESIZE)

    validation_data = MyDataset(DATASET_PATH, normalization=TRAINING_DATASET_STAT, validation=True,
                                annotation_type='int', fixation_map=True, resize=RESIZE)

    #training_data=My_Dataset(DATASET_PATH, RESIZE=(320,320))
    optimizer = pt.optim.Adam(filter(lambda p: p.requires_grad is True, model.parameters()),
                                  lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = pt.nn.BCELoss()
    vis = visdom.Visdom(port=8098)
    evaluation = EvaluationMethods()
    # used in RNN
    #hidden_state = None
    #hidden_state = [hidden_state]
    best_nss = 0


    for epoch in range(epochs):
        training_data_loader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True,
                                          num_workers=20, drop_last=True, pin_memory=True)
        validation_data_loader = DataLoader(validation_data, batch_size=BATCH_SIZE, shuffle=True,
                                            num_workers=20, drop_last=True, pin_memory=True)
        total_loss = 0
        recent_loss = 0

        for batch_idx, data in tqdm(enumerate(training_data_loader), unit = 'batchs',
                                    total = len(training_data) // BATCH_SIZE):
            model.train()
            input_data = pt.autograd.Variable(data['image'])
            target = pt.autograd.Variable(data['annotation'])
            fixation_map = data['fixation_map'].numpy()
            target = target.view(target.size()[0], 1, target.size()[1], target.size()[2])
            if USE_GPU:
                input_data, target = input_data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(input_data)
            # Original Loss
            #loss = criterion(output, target)
            #####################################
            loss = model.loss_func(output, target)
            ################ TEST ###################
            #loss = loss - evaluation.pytorch_nss(fixation_map, output).sum()/BATCH_SIZE
            ######################################
            recent_loss += loss.data.cpu().numpy()
            loss.backward()
            optimizer.step()

            if batch_idx % 500 == 0:
                total_loss += recent_loss
                _, val_data = next(enumerate(validation_data_loader))
                random = randint(0, BATCH_SIZE - 1)
                val_pic = val_data['image']
                val_pic = pt.autograd.Variable(val_pic)
                # training data * 255, validation data don't * 255
                val_anno = val_data['annotation'][random].numpy()

                if USE_GPU:
                    val_pic = val_pic.cuda()
                model.eval()
                pred_label = np.uint8(model(val_pic).data.cpu()[random].numpy() * 255)
                pred_label = pred_label.reshape(pred_label.shape[1], pred_label.shape[2])
                pred_label = np.array((pred_label, pred_label, pred_label))
                val_anno = np.array((val_anno, val_anno, val_anno))
                vis.images([pred_label, val_anno])
                acc = 0
                print('Epoch: {}, {:.2f}% completed.\nAverageLoss: {}\nRecentLoss: {}\npixel_acc: {:.2f}%'.format(epoch, 100*BATCH_SIZE*batch_idx/len(training_data), total_loss/(batch_idx+1), recent_loss, 100*acc/(256*256)))
                recent_loss = 0
        #pt.save(model, './Models/ML_pool4_false.model')




        #Validation
        if epoch % 100 == 100:
            model.eval()
            log_file = open("./logs/evaluation_logs.txt", "a+")
            print('Validation Phase...')
            #SAUC = 0 #not vectorized
            cc = 0
            nss = 0
            for batch_idx, data in tqdm(enumerate(validation_data_loader), unit = 'batches',
                                    total = len(validation_data) // BATCH_SIZE):
                input_data = pt.autograd.Variable(data['image'])
                annotation_map = data['annotation'].numpy()
                #annotation_map = annotation_map.reshape(annotation_map.shape[1], annotation_map.shape[2])
                fixation_map = data['fixation_map'].numpy()
                #fixation_map = fixation_map.reshape(fixation_map.shape[1], fixation_map.shape[2])
                if USE_GPU:
                    input_data = input_data.cuda()
                pred_label = np.uint8(model(input_data).data.cpu().numpy() * 255)
                pred_label = pred_label.reshape(pred_label.shape[0], pred_label.shape[2], pred_label.shape[3])
                #SAUC += evaluation.SAUC(fixation_map, pred_label, np.ones((256,256)))
                nss += np.sum(evaluation.nss(fixation_map, pred_label))
                cc += np.sum(evaluation.cc(annotation_map, pred_label))
            log_file.write("Epoch: {}, SAUC: {:.3f}, NSS: {:.3f}, CC: {:.3f}\n".format(epoch, 0, nss, cc));
            log_file.close()
            if nss >= best_nss:
                best_nss = nss
                pt.save(model, './Models/play.model')






if __name__ == '__main__':
    train_model(EPOCHS)
