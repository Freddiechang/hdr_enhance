#LC: 这个是单纯对空域利用GAN训练？
from random import randint
import torch as pt
from torch.utils.data import DataLoader
import numpy as np
import visdom
from tqdm import tqdm
from custom_dataset import MyDataset
from models import Discriminator, Test
from evaluation import EvaluationMethods

#dataset_path = '/data/freddie/FCN.tensorflow-master/Data_zoo/MIT_SceneParsing/ADEChallengeData2016'
DATASET_PATH = '/data/freddie/pytorch/datasets/SALICON/'
USE_GPU = True
EPOCHS = 30
BATCH_SIZE = 24
LR = 1e-3
WEIGHT_DECAY = 0
RESIZE = (256, 320)
LAMBDA = 0.5
#TRAINING_DATASET_STAT = ([137.36534, 114.96422, 102.78216],[68.89384, 65.25822, 61.49445])
# mean and std from pre-trained vgg16, more info on github/torchvision
TRAINING_DATASET_STAT = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

def calc_gradient_penalty(net, real_data, fake_data):
    '''
    real_data, fake_data are output of G or ground truth from the dataset
    should all be tensors, use .data to extract tensors from variables
    '''
    alpha = pt.rand(real_data.size()[0], 1, 1, 1)
    if USE_GPU:
        alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if USE_GPU:
        interpolates = interpolates.cuda()
    interpolates = pt.autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = net(interpolates)
    if USE_GPU:
        tmp = pt.ones(disc_interpolates.size()).cuda()
    else:
        tmp = pt.ones(disc_interpolates.size())

    gradients = pt.autograd.grad(outputs=disc_interpolates, inputs=interpolates, grad_outputs=tmp, create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.view(gradients.size()[0], -1).norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

#LC:如果之前有训练过的，或者训练过一半的，就把 trained=“设置成模型的路径”
def train_model(epochs, trained=False):
    if trained:
        G = Test()
        G.load_state_dict(pt.load(trained).state_dict())
        D = Discriminator(RESIZE, BATCH_SIZE)
    else:
        G = Test()
        D = Discriminator(RESIZE, BATCH_SIZE)
        #G.load_state_dict(pt.load('./Models/SALICON_BN_adadelta.model').state_dict())

    if USE_GPU:
        G.cuda()
        D.cuda()
    training_data = MyDataset(DATASET_PATH, normalization=TRAINING_DATASET_STAT, validation=False, annotation_type='float', resize=RESIZE)
    #LC:validation利用fixationmap计算NSS
    validation_data = MyDataset(DATASET_PATH, normalization=TRAINING_DATASET_STAT, validation=True, annotation_type='int', fixation_map=True, resize=RESIZE)

    #training_data=My_Dataset(DATASET_PATH, resize=(320,320))
    G_optimizer = pt.optim.Adam(filter(lambda p: p.requires_grad is True, G.parameters()), lr=LR)
    D_optimizer = pt.optim.Adam(filter(lambda p: p.requires_grad is True, D.parameters()), lr=LR)

    vis = visdom.Visdom()
    criterion = pt.nn.BCELoss()
    evaluation = EvaluationMethods()

    for epoch in range(epochs):
        D_loss = 0
        G_loss = 0
        W_Distance = 0
        BCE_loss = 0
        best_nss = 0
        training_data_loader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, drop_last=True)
        validation_data_loader = DataLoader(validation_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, drop_last=True)
        for batch_idx, data in tqdm(enumerate(training_data_loader), unit='batches',
                                    total=len(training_data) // BATCH_SIZE):
            input_data = pt.autograd.Variable(data['image'])
            target = pt.autograd.Variable(data['annotation'])
            target = target.resize(target.size()[0], 1, target.size()[1], target.size()[2])
            if USE_GPU:
                input_data, target = input_data.cuda(), target.cuda()

            if batch_idx % 2 == 0:
                #update D
                G.eval()
                D.train()
                for p in D.parameters():  # reset requires_grad
                    p.requires_grad = True  # they are set to False below in netG update
                D_optimizer.zero_grad()
                #train D with real
                D_real = D(pt.cat((input_data, target), 1))
                D_real = -D_real.mean()
                #D_real.backward(mone)
                #train D with fake
                G_sample = G(pt.autograd.Variable(data['image']).cuda()).detach()
                D_fake = D(pt.cat((input_data, G_sample), 1))
                D_fake = D_fake.mean()
                #D_fake.backward(one)
                # train with gradient penalty
                gradient_penalty = calc_gradient_penalty(D, pt.cat((input_data, target), 1).data, pt.cat((input_data, G_sample), 1).data)
                #gradient_penalty.backward(one)
                #for p in D.parameters():
                    #p.data.clamp_(-0.05, 0.05)
                #cost
                D_cost = D_fake + D_real + gradient_penalty
                #print("D_f: {}, d_r: {}, d_g-p: {}".format(D_fake.data.cpu().numpy()[0],D_real.data.cpu().numpy()[0],gradient_penalty.data.cpu().numpy()[0]))
                D_cost.backward()
                #D_cost = D_fake - D_real
                Wasserstein_D = D_real - D_fake
                D_optimizer.step()
                
                D_loss += D_cost.data.cpu().numpy()
                W_Distance += Wasserstein_D.data.cpu().numpy()
            else:
                for p in D.parameters():
                    p.requires_grad = False
                D.eval()
                G.train()
                G_optimizer.zero_grad()           
                G_sample = G(input_data)
                D_fake = D(pt.cat((input_data, G_sample), 1))
                D_fake = -D_fake.mean()
                #BCE_loss = criterion(G_sample, target)
                BCE_loss = G.loss_func(G_sample, target)
                #D_fake.backward(mone)
                #LC:这里的权重可以自己改
                G_cost = 0.7 * D_fake + 0.3 * BCE_loss
                G_cost.backward()
                G_optimizer.step()
                G_loss += G_cost.data.cpu().numpy()






            if batch_idx % 500 == 0:
                _, val_data = next(enumerate(validation_data_loader))
                random = randint(0, BATCH_SIZE - 1)
                val_pic = val_data['image']
                val_pic = pt.autograd.Variable(val_pic)
                # training data * 255, validation data don't * 255
                val_anno = val_data['annotation'][random].numpy()

                if USE_GPU:
                    val_pic = val_pic.cuda()
                G.eval()
                pred_label = np.uint8(G(val_pic).data.cpu()[random].numpy() * 255)
                pred_label = pred_label.reshape(pred_label.shape[1], pred_label.shape[2])
                pred_label = np.array((pred_label, pred_label, pred_label))
                val_anno = np.array((val_anno, val_anno, val_anno))
                vis.images([pred_label, val_anno])
                print('Epoch: {}, {:.2f}% completed.\nD_Loss: {}\nW_D: {}\nG_loss: {}'.format(epoch, 100*BATCH_SIZE*batch_idx/len(training_data), D_loss, W_Distance, G_loss))
                D_loss = 0
                G_loss = 0
                W_Distance = 0


        #Validation
        if epoch % 1 == 0:
            G.eval()
            log_file = open("./logs/evaluation_logs_gan_SALICON.txt", "a+")
            print('Validation Phase...')
            sauc = 0
            cc = 0
            nss = 0
            for batch_idx, data in tqdm(enumerate(validation_data_loader), unit='batches',
                                        total=len(validation_data) // BATCH_SIZE):
                input_data = pt.autograd.Variable(data['image'])
                annotation_map = data['annotation'].numpy()
                fixation_map = data['fixation_map'].numpy()
                if USE_GPU:
                    input_data = input_data.cuda()
                pred_label = np.uint8(G(input_data).data.cpu().numpy() * 255)
                pred_label = pred_label.reshape(pred_label.shape[0], pred_label.shape[2], pred_label.shape[3])
                #SAUC += evaluation.SAUC(fixation_map, pred_label, np.ones((256,256)))
                nss += np.sum(evaluation.nss(fixation_map, pred_label))
                cc += np.sum(evaluation.cc(annotation_map, pred_label))
            log_file.write("Epoch: {}, SAUC: {:.3f}, NSS: {:.3f}, CC: {:.3f}\n".format(epoch, sauc, nss, cc));
            log_file.close()
            if nss >= best_nss:
                best_nss = nss
                pt.save(G, './Models/SALICON_GAN_G.model')
                pt.save(D, './Models/SALICON_GAN_D.model')


if __name__ == '__main__':
    train_model(EPOCHS)
