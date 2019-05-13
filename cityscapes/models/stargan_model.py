import torch
import itertools
from .base_model import BaseModel
from collections import OrderedDict
from . import networks
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

class StarGANModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        parser.add_argument('--supervised', default=False, help='use supervised setting MMIALI', action="store_true")
        if is_train:
            parser.add_argument('--lambda_cls', type=float, default=1, help='weight for domain classification loss')
            parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')
            parser.add_argument('--lambda_rec', type=float, default=10, help='weight for reconstruction loss')
            parser.add_argument('--n_critic', type=int, default=1, help='number of D updates per each G update')
        return parser
    
    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.iter_times = 0
        self.loss_names = ['d', 'd_cls', 'd_gp', 'g', 'g_rec', 'g_cls']
        if self.isTrain:
            self.model_names = ['G', 'D']
            self.visual_names = ['reals', 'fakes', 'recs']
        else:  # during test time, only load Gs
            self.model_names = ['G']
            self.visual_names = ['reals', 'fakes']

        self.netG = networks.define_G(opt.input_nc + opt.ndomains, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                    not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        if self.isTrain:  # define discriminators
            self.netD = networks.define_CondD(opt.ndomains,opt.ndf,
                    opt.crop_size,  opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionL1 = torch.nn.L1Loss()
            
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
    
    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out
    
    def set_input(self, input):
        if self.opt.ndomains == 3:
            self.reals = [input['A'].to(self.device), input['B'].to(self.device), input['C'].to(self.device)]
        else:
            self.reals = [input['A'].to(self.device), input['B'].to(self.device)]
        self.image_paths = input['A_paths']
        self.labels = []
        self.cvecs = []
        for i in range(self.opt.ndomains):
            a = torch.Tensor([i] * len(input['A'])) #domain labels.
            c = self.label2onehot(a, self.opt.ndomains) # Labels for computing classification loss.
            self.labels.append(a.to(self.device))
            self.cvecs.append(c.to(self.device))
    def getxv(self, x, c):
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)
        return x
    def forward(self):
        self.fakes = {}
        self.recs = []
        keyf = '{}_{}'
        for i in range(self.opt.ndomains):
            for j in range(self.opt.ndomains):
                if j == i:
                    continue
                fake_xj = self.netG(self.getxv(self.reals[i], self.cvecs[j]))
                key = keyf.format(i, j)
                self.fakes[key] = fake_xj
            rec_xi = self.netG(self.getxv(self.reals[i], self.cvecs[i]))
            self.recs.append(rec_xi)
    
    def classification_loss(self, logit, target):
        return F.cross_entropy(logit, target.long())
    
    def backward_G(self):
        lambda_rec = self.opt.lambda_rec
        lambda_cls = self.opt.lambda_cls

        keyf = '{}_{}'
        self.loss_g = []
        self.loss_g_rec = []
        self.loss_g_cls = []
        for i in range(self.opt.ndomains):
            loss_g = 0
            loss_g_cls = 0
            loss_g_rec = 0
            for j in range(self.opt.ndomains):
                if j == i:
                    continue
                key = keyf.format(i, j)
                out_src, out_cls = self.netD(self.fakes[key])
                # loss_g += - torch.mean(out_src)
                loss_g += self.criterionGAN(out_src, True)
                loss_g_cls += self.classification_loss(out_cls, self.labels[j]) * lambda_cls
                if self.opt.supervised:
                    # loss_g_rec += torch.mean(torch.abs(self.reals[j] - self.fakes[key])) * lambda_rec
                    loss_g_rec += self.criterionL1(self.fakes[key], self.reals[j]) * lambda_rec
            #loss_g_rec += torch.mean(torch.abs(self.reals[i] - self.recs[i])) * lambda_rec
            loss_g_rec += self.criterionL1(self.recs[i], self.reals[i]) * lambda_rec
            self.loss_g.append(loss_g)
            self.loss_g_rec.append(loss_g_rec)
            self.loss_g_cls.append(loss_g_cls)
            
        g_loss = 0
        for a, b, c in zip(self.loss_g, self.loss_g_cls, self.loss_g_rec):
            g_loss += a + b + c
        g_loss.backward() 
    
    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)
    
    def backward_D(self):
        lambda_cls = self.opt.lambda_cls
        lambda_gp = self.opt.lambda_gp
        self.loss_d = []
        self.loss_d_gp = []
        self.loss_d_cls = []
        keyf = '{}_{}'
        for i in range(self.opt.ndomains):
            loss_d = 0
            loss_d_cls = 0
            # loss_d_gp = 0
            for j in range(self.opt.ndomains):
                if j == i:
                    continue
                key = keyf.format(i, j)
                out_src, out_cls = self.netD(self.fakes[key].detach())
                # loss_d += torch.mean(out_src)
                loss_d += self.criterionGAN(out_src, False)

                #alpha = torch.rand(self.reals[i].size(0), 1, 1, 1).to(self.device)
                #x_hat = (alpha * self.reals[i].data + (1 - alpha) * self.fakes[key].data).requires_grad_(True)
                #out_src, _ = self.netD(x_hat)
                #loss_d_gp += self.gradient_penalty(out_src, x_hat) * lambda_gp
            
            out_src, out_cls = self.netD(self.reals[i])
            # loss_d += - torch.mean(out_src) * (self.opt.ndomains - 1)
            loss_d += self.criterionGAN(out_src, True) * (self.opt.ndomains - 1)
            loss_d_cls += self.classification_loss(out_cls, self.labels[i]) * lambda_cls
        
            self.loss_d.append(loss_d)
            self.loss_d_cls.append(loss_d_cls)
            # self.loss_d_gp.append(loss_d_gp)
        d_loss = 0
        for a, b in zip(self.loss_d, self.loss_d_cls): #, self.loss_d_gp):
            d_loss += a + b
        d_loss.backward()
    
    def optimize_parameters(self):
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # optimize D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D()      # calculate gradients for D_A
        self.optimizer_D.step()  # update D_A and D_B's weights
        # optimize G
        self.set_requires_grad(self.netD, False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()
        if (self.iter_times + 1) % self.opt.n_critic == 0:
            self.backward_G()             # calculate gradients for G_A and G_B
            self.optimizer_G.step()       # update G_A and G_B's weights

        self.iter_times += 1
    
    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            loss_dict = getattr(self, 'loss_'+name)
            if isinstance(loss_dict, list):
                for k, v in enumerate(loss_dict):
                    errors_ret[name + '_' + str(k)] = float(v)
            elif isinstance(loss_dict, dict):
                for k, v in loss_dict.items():
                    errors_ret[name + '_' + k] = float(v)
            else:
                errors_ret[name] = float(loss_dict)  # float(...) works for both scalar tensor and float number
        return errors_ret

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_dict = getattr(self, name)
                if isinstance(visual_dict, list):
                    for k, v in enumerate(visual_dict):
                        visual_ret[name + '_' + str(k)] = v
                elif isinstance(visual_dict, dict):
                    for k, v in visual_dict.items():
                        visual_ret[name + '_' + k] = v
        return visual_ret
