import torch
import itertools
from util.image_pool import ImageVectorPool, ImagePool
from .base_model import BaseModel
from collections import OrderedDict
from . import networks
import random

STD = 0.1
class MMIALIModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        parser.add_argument('--supervised', default=False, help='use supervised setting MMIALI', action="store_true")
        if is_train:
            parser.add_argument('--lambda_ali', type=float, default=0.5, help='weight for ALI loss, 1-lambda_ali == weight for DMAE loss')
            parser.add_argument('--lambda_cyc', type=float, default=10, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_in', type=float, default=1.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_semi', type=float, default=1.0, help='weight for cycle loss (B -> A -> B)')
        return parser
    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        if self.isTrain and opt.lambda_ali > 0:
            self.loss_names = ['dmae', 'g', 'dsample', 'd', 'con']
        else:
            self.loss_names = ['dmae', 'd', 'con']

        if self.isTrain and opt.lambda_in > 0:
            self.loss_names.append('incyc')
        if self.isTrain and opt.lambda_semi > 0:
            self.loss_names.append('semicyc')

        self.netGEs = []
        self.netGDs = []

        self.visual_names = ['reals', 'fakes']
        if self.isTrain and opt.lambda_ali > 0:
            self.visual_names.append('samples')
        if self.isTrain and not self.opt.supervised:
            self.visual_names.append('recs')

        self.model_names = []
        self.netGEs = []
        self.netGDs = []
        for i in range(opt.ndomains):
            self.netGEs.append(networks.define_GEn(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids))
            setattr(self, 'netGE_'+str(i), self.netGEs[-1])

            self.netGDs.append(networks.define_GDn(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids))
            setattr(self, 'netGD_'+str(i), self.netGDs[-1])

            self.model_names.extend(['GE_'+str(i), 'GD_'+str(i)])
            
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.netDs = []
            self.fake_pools = []
            self.z_pool = ImagePool(opt.pool_size)
            for i in range(opt.ndomains):
                suf = str(i)
                self.model_names.append('D_'+suf)
                self.netDs.append(networks.define_MMID(opt.output_nc, opt.ndf * 4, opt.ndf, opt.netD,
                                opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids))
                setattr(self, 'netD_'+suf, self.netDs[-1])
                self.fake_pools.append(ImageVectorPool(opt.pool_size))
                #self.z_pools.append(ImagePool(opt.pool_size)) # save latent vector
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss().to(self.device)

            G_param = [a.parameters() for a in self.netGEs] + [a.parameters() for a in self.netGDs]
            # filter(lambda p: p.requires_grad, net.parameters())
            G_param = filter(lambda p: p.requires_grad, itertools.chain(*G_param))
            self.optimizer_G = torch.optim.Adam(G_param, lr=opt.lr, betas=(opt.beta1, 0.999))
            D_param = filter(lambda p: p.requires_grad, itertools.chain(*[a.parameters() for a in self.netDs]))
            self.optimizer_D = torch.optim.Adam(D_param, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        #AtoB = self.opt.direction == 'AtoB'
        #self.real_A = input['A' if AtoB else 'B'].to(self.device)
        #self.real_B = input['B' if AtoB else 'A'].to(self.device)
        #self.image_paths = input['A_paths' if AtoB else 'B_paths']
        if self.opt.ndomains == 3:
            self.reals = [input['A'].to(self.device), input['B'].to(self.device), input['C'].to(self.device)]
        else:
            self.reals = [input['A'].to(self.device), input['B'].to(self.device)]
        self.image_paths = input['A_paths']
        
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fakes = {}
        self.recs = {}
        self.samples = []
        self.zis = []
        keyf = '{}_{}'
        for i in range(self.opt.ndomains):
            zi = self.netGEs[i](self.reals[i])
            self.zis.append(zi)
            for j in range(self.opt.ndomains):
                if j == i:
                    continue
                fake_xj = self.netGDs[j](zi)
                key = keyf.format(i, j)
                self.fakes[key] = fake_xj
                if self.isTrain and not self.opt.supervised:
                    rec_xi = self.netGDs[i](self.netGEs[j](fake_xj))
                    self.recs[key] = rec_xi
        
        if self.isTrain:
            t = torch.FloatTensor(self.zis[0].shape).zero_().to(self.device)
            d = torch.FloatTensor(self.zis[0].shape).fill_(STD).to(self.device)
            m = torch.distributions.Normal(t,d)

            rndi = random.randint(0, len(self.zis) - 1)
            zi = self.zis[rndi].detach()
            zi = self.z_pool.query(zi)
            self.z_sample = m.sample()
            self.z_sample += zi

            #zi = torch.cat(self.zis).detach()
            #self.z_sample += torch.mean(zi, 0, keepdim = True).expand(self.z_sample.shape)
            # print(self.z_sample.shape) # [b, 256, 64, 64]
            for i in range(self.opt.ndomains):
                self.samples.append(self.netGDs[i](self.z_sample))
        
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad(self.netDs, False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad(self.netDs, True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.loss_dsample = 0.0
        self.loss_D = 0.0
        self.loss_d = 0.0
        lambda_ali = self.opt.lambda_ali
        for i, D in enumerate(self.netDs):
            for j in range(self.opt.ndomains):
                if i == j:
                    continue
                fake_i = self.fakes['{}_{}'.format(j, i)]
                fake_img, self.zis[j] = self.fake_pools[i].query(fake_i, self.zis[j])
                self.loss_d += self.backward_D_basic(D, self.reals[i],
                        self.zis[i], fake_img, self.zis[j]) * (1 - lambda_ali)
            
            fake_img, self.z_sample = self.fake_pools[i].query(self.samples[i], self.z_sample) # directly use self.samples[i] ?
            self.loss_dsample += self.backward_D_basic(D, self.reals[i], self.zis[i], fake_img, self.z_sample) * lambda_ali
        self.loss_D = self.loss_dsample + self.loss_d
        self.loss_D.backward()
        self.optimizer_D.step()  # update D_A and D_B's weights
    
    def backward_D_basic(self, netD, real, z1, fake, z2):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real, z1.detach())
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach(), z2.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # loss_D.backward()
        return loss_D
    
    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_ali = self.opt.lambda_ali
        lambda_semi = self.opt.lambda_semi
        lambda_cyc = self.opt.lambda_cyc
        lambda_in = self.opt.lambda_in
        keyf = '{}_{}'
        self.loss_con = {}
        # Consistency loss / outer cycle loss
        if self.opt.supervised:
            # consistency loss
            for i in range(self.opt.ndomains):
                self.loss_con[str(i)] = 0.0
                for j in range(self.opt.ndomains):
                    if i == j:
                        continue
                    key = keyf.format(j, i) # from_j_fake_i generate i !
                    self.loss_con[str(i)] += self.criterionCycle(self.fakes[key], self.reals[i]) * lambda_cyc
        else:
            for i in range(self.opt.ndomains):
                self.loss_con[str(i)] = 0.0
                for j in range(self.opt.ndomains):
                    if i == j:
                        continue
                    key = keyf.format(i, j) # from_i_rec_j reconstruct i !
                    self.loss_con[str(i)] += self.criterionCycle(self.recs[key], self.reals[i]) * lambda_cyc

        # GAN loss
        self.loss_dmae = {}
        self.loss_g = []
        self.loss_incyc = []
        self.loss_semicyc = {}
        for i in range(self.opt.ndomains):
            self.loss_dmae[str(i)] = 0.0
            self.loss_semicyc[str(i)] = 0.0
            for j in range(self.opt.ndomains):
                if i == j:
                    continue
                key = keyf.format(i, j) # from i to j
                self.loss_dmae[str(i)] += self.criterionGAN(self.netDs[j](self.fakes[key], self.zis[i]), True) * (1-lambda_ali)# also backward from D

                tmpfake = self.netGDs[i](self.netGEs[j](self.samples[j]))
                self.loss_semicyc[str(i)] += self.criterionCycle(tmpfake, self.samples[i].detach()) * lambda_semi

            self.loss_g.append(self.criterionGAN(self.netDs[i](self.samples[i], self.z_sample), True)* lambda_ali)  # also backward from D
            self.loss_incyc.append(self.criterionCycle(self.netGEs[i](self.samples[i]), self.z_sample) * (self.opt.ndomains - 1) * lambda_in)
        
        # TODO: add inner loss and semi loss
        # combined loss and calculate gradients
        self.loss_R = 0.0
        for k, v in self.loss_con.items():
            self.loss_R += v
        for v in self.loss_incyc:
            self.loss_R += v
        for k, v in self.loss_semicyc.items():
            self.loss_R += v
        
        self.loss_DMAE = 0.0
        for k, v in self.loss_dmae.items():
            self.loss_DMAE += v
        self.loss_sample = 0.0
        for v in self.loss_g:
            self.loss_sample += v
        
        self.loss_G = self.loss_sample + self.loss_DMAE + self.loss_R
        self.loss_G.backward()
    
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
