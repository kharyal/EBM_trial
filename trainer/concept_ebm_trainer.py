'''
data is of type:
([[x,y,colour], ...points], [observer_x, observer_y], [[<concept object>, <concept subject>], ...labels])

this file trains the concept Energy Based Model (EBM) on the data
'''

import os
import numpy as np
import torch
from models.concept_EBM import ConceptEBM, ConceptEBMMLP
from buffers.buffer import Buffer
import tqdm
from data_generator.random_points_data_generator import RandomPointsDataGenerator, COLOURS
import matplotlib.pyplot as plt
from copy import deepcopy

class ConceptEBMTrainer:
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.model = ConceptEBMMLP()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.args = args
        self.buffer = Buffer(args.buffer_size)
        self.viz_data_generator = RandomPointsDataGenerator(None, None, 15, 5)
        self.viz_data = self.viz_data_generator.generate_data(save=False)

    def _sample_from_buffer(self, inputs, sample_concept=True):
        xs, observer, concept = inputs
        '''
            if sample concept is True, then replace 30% of the concept with samples from the buffer
            else, replace 30% of the observer with samples from the buffer
        '''
        batch_size = xs.shape[0]

        if not len(self.buffer) > batch_size:
            return inputs
        r = np.random.rand(batch_size)>0.5
        buffer_samples = deepcopy(self.buffer.sample(batch_size))
        if sample_concept:
            # sample from the buffer
            concept_ = []
            for i in range(batch_size):
                if r[i]:
                    concept_.append(buffer_samples[i][1].to(self.device))
                else:
                    concept_.append(concept[i])
            concept = torch.stack(concept_, dim=0)
            return xs, observer, concept
        else:
            # sample from the buffer
            observer_ = []
            for i in range(batch_size):
                if r[i]:
                    observer_.append(buffer_samples[i][0].to(self.device))
                else:
                    observer_.append(observer[i])
            observer = torch.stack(observer_, dim=0)
            return xs, observer, concept
    
    def _add_to_buffer(self, inputs):
        observer, concept = inputs
        observer, concept = observer.clone().detach(), concept.clone().detach()
        for i in range(observer.shape[0]):
            self.buffer.push([observer[i].cpu(), concept[i].cpu()])

    def langevin(self, neg, over_concepts = True):
        step_lr = self.args.step_lr
        xs, observer, concept = neg
        if over_concepts:
            noise = torch.randn_like(concept.float()).detach()
        else:
            noise = torch.randn_like(observer).detach()
        negs_samples = []
        observer_init = observer.clone().detach()
    
        for i in range(self.args.num_langevin_steps):
            step_lr = step_lr
            noise.normal_()
            if over_concepts:
                concept = concept.clone().float() + noise
                concept.requires_grad_(requires_grad=True)
            else:
                observer = observer.clone().float() + noise
                observer.requires_grad_(requires_grad=True)
                # if i == self.args.num_langevin_steps - 1:
                    # print(observer-observer_init)
            
            xs_ = xs.clone().detach()

            energy = self.model(xs_, observer, concept)

            _grad = torch.autograd.grad([energy.sum()], [observer if not over_concepts else concept])[0]
            # if i == self.args.num_langevin_steps - 1:
                # print(_grad.norm(dim=1).mean())
                # print(_grad.norm(dim=1).shape)


            if over_concepts:
                concept = concept - step_lr * _grad
            else:
                observer = observer - step_lr * _grad

            observer_kl = observer.clone().detach().float().requires_grad_(requires_grad=True)
            concept_kl = concept.clone().detach().float().requires_grad_(requires_grad=True)

            if i==self.args.num_langevin_steps-1:
                # with torch.backends.cuda.sdp_kernel(enable_mem_efficient=False):
                    xs_ = xs.clone().detach()
                    energy = self.model(xs_, observer_kl if not over_concepts else observer_kl, concept_kl if over_concepts else concept_kl)
                    _grad = torch.autograd.grad([energy.sum()], [observer_kl if not over_concepts else concept_kl], 
                                                create_graph=True)[0]
                    if over_concepts:
                        concept_kl = concept_kl - step_lr * _grad
                        concept_kl = concept_kl.clamp(0, 1)
                    else:
                        observer_kl = observer_kl - step_lr * _grad
                        # print grad norm
                        # print(_grad.norm())
                        observer_kl = observer_kl.clamp(-1, 1)

            # clip observer between 0 and 10
            if not over_concepts:
                observer = torch.clamp(observer.detach(), -1, 1)
            # clip concept between 0 and 5
            if over_concepts:
                concept = torch.clamp(concept.detach(), 0, 1)
                
            
            negs_samples.append((xs, observer, concept))
        return (
            (xs.detach(), observer.detach(), concept.detach()),
            (xs.detach(), observer_kl if not over_concepts else observer_kl.detach(), concept_kl if over_concepts else concept_kl.detach()),
            negs_samples
        )
    
    def _prepare_input(self, batch_xs, batch_observer, batch_concept):
        return {
            "pos": [batch_xs, batch_observer, batch_concept],
            "neg": [batch_xs, batch_observer, batch_concept],
        }
    
    def _transform_to_ovserver_center(self, inputs):
        xs, observer, concept = inputs
        observer = observer.unsqueeze(1).expand_as(xs[..., :2])
        xs[..., :2] = xs[..., :2] - observer
        return xs, concept
    
    def train_test_epoch(self, epoch, mode='train'):
        if mode == 'train':
            self.model.train()
        else:
            self.model.eval()

        acc, samples, gt_acc = 0, 0, 0
        progress_bar = tqdm.tqdm(range(len(self.dataset[mode])), desc=f"Epoch: {epoch}", unit="batch")
        for i, (batch_xs, batch_observer, batch_concept) in zip(progress_bar, self.dataset[mode]):
            batch_xs = batch_xs.to(self.device) + 0.01*torch.randn_like(batch_xs.float()).to(self.device)
            batch_observer = batch_observer.to(self.device) + 0.01*torch.randn_like(batch_observer.float()).to(self.device)
            batch_concept = (batch_concept + 0.01*torch.randn_like(batch_concept.float())).to(self.device)
            inputs = self._prepare_input(batch_xs, batch_observer, batch_concept)
            inputs_ = deepcopy(inputs)
            inputs_['neg'] = self._sample_from_buffer(inputs_['neg'], sample_concept=True)
            neg_con, neg_con_kl, _ = self.langevin(inputs_['neg'], over_concepts=True)

            inputs_= deepcopy(inputs)
            inputs_['neg'] = self._sample_from_buffer(inputs_['neg'], sample_concept=False)
            neg_xs, neg_xs_kl, _ = self.langevin(inputs_['neg'], over_concepts=False)

            energy_pos = self.model(*inputs['pos'])
            energy_neg_con = self.model(*neg_con)
            energy_neg_xs = self.model(*neg_xs)

            # self._add_to_buffer([neg_xs[1], neg_con[2]])

            loss = 0
            loss1 = torch.mean(energy_pos) - torch.mean(energy_neg_xs)
            # loss = 2*torch.mean(energy_pos) - (torch.mean(energy_neg_con/20) - torch.mean(energy_neg_xs))/2
            # loss += (energy_pos**2).mean() + (((energy_neg_con/10)**2).mean() + (energy_neg_xs**2).mean())/2
            # loss += torch.abs(energy_pos).mean() + (torch.abs(energy_neg_con).mean() + torch.abs(energy_neg_xs).mean())/2
            loss2 = 0.5*(energy_pos**2).mean() + (energy_neg_xs**2).mean()

            # kl loss
            self.model.requires_grad_(False)
            # loss_kl = (torch.mean(self.model(*neg_con[:2], neg_con_kl[2])/20) + torch.mean(self.model(neg_xs[0], neg_xs_kl[1], neg_xs[2])))/2
            loss_kl = torch.mean(self.model(*neg_xs_kl))
            self.model.requires_grad_(True)

            loss += self.args.kl_weight * loss_kl

            # loss = loss1 + loss2 + los3
            loss = loss1 + loss2
            if mode == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

            # if i%500 == 0:
            #     self.vizualize(epoch, i)


            training_stats = {
                "loss": f"{loss.item():+.2f}",
                # "loss1": f"{loss1.item():+.2f}",
                # "loss2": f"{loss2.item():+.2f}",
                # "loss_kl": f"{loss_kl.item():+.2f}",
                "energy_pos": f"{energy_pos.mean().item():+.2f}",
                "energy_neg_con": f"{energy_neg_con.mean().item():+.2f}",
                "energy_neg_xs": f"{energy_neg_xs.mean().item():+.2f}",
                "energy_diff_con": f"{(energy_pos - energy_neg_con).mean().item():+.2f}",
                "energy_diff_xs": f"{(energy_pos - energy_neg_xs).mean().item():+.2f}",
            }
            progress_bar.set_postfix(training_stats)

            
        return {
            "pos_energy": energy_pos.mean().item(),
            # "neg_con_energy": energy_neg_con.mean().item(),
            "neg_xs_energy": energy_neg_xs.mean().item(),
            # "energy_diff_con": (energy_pos - energy_neg_con).mean().item(),
            "energy_diff_xs": (energy_pos - energy_neg_xs).mean().item(),
            "loss": loss.item(),
        }
    
    def train(self):
        progress_bar = tqdm.tqdm(range(1, self.args.num_epochs+1), desc="Training", unit="epoch")
        self.vizualize(0, 0)
        for epoch in progress_bar:
            
            train_stats = self.train_test_epoch(epoch, mode='train')
            progress_bar.set_postfix(train_stats)
            
            if epoch % 10 == 0:
                self.vizualize(epoch, 0)

            # if epoch % self.args.eval_interval == 0:
            #     test_stats = self.train_test_epoch(epoch, mode='test')

    def vizualize(self, epoch=0, batch=0):
        '''
            use the viz data to plot the energy of the model on a grid by placing the observer at each point
        '''
        for i, viz_data in enumerate(self.viz_data):
            points, _, label = viz_data
            # print(points)
            human_readable_concept = self.viz_data_generator.decode_concept(label)
            points_ = torch.tensor(points, dtype=torch.float32).to(self.device)
            label = torch.tensor(label, dtype=torch.float32).to(self.device).long()
            object = label[..., 0].long()
            subject = label[..., 1].long()
            object_onehot = torch.nn.functional.one_hot(object, num_classes=6)
            subject_onehot = torch.nn.functional.one_hot(subject, num_classes=4)
            label = torch.cat((object_onehot, subject_onehot), dim=-1).unsqueeze(0)

            grid_xmin, grid_xmax, grid_ymin, grid_ymax = -1, 1, -1, 1
            num_samples = 20
            x = np.linspace(grid_xmin, grid_xmax, num_samples)
            y = np.linspace(grid_ymin, grid_ymax, num_samples)
            xx, yy = np.meshgrid(x, y)
            energies = np.zeros((num_samples, num_samples))
            # for each point in the grid, place the observer at that point and calculate the energy
            grid_points = np.array([[xx[i, j], yy[i, j]] for i in range(num_samples) for j in range(num_samples)])
            grid_points = torch.tensor(grid_points, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                for idx, gp in enumerate(grid_points):
                    energy = self.model(points_.clone().detach().unsqueeze(0), gp.unsqueeze(0), label)
                    gp = gp.cpu().numpy()
                    energies[idx // num_samples, idx % num_samples] = energy.detach().cpu().numpy()

            # plot the energy
            # plt.imshow(energies, extent=(grid_xmin, grid_xmax, grid_ymin, grid_ymax), origin='lower', aspect='auto')
            # plot the points with their respective colours
            col = np.array(COLOURS)
            c = col[points[..., 2].astype(int)]
            # print(points)
            plt.scatter(points[..., 0], points[..., 1], c=c)
            plt.imshow(energies, extent=(grid_xmin, grid_xmax, grid_ymin, grid_ymax), origin='lower', aspect='auto', alpha=0.5)
            plt.colorbar()
            plt.title(f"{human_readable_concept}")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.savefig(f"viz/epoch_{epoch}_batch_{batch}_{i}.png")
            plt.clf()
            plt.close()
        



        

            







            
