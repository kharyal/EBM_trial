'''
data is of type:
([[x,y,colour], ...points], [observer_x, observer_y], [[<concept object>, <concept subject>], ...labels])

this file trains the concept Energy Based Model (EBM) on the data
'''

import os
import numpy as np
import torch
from models.concept_EBM import ConceptEBM
from buffers.buffer import Buffer
import tqdm

class ConceptEBMTrainer:
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.model = ConceptEBM()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.args = args
        self.buffer = Buffer(args.buffer_size)

    def langevin(self, neg, over_concepts = True):
        xs, concept = neg
        if over_concepts:
            noise = torch.randn_like(concept).detach()
        else:
            noise = torch.randn_like(xs).detach()
        negs_samples = []
    
        for i in range(self.args.num_langevin_steps):
            noise._normal()
            if over_concepts:
                concept = concept + noise
                concept.requires_grad = True
            else:
                xs = xs + noise
                xs.requires_grad = True
            
            energy = self.model(xs, concept)

            _grad = torch.autograd.grad([energy.sum()], [xs if not over_concepts else concept])[0]
            xs_kl = xs.clone()
            concept_kl = concept.clone()

            if over_concepts:
                concept = concept - self.args.step_lr * _grad
            else:
                xs = xs - self.args.step_lr * _grad

            if i==self.args.num_langevin_steps-1:
                energy = self.model(xs_kl, concept_kl)
                _grad = torch.autograd.grad([energy.sum()], [xs_kl if not over_concepts else concept_kl], 
                                            create_graph=True)[0]
                if over_concepts:
                    concept_kl = concept_kl - self.args.step_lr * _grad
                    concept_kl = torch.clamp(concept_kl, -1, 1)
                else:
                    xs_kl = xs_kl - self.args.step_lr * _grad
                    xs_kl = torch.clamp(xs_kl, -1, 1)
                
            
            negs_samples.append((xs, concept))
            return (
                (xs, concept),
                (xs_kl, concept_kl),
                negs_samples
            )
    
    def _prepare_input(self, batch_xs, batch_concept):
        return {
            "pos": [batch_xs, batch_concept],
            "neg": [batch_xs, batch_concept]
        }
    
    def train_test_epoch(self, epoch, mode='train'):
        if mode == 'train':
            self.model.train()
        else:
            self.model.eval()

        acc, samples, gt_acc = 0, 0, 0
        for i, (batch_xs, batch_concept) in enumerate(self.dataset[mode]):
            batch_xs = batch_xs.to(self.device)
            batch_concept = batch_concept.to(self.device)
            inputs = self._prepare_input(batch_xs, batch_concept)
            neg_con, neg_con_kl, _ = self.langevin(inputs['neg'], over_concepts=True)
            neg_xs, neg_xs_kl, _ = self.langevin(inputs['neg'], over_concepts=False)

            energy_pos = self.model(*inputs['pos'])
            energy_neg_con = self.model(*neg_con)
            energy_neg_xs = self.model(*neg_xs)

            loss = 2*torch.mean(energy_pos) - torch.mean(energy_neg_con) - torch.mean(energy_neg_xs)
            loss += 2*(energy_pos**2).mean() - (energy_neg_con**2).mean() - (energy_neg_xs**2).mean()

            # kl loss
            self.model.requires_grad__(False)
            loss_kl = torch.mean(self.model(*neg_con_kl)) + torch.mean(self.model(*neg_xs_kl))
            self.model.requires_grad__(True)

            loss = loss + self.args.kl_weight * loss_kl


            if mode == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                self.optimizer.zero_grad()

            
        return {
            "pos_energy": energy_pos.mean().item(),
            "neg_con_energy": energy_neg_con.mean().item(),
            "neg_xs_energy": energy_neg_xs.mean().item(),
            "energy_diff_con": (energy_pos - energy_neg_con).mean().item(),
            "energy_diff_xs": (energy_pos - energy_neg_xs).mean().item(),
            "loss": loss.item(),
        }
    
    def train(self):
        progress_bar = tqdm.tqdm(range(self.args.num_epochs), desc="Training", unit="epoch")
        for epoch in progress_bar:
            train_stats = self.train_test_epoch(epoch, mode='train')
            progress_bar.set_postfix(train_stats)

            # if epoch % self.args.eval_interval == 0:
            #     test_stats = self.train_test_epoch(epoch, mode='test')
        
        
            







            
