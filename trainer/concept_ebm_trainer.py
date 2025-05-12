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
from data_generator.random_points_data_generator import RandomPointsDataGenerator, COLOURS
import matplotlib.pyplot as plt

class ConceptEBMTrainer:
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.model = ConceptEBM()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.args = args
        self.buffer = Buffer(args.buffer_size)
        self.viz_data_generator = RandomPointsDataGenerator(None, None, 1, 5)
        self.viz_data = self.viz_data_generator.generate_data(save=False)

    def langevin(self, neg, over_concepts = True):
        xs, concept = neg
        if over_concepts:
            noise = torch.randn_like(concept).detach()
        else:
            noise = torch.randn_like(xs).detach()
        negs_samples = []
    
        for i in range(self.args.num_langevin_steps):
            noise.normal_()
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
        progress_bar = tqdm.tqdm(range(len(self.dataset[mode])), desc=f"Epoch: {epoch}", unit="batch")
        for i, (batch_xs, batch_concept) in zip(progress_bar, self.dataset[mode]):
            batch_xs = batch_xs.to(self.device)
            batch_concept = batch_concept.to(self.device)
            inputs = self._prepare_input(batch_xs, batch_concept)
            neg_con, neg_con_kl, _ = self.langevin(inputs['neg'], over_concepts=True)
            neg_xs, neg_xs_kl, _ = self.langevin(inputs['neg'], over_concepts=False)

            energy_pos = self.model(*inputs['pos'])
            energy_neg_con = self.model(*neg_con)
            energy_neg_xs = self.model(*neg_xs)

            loss = 2*torch.mean(energy_pos) - torch.mean(energy_neg_con) - torch.mean(energy_neg_xs)
            loss += 2*(energy_pos**2).mean() + (energy_neg_con**2).mean() + (energy_neg_xs**2).mean()

            # kl loss
            self.model.requires_grad_(False)
            loss_kl = torch.mean(self.model(*neg_con_kl)) + torch.mean(self.model(*neg_xs_kl))
            self.model.requires_grad_(True)

            loss = loss + self.args.kl_weight * loss_kl

            if i%500 == 0:
                self.vizualize(epoch, i)


            if mode == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                self.optimizer.zero_grad()

            training_stats = {
                "loss": f"{loss.item():+.2f}",
                "energy_pos": f"{energy_pos.mean().item():+.2f}",
                "energy_neg_con": f"{energy_neg_con.mean().item():+.2f}",
                "energy_neg_xs": f"{energy_neg_xs.mean().item():+.2f}",
                "energy_diff_con": f"{(energy_pos - energy_neg_con).mean().item():+.2f}",
                "energy_diff_xs": f"{(energy_pos - energy_neg_xs).mean().item():+.2f}",
            }
            progress_bar.set_postfix(training_stats)

            
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

            # self.vizualize()

            train_stats = self.train_test_epoch(epoch, mode='train')
            progress_bar.set_postfix(train_stats)

            # if epoch % self.args.eval_interval == 0:
            #     test_stats = self.train_test_epoch(epoch, mode='test')

    def vizualize(self, epoch=0, batch=0):
        '''
            use the viz data to plot the energy of the model on a grid by placing the observer at each point
        '''
        for i, viz_data in enumerate(self.viz_data):
            points, _, label = viz_data
            human_readable_concept = self.viz_data_generator.decode_concept(label)
            points = torch.tensor(points, dtype=torch.float32).to(self.device)
            label = torch.tensor(label, dtype=torch.float32).to(self.device).unsqueeze(0)

            grid_xmin, grid_xmax, grid_ymin, grid_ymax = 0, 10, 0, 10
            num_samples = 20
            x = np.linspace(grid_xmin, grid_xmax, num_samples)
            y = np.linspace(grid_ymin, grid_ymax, num_samples)
            xx, yy = np.meshgrid(x, y)
            energies = np.zeros((num_samples, num_samples))
            # for each point in the grid, place the observer at that point and calculate the energy
            grid_points = np.array([[xx[i, j], yy[i, j]] for i in range(num_samples) for j in range(num_samples)])
            grid_points = torch.tensor(grid_points, dtype=torch.float32).to(self.device)
            for idx, gp in enumerate(grid_points):
                points_ = points.clone()
                points_[..., :2] = points[...,:2] - gp
                points_ = points_.unsqueeze(0)
                energy = self.model(points_, label)
                gp = gp.cpu().numpy()
                energies[idx // num_samples, idx % num_samples] = energy.detach().cpu().numpy()

            # plot the energy
            # plt.imshow(energies, extent=(grid_xmin, grid_xmax, grid_ymin, grid_ymax), origin='lower', aspect='auto')
            # plot the points with their respective colours
            col = np.array(COLOURS)
            c = col[points[..., 2].cpu().numpy().astype(int)]
            plt.scatter(points[..., 0].cpu().numpy(), points[..., 1].cpu().numpy(), c=c)
            plt.imshow(energies, extent=(grid_xmin, grid_xmax, grid_ymin, grid_ymax), origin='lower', aspect='auto', alpha=0.5)
            plt.colorbar()
            plt.title(f"{human_readable_concept}")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.savefig(f"viz/epoch_{epoch}_batch_{batch}_{i}.png")
            plt.clf()
            plt.close()
        



        

            







            
