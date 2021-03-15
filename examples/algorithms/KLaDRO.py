import torch
from algorithms.single_model_algorithm import SingleModelAlgorithm
from models.initializer import initialize_model
from wilds.common.utils import split_into_groups

class KLaDRO(SingleModelAlgorithm):
    """
    Group distributionally robust optimization.
    and
    KL Divergence Loss. (This algorithm is modified based on Deep CORAL.)

    Original paper:
        @inproceedings{sagawa2019distributionally,
          title={Distributionally robust neural networks for group shifts: On the importance of regularization for worst-case generalization},
          author={Sagawa, Shiori and Koh, Pang Wei and Hashimoto, Tatsunori B and Liang, Percy},
          booktitle={International Conference on Learning Representations},
          year={2019}
        }
    Original paper:
        @inproceedings{sun2016deep,
          title={Deep CORAL: Correlation alignment for deep domain adaptation},
          author={Sun, Baochen and Saenko, Kate},
          booktitle={European Conference on Computer Vision},
          pages={443--450},
          year={2016},
          organization={Springer}
        }

    The CORAL penalty function below is adapted from DomainBed's implementation:
    https://github.com/facebookresearch/DomainBed/blob/1a61f7ff44b02776619803a1dd12f952528ca531/domainbed/algorithms.py#L539    
    """
    def __init__(self, config, d_out, grouper, loss, metric, n_train_steps, is_group_in_train):
        # check config
        assert config.train_loader == 'group'
        assert config.uniform_over_groups
        assert config.distinct_groups
        # initialize models
        featurizer, classifier = initialize_model(config, d_out=d_out, is_featurizer=True)
        featurizer = featurizer.to(config.device)
        classifier = classifier.to(config.device)
        model = torch.nn.Sequential(featurizer, classifier).to(config.device)
        # initialize module
        super().__init__(
            config=config,
            model=model,
            grouper=grouper,
            loss=loss,
            metric=metric,
            n_train_steps=n_train_steps,
        )
        # algorithm hyperparameters
        self.penalty_weight = config.coral_penalty_weight
        # additional logging
        self.logged_fields.append('penalty')
        self.logged_fields.append('group_weight')
        # set model components
        self.featurizer = featurizer
        self.classifier = classifier
        # step size
        self.group_weights_step_size = config.group_dro_step_size
        # initialize adversarial weights
        self.group_weights = torch.zeros(grouper.n_groups)
        self.group_weights[is_group_in_train] = 1
        self.group_weights = self.group_weights/self.group_weights.sum()
        self.group_weights = self.group_weights.to(self.device)

    def coral_penalty(self, x, y):
        if x.dim() > 2: 
            # featurizers output Tensors of size (batch_size, ..., feature dimensionality).
            # we flatten to Tensors of size (*, feature dimensionality)
            x = x.view(-1, x.size(-1))
            y = y.view(-1, y.size(-1))

        x_index = torch.arange(x.size(0)).repeat(y.size(0))
        y_index = torch.arange(y.size(0)).repeat_interleave(x.size(0))

        xx = torch.nn.functional.logsigmoid(x[x_index])
        yy = torch.nn.functional.logsigmoid(y[y_index])
        mm = torch.log(0.5 * (torch.exp(xx) + torch.exp(yy)))

        # m = 0.5 (x[x_index] + y[y_index])
        # mm = torch.nn.functional.logsigmoid(m)
        loss = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)

        return 0.5 * loss(xx, mm) + 0.5 * loss(yy, mm)

    def process_batch(self, batch):
        """5
        Override
        """
        # forward pass
        x, y_true, metadata = batch
        x = x.to(self.device)
        y_true = y_true.to(self.device)
        g = self.grouper.metadata_to_group(metadata).to(self.device)
        features = self.featurizer(x)
        outputs = self.classifier(features)

        # package the results
        results = {
            'g': g,
            'y_true': y_true,
            'y_pred': outputs,
            'metadata': metadata,
            'features': features,
            }
        results['group_weight'] = self.group_weights
        return results

    def objective(self, results):
        # extract features
        features = results.pop('features')
        if self.is_training:
            # split into groups
            unique_groups, group_indices, _ = split_into_groups(results['g'])
            # compute penalty
            n_groups_per_batch = unique_groups.numel()
            penalty = torch.zeros(1, device=self.device)
            for i_group in range(n_groups_per_batch):
                for j_group in range(i_group+1, n_groups_per_batch):
                    penalty += self.coral_penalty(features[group_indices[i_group]], features[group_indices[j_group]])
            if n_groups_per_batch > 1:
                penalty /= (n_groups_per_batch * (n_groups_per_batch-1) / 2) # get the mean penalty
            # save penalty
        else:
            penalty = 0.
        if isinstance(penalty, torch.Tensor):
            results['penalty'] = penalty.item()
        else:
            results['penalty'] = penalty

        group_losses, _, _ = self.loss.compute_group_wise(
            results['y_pred'],
            results['y_true'],
            results['g'],
            self.grouper.n_groups,
            return_dict=False)
        return group_losses @ self.group_weights + penalty * self.penalty_weight

    def _update(self, results):
        """
        Process the batch, update the log, and update the model, group weights, and scheduler.
        Args:
            - batch (tuple of Tensors): a batch of data yielded by data loaders
        Output:
            - results (dictionary): information about the batch, such as:
                - g (Tensor)
                - y_true (Tensor)
                - metadata (Tensor)
                - loss (Tensor)
                - metrics (Tensor)
                - objective (float)
        """
        # compute group losses
        group_losses, _, _ = self.loss.compute_group_wise(
            results['y_pred'],
            results['y_true'],
            results['g'],
            self.grouper.n_groups,
            return_dict=False)
        # update group weights
        self.group_weights = self.group_weights * torch.exp(self.group_weights_step_size*group_losses.data)
        self.group_weights = (self.group_weights/(self.group_weights.sum()))
        # save updated group weights
        results['group_weight'] = self.group_weights
        # update model
        super()._update(results)
