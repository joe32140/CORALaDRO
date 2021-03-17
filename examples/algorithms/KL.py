import torch
from models.initializer import initialize_model
from algorithms.single_model_algorithm import SingleModelAlgorithm
from wilds.common.utils import split_into_groups

class KL(SingleModelAlgorithm):
    """
    Deep CORAL.
    This algorithm was originally proposed as an unsupervised domain adaptation algorithm.

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
    def __init__(self, config, d_out, grouper, loss, metric, n_train_steps):
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
        # set model components
        self.featurizer = featurizer
        self.classifier = classifier

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
        """
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


        avg_loss = self.loss.compute(results['y_pred'], results['y_true'], return_dict=False)

        return avg_loss + penalty * self.penalty_weight
