from torch.optim.lr_scheduler import SequentialLR, ConstantLR, ReduceLROnPlateau
from bisect import bisect_right

class SequentialLRwithRLROP(SequentialLR):
    '''
    This class is specified for the use with pytorch lightning [one optimizer, 
    two schedulers (ReduceLROnPlateau is the second one)]
    _milestones here is used with steps rather than epoches
    '''
    def step(self, monitor=None):
        assert len(self._milestones) == 1, 'Only support two schedulers'
        if monitor is None:
            # count for the first scheduler
            if not hasattr(self,'steps'):
                self.steps = 0
            self.steps += 1
        else:
            # count for the first scheduler
            self.last_epoch += 1 
        idx = bisect_right(self._milestones, self.steps)
        scheduler = self._schedulers[idx]
        # if idx > 0 and self._milestones[idx - 1] == self.last_epoch:
        #     scheduler.step(0)
        # else:
        if idx == 0:
            scheduler.step()
        elif idx == 1 and monitor is not None:
            scheduler.step(monitor)
    
        # print("idx ",self.steps, self.last_epoch, idx,isinstance(scheduler, ReduceLROnPlateau))
        if isinstance(scheduler, ReduceLROnPlateau):
            self._last_lr = scheduler.optimizer.param_groups[0]['lr']
        else:
            self._last_lr = scheduler.get_last_lr()
        # print("self._last_lr ",self._last_lr)
