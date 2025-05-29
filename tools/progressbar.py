from typing import Any, Dict, Optional, Union
from pytorch_lightning.callbacks import TQDMProgressBar
from tqdm import tqdm
import sys
# from lightning.pytorch.utilities.types import STEP_OUTPUT
def _update_n(bar: tqdm, value: int) -> None:
    # print('bar.disable: ',bar.disable)
    if not bar.disable:
        bar.n = value
        bar.refresh()

class MyTQDMProgressBar(TQDMProgressBar):
	# def __init__(self):
    # def init_validation_tqdm(self):
    #     bar = super().init_validation_tqdm()
    #     bar.set_description("running validation...")
    #     return bar
    # def init_validation_tqdm(self):
    #     """Override this to customize the tqdm bar for validation."""
    #     # The main progress bar doesn't exist in `trainer.validate()`
    #     has_main_bar = self.trainer.state.fn != "validate"
    #     # print("has_main_bar  ",has_main_bar)
    #     bar = tqdm(
    #         desc=self.validation_description,
    #         position=(2 * self.process_position + has_main_bar),
    #         # position=(2 * self.process_position),
    #         disable=self.is_disabled,
    #         leave=not has_main_bar,
    #         dynamic_ncols=True,
    #         # smoothing=0,
    #         file=sys.stdout,
    #     )
    #     # bar.set_description("running validation...")
    #     return bar

    # def on_validation_batch_end(self,
    #     trainer: "pl.Trainer",
    #     pl_module: "pl.LightningModule",
    #     outputs: Optional[STEP_OUTPUT],
    #     batch: Any,
    #     batch_idx: int,
    #     dataloader_idx: int = 0,) -> None:
    #     if self._should_update(batch_idx, self._val_progress_bar.total):
    #         _update_n(self._val_progress_bar, batch_idx)

    #     # print('self.train_batch_idx + self._val_processed',self.train_batch_idx,self._val_processed)
    
    def get_metrics(self, *args, **kwargs):
        # don't show the version number
        items = super().get_metrics(*args, **kwargs)
        items.pop("v_num", None)
        items.pop("loss", None)
        return items