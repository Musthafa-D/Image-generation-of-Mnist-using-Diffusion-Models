import os
from abc import abstractmethod, ABC

from ccbdl.learning.base import BaseDeepLearning
from ccbdl.utils.datatypes import TaskPool


class BaseDifussionLearning(BaseDeepLearning, ABC):
    """
    
    Abstractmethods
    ---------------
    _train_epoch:
        training loop of one epoch.
    _test_epoch:
        testing loop of epoch.
    _validate_epoch:
        validation loop of one epoch.
    evaluate:
        evaluation (ploting + saving) of important metrics.
    learn:
        training, validating, testing and evaluating.
    _save:
        saving e.g. network and storages.
    """

    def __init__(self, train_data, test_data, val_data, path, config,
                 data_storage_names=[
                     "epoch", "batch", "train_loss", "test_loss"],
                 task=TaskPool.FORECAST, debug: bool = False, logging: bool = False):
        super().__init__(train_data, test_data, val_data,
                         task, path, config, data_storage_names, debug, logging)

        self.best_state_dict = None
        self.best_values = {"TrainLoss": float('Inf'),
                            "TestLoss": float('Inf'),
                            "Batch": -1}
        self.test_loss = None
        self.train_loss = None

    @abstractmethod
    def noising_images(self):
        pass
    
    @abstractmethod
    def noise_prediction(self):
        pass


class EmptyDiffusionLearning(BaseDifussionLearning):
    def __init__(self,
                 train_data=None,
                 test_data=None,
                 val_data=None,
                 path=os.getcwd(),
                 config={},
                 debug: bool = False,
                 logging: bool = False):
        super().__init__(train_data, test_data, val_data,
                         path, config, debug=debug, logging=logging)

    def _train_epoch(self):
        pass

    def _test_epoch(self):
        pass

    def _validate_epoch(self):
        pass

    def _hook_every_epoch(self):
        pass

    def _update_best(self):
        pass

    def evaluate(self):
        pass

    def _save(self):
        pass

    def noising_images(self):
        pass
    
    def noise_prediction(self):
        pass


class BaseGANLearning(BaseDeepLearning, ABC):
    """
    
    Abstractmethods
    ---------------
    _train_epoch:
        training loop of one epoch.
    _test_epoch:
        testing loop of epoch.
    _validate_epoch:
        validation loop of one epoch.
    evaluate:
        evaluation (ploting + saving) of important metrics.
    learn:
        training, validating, testing and evaluating.
    _save:
        saving e.g. network and storages.
    """

    def __init__(self, train_data, test_data, val_data, path, config,
                 data_storage_names=[
                     "epoch", "batch", "train_acc", "test_acc", "dis_loss", "gen_loss", "fid_score"],
                 task=TaskPool.GENERATE, debug: bool = False, logging: bool = False):
        super().__init__(train_data, test_data, val_data,
                         task, path, config, data_storage_names, debug, logging)

        self.best_state_dict = None
        self.best_values = {"GenLoss": float('Inf'),
                            "DisLoss": float('Inf'),
                            "FidScore": float('Inf'),
                            "Batch": -1}

        self.test_loss = None
        self.train_loss = None

    @abstractmethod
    def _generate(self):
        pass

    @abstractmethod
    def _discriminate(self):
        pass


class EmptyGANLearning(BaseGANLearning):
    def __init__(self,
                 train_data=None,
                 test_data=None,
                 val_data=None,
                 path=os.getcwd(),
                 config={},
                 debug: bool = False,
                 logging: bool = False):
        super().__init__(train_data, test_data, val_data,
                         path, config, debug=debug, logging=logging)

    def _train_epoch(self):
        pass

    def _test_epoch(self):
        pass

    def _validate_epoch(self):
        pass

    def _hook_every_epoch(self):
        pass

    def _update_best(self):
        pass

    def evaluate(self):
        pass

    def _save(self):
        pass

    def _generate(self):
        pass
    
    def _discriminate(self):
        pass


class BaseCGANLearning(BaseDeepLearning, ABC):
    """
    
    Abstractmethods
    ---------------
    _train_epoch:
        training loop of one epoch.
    _test_epoch:
        testing loop of epoch.
    _validate_epoch:
        validation loop of one epoch.
    evaluate:
        evaluation (ploting + saving) of important metrics.
    learn:
        training, validating, testing and evaluating.
    _save:
        saving e.g. network and storages.
    """

    def __init__(self, train_data, test_data, val_data, path, config,
                 data_storage_names=[
                     "epoch", "batch", "train_acc", "test_acc", "dis_loss", "gen_loss", "fid_score"],
                 task=TaskPool.CONDITIONAL_GENERATE, debug: bool = False, logging: bool = False):
        super().__init__(train_data, test_data, val_data,
                         task, path, config, data_storage_names, debug, logging)

        self.best_state_dict = None
        self.best_values = {"GenLoss": float('Inf'),
                            "DisLoss": float('Inf'),
                            "FidScore": float('Inf'),
                            "Batch": -1}

        self.test_loss = None
        self.train_loss = None
    
    @abstractmethod
    def _generate(self):
        pass

    @abstractmethod
    def _discriminate(self):
        pass   

    
class EmptyCGANLearning(BaseCGANLearning):
    def __init__(self,
                 train_data=None,
                 test_data=None,
                 val_data=None,
                 path=os.getcwd(),
                 config={},
                 debug: bool = False,
                 logging: bool = False):
        super().__init__(train_data, test_data, val_data,
                         path, config, debug=debug, logging=logging)

    def _train_epoch(self):
        pass

    def _test_epoch(self):
        pass

    def _validate_epoch(self):
        pass

    def _hook_every_epoch(self):
        pass

    def _update_best(self):
        pass

    def evaluate(self):
        pass

    def _save(self):
        pass
    
    def _generate(self):
        pass
    
    def _discriminate(self):
        pass