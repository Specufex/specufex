# base class for model saver
import os

import h5py


class SaveableModel:
    """Abstract mixin class to provide save/load functionality to models.
    the `save` method is genericand can be used as-is. The `_load` class
    needs to be called by a custom `load` function in the other class
    in order to ensure that the constructor for that class gets the
    correct arguments.

    Example::

        class MyModel(SaveableModel):
            def __init__(self, parameter):
                self.parameter = parameter

            @classmethod
            def load(cls, fiilename, parameter):
                return cls(parameter)._load(filename)

        my_model = MyModel(parameter).load(filename)
        my_model.save(filename2)

    """

    def save(self, filename, overwrite=False):
        """Saves a trained model to an hdf5 file.

        Arguments
        ----------
        filename: string
            Filename for saving the model
        overwrite: bool, default=False
            If the file already exists, set to True to
            overwrite it.
        """
        members = [
            attr
            for attr in dir(self)
            if not callable(getattr(self, attr)) and not attr.startswith("__")
        ]

        if os.path.exists(filename) and not overwrite:
            # this should probably raise an error
            print(
                f"""{filename} already exists. If you wish to overwrite this file,
            pass overwrite=True to the this function."""
            )
            return

        else:
            with h5py.File(filename, "w") as f:
                for member in members:
                    f.create_dataset(member, data=self.__getattribute__(member))

            print(f"Model saved to {filename}")

    def _load(self, filename):
        """Helper function for loading a saved model.

        Arguments
        ----------
        filename: str
            Path to file that contains the saved model parameters.
            This should be an HDF5 file, and is typically created by using
            the `save{}` method.

        Returns
        ----------
        BayesianNonparametricNMF
            An instance of BayesianNonparametricNMF with the parameters in the file.
        """
        with h5py.File(filename, "r") as f:
            for member in f.keys():
                self.__setattr__(member, f[member][()])
        return self
