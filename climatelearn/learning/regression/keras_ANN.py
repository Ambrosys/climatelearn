from copy import copy


#from amb4cast.utils import doc_inherit
from climatelearn.learning.base_class import RegressionModel
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping
from keras.layers import BatchNormalization
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential, model_from_json
from keras.optimizers import RMSprop
from keras.regularizers import l2


class KerasRegressionModel(RegressionModel):
    def __init__(self, arity=1, network_structure=(1,), activation_function="tanh",
                 error_metric="rmse",
                 optimizer_type="nadam", learning_rate=None, loss_function="mse", nb_epoch=1, batch_size=100,
                 early_stopping=False,
                 weight_init_method="normal", graphical_verbose=False,
                 validation_split=0.1, dropout=False, dropout_input_layer_fraction=0.2,
                 dropout_hidden_layer_fraction=0.5, batch_normalization=False, verbose=False, weight_decay=False,
                 weigt_decay_parameter=0.001,
                 **kwargs):
        """

        A class to construct arbitrary artifical neural networks using Keras library (http://keras.io/).
        The module supports state-of-the-art technologies for optimization and regularization of ANNs.

        :param network_structure: A tuple which specifies the number of neurons for each layer
        :param activation_function: Activation function used, cf. http://keras.io/activations/
        :param error_metric: Error metric
        :param optimizer_type: Specifies the optimization method used
        :param loss_function: Loss function (given by Keras or custom loss functions), cf. http://keras.io/objectives/
        :param nb_epoch: Number of training epochs
        :param batch_size: Batch size used for mini-batch learning
        :param early_stopping: If set True, training will be interruptped when the loss isn't decaying anymore
        :param init: Method of weight initialization, e.g normal, glorot_normal, uniform
        :param arity: Input dimension
        :param verbose: Verbose mode, verbose=1 show progress bar logging, verbose=2 show console logging
        :param graphical_verbose: If True,
        :param dropout: Use dropout layers for regularization
        :param dropout_input_layer_fraction: Fraction of input units to drop
        :param dropout_hidden_layer_fraction: Fraction hidden layer units to drop
        :param batch_normalization: Activate batch normalization
        :param weight_decay: Activate weight decay regularization method
        :param weight_decay_parameter: Sets the weight decay regularization parameter
        :param kwargs:
        """

        super(RegressionModel, self).__init__()
        #self.logger.info("Compiling ANN...")
        self.__dict__.update(locals())

        # Initialize ANN structure

        self.__model = Sequential()

        self.input_layer_params = {"input_shape": (self.arity,), "activation": self.activation_function,
                                       "output_dim": self.network_structure[0], "init": self.weight_init_method}

        self.hidden_layer_params = {"activation": self.activation_function, "init": self.weight_init_method}
        if self.weight_decay:
            self.hidden_layer_params["W_regularizer"] = l2(weigt_decay_parameter)

        self.output_layer_params = {"activation": "linear", "init": self.weight_init_method}

        self.create_input_layer()  # stack up remaining layers.
        self.create_hidden_layers()
        self.create_output_layer()


        # compile the neural network
        self.__model.compile(optimizer=RMSprop(lr=0.001), loss=self.loss_function)
        #self.logger.info("Compilation completed...")
        self.func = self.__model.predict


    def add_layer(self, num_nodes, layer_params, dropout=False):
        self.__model.add(Dense(num_nodes, **layer_params))
        if (dropout):
            self.__model.add(Dropout(self.dropout_hidden_layer_fraction))

    def create_input_layer(self):
        if self.dropout:
            self.__model.add(Dropout(self.dropout_input_layer_fraction, input_shape=(self.arity,)))
            del self.input_layer_params["input_shape"]
        self.__model.add(Dense(**self.input_layer_params))
        if self.batch_normalization:
            self.__model.add(BatchNormalization())

    def create_hidden_layers(self):
        for num_nodes in self.network_structure[1:-1]:
            self.add_layer(num_nodes, self.hidden_layer_params, dropout=self.dropout)
            if self.batch_normalization:
                self.__model.add(BatchNormalization())

    def create_output_layer(self):
        self.add_layer(self.network_structure[-1], self.output_layer_params)
        if self.batch_normalization:
            self.__model.add(BatchNormalization())

    #@property
    #def callbacks(self):
    #    cbs = [EpochLogger(self.logger)]
    #    if self.early_stopping:
    #        cbs += [EarlyStopping(monitor='val_loss', patience=2)]
    #    return cbs

    @property
    def weights(self):
        return self.__model.get_weights()

    #@doc_inherit
    def fit(self, xfit, yfit):
        self.hist = self.__model.fit(xfit, yfit, nb_epoch=self.nb_epoch, batch_size=self.batch_size,
                                     verbose=self.verbose, validation_split=self.validation_split)#, callbacks=self.callbacks)
        return self

    def __getstate__(self):
        """
        Function to make ANNRegressionModel pickable.
        The weights, the architecture as also the ANN compilation settings are stored in a dictionary.
        :return: The dictionary containing ANN architecture in json format, weight and ANN compilation setting
        """

        state = copy(self.__dict__)
        del state["func"]
        #del state["logger"]
        #del state["_ANNRegressionModel__model"]
        del state["hist"]
        return dict(json_model=self.__model.to_json(), weights=self.__model.get_weights(), config=state)

    def __setstate__(self, d):
        """
        Function to make ANNRegressionModel pickable
        :param d:
        :return:
        """

        self.__dict__ = d["config"]
        self.__model = model_from_json(d["json_model"])
        self.__model.set_weights(d["weights"])
        self.func = self.__model.predict

    def print_summary(self):
        """
        Print summary of the neural network, includes architecture and compiling setting.
        :return:
        """
        self.__model.summary()

