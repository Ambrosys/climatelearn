import classification.weka_class_MP as weka


def classification_train(method, params, train_data):
    """
    It is called to perform a single training with random validation set. The model trained until the default stopping
    condition is met, is then returned,

    config: dictionary
            It contains all the configuration data used for setting up the method and run the training,

    train_data: pandas Dataframe
            It contains the training set with features and target variables. It will be divided into training
            and validation sets by the method itself

    Returns: lambda function, string, dictionary
            The first return is a lambda function that can be used for testing the model on a new train set or for
            predicting targets given new test features. The second return is a unique string function determining the
            state of the model and can be used to save into the database. The third return is a dictionary containing
            all the relatives error on training and validation sets.

    """
    if method == 'MPWEKA':
        model = train_MPWEKA(params, train_data)
    else:
        print 'method not recogniseed. Quitting!!'
        model = None
        exit()
    return model


def train_MPWEKA(params, train_data):
    model = weka.MP_Weka_class(train_data, params)
    model_trained , model_train_error, _ = model.learn()
    return [lambda X : model.test(X)]