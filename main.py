import arff
import xml.etree.ElementTree as ET
import numpy as np
import theano
import theano.tensor as T
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from mlp import test_mlp
from DBN import test_DBN
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import hamming_loss


def shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x,
                                        dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(np.asarray(data_y,
                                        dtype=theano.config.floatX),
                             borrow=borrow)
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    return shared_x, T.cast(shared_y, 'int32')


def load_data(arffPath, xmlPath):
    # load data from file
    arff_data = arff.load(open(arffPath), 'rb')
    xml_root = ET.parse(xmlPath).getroot()

    # prepare variable
    attributes = arff_data['attributes']
    isLabels = {}
    for el in xml_root:
        isLabels[el.attrib['name']] = True

    # create data map for using in create data
    dataToAttrMap = []
    for i in range(len(attributes)):
        attributesName = attributes[i][0].decode('string_escape')
        if type(attributes[i][1]) is list:
            if not attributesName in isLabels:
                for j in range(len(attributes[i][1])):
                    dataToAttrMap.append(
                        {'type': 'NOMINAL', 'name': attributesName, 'index': i, 'value': attributes[i][1][j]})
            else:
                dataToAttrMap.append({'type': 'LABEL', 'name': attributesName, 'index': i})
            pass
        elif type(attributes[i][1]) is unicode and attributes[i][1] == 'NUMERIC':
            dataToAttrMap.append({'type': 'NUMERIC', 'name': attributesName, 'index': i})
            pass
        else:
            print('pass attribute fail, there is something wrong')

    # create data
    data_x = []
    for i in range(len(arff_data['data'])):
        data_x.append([])
        for j in range(len(dataToAttrMap)):
            attrInd = dataToAttrMap[j]['index']
            if dataToAttrMap[j]['type'] == 'NOMINAL':
                if str(arff_data['data'][i][attrInd]) == str(dataToAttrMap[j]['value']):
                    data_x[i].append(1)
                else:
                    data_x[i].append(0)
            elif dataToAttrMap[j]['type'] == 'NUMERIC':
                data_x[i].append(arff_data['data'][i][attrInd])
    data_y = []
    for i in range(len(arff_data['data'])):
        data_y.append([])
        for j in range(len(dataToAttrMap)):
            attrInd = dataToAttrMap[j]['index']
            if dataToAttrMap[j]['type'] == 'LABEL':
                data_y[i].append(int(arff_data['data'][i][attrInd]))

    # data scaler
    data_x = preprocessing.MinMaxScaler().fit_transform(np.array(data_x))

    # split data to train, valid, test
    data_x_train, data_x_test, data_y_train, data_y_test = train_test_split(data_x, data_y, test_size=0.2)
    data_x_train, data_x_valid, data_y_train, data_y_valid = train_test_split(data_x_train, data_y_train, test_size=0.25)
    data_train = data_x_train, data_y_train
    data_valid = data_x_valid, data_y_valid
    data_test = data_x_test, data_y_test

    # create shared dataset
    train_set_x, train_set_y = shared_dataset(data_train)
    valid_set_x, valid_set_y = shared_dataset(data_valid)
    test_set_x, test_set_y = shared_dataset(data_test)
    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]

    return rval, len(data_x[0]), len(data_y[0]), data_y_test


# datasets, attrSize, classSize, actualLabels = load_data('datasets/flags/flags.arff', 'datasets/flags/flags.xml')
# datasets = load_data('datasets/birds/birds-train.arff','datasets/birds/birds.xml')
datasets, attrSize, classSize, actualLabels = load_data('datasets/yeast/yeast.arff', 'datasets/yeast/yeast.xml')

# error, predictedLabels = test_mlp(datasets = datasets, n_in=attrSize, n_out=classSize, n_hidden=60, batch_size=5, n_epochs=1000, learning_rate=0.03, L1_reg=0.001, L2_reg=0.001)
error, predictedLabels = test_DBN(datasets=datasets, n_ins=attrSize, n_outs=classSize, hidden_layers_sizes=[100, 100], pretraining_epochs=1000, pretrain_lr=1.0, training_epochs=2000, finetune_lr=0.1, batch_size=20)

# print(error)
# print(predictedLabels)

actualLabels = np.array(actualLabels)
predictedLabels = np.rint(predictedLabels).astype(int)

print('Accuracy: ',jaccard_similarity_score(actualLabels,predictedLabels))
print('Hamming Loss: ',hamming_loss(actualLabels,predictedLabels))
pass