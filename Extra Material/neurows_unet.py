# %% [markdown]
# # **Neuroengineering Workshop 2021** 
# ## Deep Learning for CT SuperResolution
# .......................................................................................................................................................................................................................................................................................................................................

# %% [markdown]
# ### Environment setup and dependency docking

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-12-01T15:45:49.767900Z","iopub.execute_input":"2021-12-01T15:45:49.768157Z","iopub.status.idle":"2021-12-01T15:45:49.783534Z","shell.execute_reply.started":"2021-12-01T15:45:49.768126Z","shell.execute_reply":"2021-12-01T15:45:49.781216Z"}}
import os
import sys
import time
import gc
import pickle
import math
from random import shuffle
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.utils.layer_utils import count_params
print(f"Using TensorFlow version: {tf.__version__}")

SEED = 69
tf.random.set_seed(SEED)
print(f"RNG Seed: {SEED}")

tfk = tf.keras
tfkl = tf.keras.layers

# %% [markdown]
# 
# 
# ### Referencing data and setting up dimensions

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-12-01T15:45:51.467072Z","iopub.execute_input":"2021-12-01T15:45:51.467529Z","iopub.status.idle":"2021-12-01T15:45:51.908864Z","shell.execute_reply.started":"2021-12-01T15:45:51.467492Z","shell.execute_reply":"2021-12-01T15:45:51.908036Z"}}
ROOT_PATH = '../input/neuroengineering-project'

# FOLDER TO LOAD DATA FROM
DATA_PATH = os.path.join(ROOT_PATH,'Data','Data')
MODEL_PATH = os.path.join(ROOT_PATH,'Model')

# Volume size
N_ROWS_VOLUME           = 128
N_COLUMNS_VOLUME        = 128
N_SLICES_VOLUME         = 64
NOISE                   = 0.0001 

# Label size
N_ROWS_LABEL            = 256
N_COLUMNS_LABEL         = 256
N_SLICES_LABEL          = 64

# Type
VOLUME_TYPE       = 'nii'

VOLUME_TEMPLATE = "{}/VolumeCT_%s_{}_{}_{}_n{}.{}".format(
    DATA_PATH,
    N_ROWS_VOLUME,
    N_COLUMNS_VOLUME,
    N_SLICES_VOLUME,
    str(NOISE),
    VOLUME_TYPE
    )
LABEL_TEMPLATE = "{}/VolumeCT_%s_{}_{}_{}.{}".format(
    DATA_PATH,
    N_ROWS_LABEL,
    N_COLUMNS_LABEL,
    N_SLICES_LABEL,
    VOLUME_TYPE
    )

# Read available data
AVAILABLE_NUMBER_OF_CASES = 58
try:
    del trainVolumes
    del trainLabels
    del validationVolumes
    del validationLabels
    del testVolumes
    del testLabels
except:
    pass
gc.collect()
volumes_list = []
labels_list = []

for index_case in range(1, AVAILABLE_NUMBER_OF_CASES+1):
    case_id = "{:0>3}".format(index_case)
    volume_path = VOLUME_TEMPLATE % (case_id)
    label_path = LABEL_TEMPLATE % (case_id)
    if (os.path.exists(volume_path) and os.path.exists(label_path)):
        AVAILABLE_NUMBER_OF_CASES += 1
        volumes_list.append(volume_path)
        labels_list.append(label_path)
    else:
        print("Not found")
        print(volume_path)
        print(label_path)

# %% [markdown]
# ### Train-Validation-Test Splitting

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-12-01T15:45:52.552871Z","iopub.execute_input":"2021-12-01T15:45:52.553181Z","iopub.status.idle":"2021-12-01T15:45:55.721483Z","shell.execute_reply.started":"2021-12-01T15:45:52.553147Z","shell.execute_reply":"2021-12-01T15:45:55.720564Z"}}
AVAILABLE_NUMBER_OF_CASES = 58
# TRAINING-VALIDATION-TEST PERCENTAGES
TRAINING_PERC_CASES                 = 0.85
VALIDATION_PERC_CASES               = 0.10
TEST_PERC_CASES                     = 1 - TRAINING_PERC_CASES - VALIDATION_PERC_CASES

TRAINING_NUMBER_OF_CASES      = int(AVAILABLE_NUMBER_OF_CASES * TRAINING_PERC_CASES);
VALIDATION_NUMBER_OF_CASES    = int(AVAILABLE_NUMBER_OF_CASES * VALIDATION_PERC_CASES);
TEST_NUMBER_OF_CASES          = AVAILABLE_NUMBER_OF_CASES - TRAINING_NUMBER_OF_CASES - VALIDATION_NUMBER_OF_CASES;
print("Number of cases for training: ",(TRAINING_NUMBER_OF_CASES))
print("Number of cases for validation: " + str(VALIDATION_NUMBER_OF_CASES))
print("Number of cases for testing: " + str(TEST_NUMBER_OF_CASES))
# Training set
trainVolumes = np.empty((TRAINING_NUMBER_OF_CASES, N_ROWS_VOLUME, N_COLUMNS_VOLUME, N_SLICES_VOLUME)) 
trainLabels = np.empty((TRAINING_NUMBER_OF_CASES, N_ROWS_LABEL, N_COLUMNS_LABEL, N_SLICES_LABEL))  
# Validation set
validationVolumes = np.empty((VALIDATION_NUMBER_OF_CASES, N_ROWS_VOLUME, N_COLUMNS_VOLUME, N_SLICES_VOLUME)) 
validationLabels = np.empty((VALIDATION_NUMBER_OF_CASES, N_ROWS_LABEL, N_COLUMNS_LABEL, N_SLICES_LABEL))  
# Training set
testVolumes = np.empty((TEST_NUMBER_OF_CASES, N_ROWS_VOLUME, N_COLUMNS_VOLUME, N_SLICES_VOLUME)) 
testLabels = np.empty((TEST_NUMBER_OF_CASES, N_ROWS_LABEL, N_COLUMNS_LABEL, N_SLICES_LABEL))  

count           = 0
countTraining   = 0
countValidation = 0
countTest       = 0
for volume, label in zip(volumes_list, labels_list):
    if countTraining < TRAINING_NUMBER_OF_CASES:
        # get the refs to training set
        volumes = trainVolumes
        labels  = trainLabels
        index = countTraining
        countTraining += 1
    elif countValidation < VALIDATION_NUMBER_OF_CASES:
        # get the refs to validation set
        volumes = validationVolumes
        labels  = validationLabels
        index = countValidation
        countValidation += 1
    else:
        # get the refs to validation set
        volumes = testVolumes
        labels  = testLabels
        index = countTest
        countTest += 1
    temp = nib.load(label) # loading current label...
    temp = temp.get_fdata()
    temp = np.asarray(temp)
    labels[index, :, :, :] = temp # ...into buffer
    
    temp = nib.load(volume) # loading corresponding volume...
    temp = temp.get_fdata()
    temp = np.asarray(temp)
    volumes[index, :, :, :] = temp # ...into buffer

trainVolumes = trainVolumes.reshape(trainVolumes.shape + (1,)) # necessary to give it as input to model  
validationVolumes = validationVolumes.reshape(validationVolumes.shape + (1,)) # necessary to give it as input to model  
testVolumes = testVolumes.reshape(testVolumes.shape + (1,)) # necessary to give it as input to model

# %% [markdown]
# .......................................................................................................................................................................................................................................................................................................................................
# 

# %% [markdown]
# ## **Model Definition**

# %% [code] {"execution":{"iopub.status.busy":"2021-12-01T15:45:59.925207Z","iopub.execute_input":"2021-12-01T15:45:59.925474Z","iopub.status.idle":"2021-12-01T15:46:01.055210Z","shell.execute_reply.started":"2021-12-01T15:45:59.925444Z","shell.execute_reply":"2021-12-01T15:46:01.054459Z"}}
stacks = 4
initial_filters = 2

class show_activations(tf.keras.callbacks.Callback):
    def on_train_batch_begin(self, batch, logs=None):
        out = [layer.output for layer in self.model.layers[12:16]]
        act_model = tfk.models.Model(inputs = self.model.input, outputs = out)
        activations = act_model.predict(np.expand_dims(testVolumes[0],0))
        for act in activations:
            for i in range(act.shape[-1]):
                plt.figure(figsize=(20,10))
                plt.subplot(1,act.shape[-1],i+1)
                plt.imshow(act[0,:,:,32,i])
                plt.show()
            
# Initialize input size
tfkl = tf.keras.layers

input_tensor = tf.keras.layers.Input(shape=(N_ROWS_VOLUME,
                                            N_COLUMNS_VOLUME,
                                            N_SLICES_VOLUME,
                                            1))
stack_output=[]
xups = tfkl.UpSampling3D(size=(2,2,1))(input_tensor)
stack_output.append(xups)

for st in range(1,stacks+1):
    st = int(st)
    xs = tfkl.Conv3DTranspose(
        filters=initial_filters*st,
        kernel_size=(N_ROWS_VOLUME//(2**st)+1,N_COLUMNS_VOLUME//(2**st)+1,1),
        padding='valid'
    )(input_tensor)
    for sp in range((2**st)-1):
        sp=int(sp)
        xs = tfkl.Conv3DTranspose(
            filters=initial_filters*st*(sp+2)*0.75,
            kernel_size=(N_ROWS_VOLUME//(2**st)+1,N_COLUMNS_VOLUME//(2**st)+1,1),
            padding='valid'
        )(xs)
    stack_output.append(xs)


xc = tfkl.Concatenate()(stack_output)

x= tfkl.Conv3D(
    filters=32,
    kernel_size=(1,1,1),
    padding='same'
)(xc)
x= tfkl.Conv3D(
    filters=16,
    kernel_size=(3,3,3),
    padding='same'
)(x)
x= tfkl.Conv3D(
    filters=8,
    kernel_size=(3,3,3),
    padding='same'
)(x)
x= tfkl.Conv3D(
    filters=4,
    kernel_size=(3,3,3),
    padding='same'
)(x)
x= tfkl.Conv3D(
    filters=2,
    kernel_size=(3,3,3),
    padding='same'
)(x)
x= tfkl.Conv3D(
    filters=1,
    kernel_size=(3,3,3),
    padding='same'
)(x)
#############################
output_tensor = x
output_tensor = tf.keras.layers.Reshape((N_ROWS_LABEL,
                                         N_COLUMNS_LABEL,
                                         N_SLICES_LABEL))(x)

model = tf.keras.Model(inputs = [input_tensor], 
                          outputs = [output_tensor])

print("TRAINABLE PARAMETERS: ","{:e}".format(count_params(model.trainable_weights)))
model.summary()
tf.keras.utils.plot_model(model,rankdir='LR',show_shapes=True)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-12-01T14:11:33.139299Z","iopub.execute_input":"2021-12-01T14:11:33.139617Z","iopub.status.idle":"2021-12-01T14:11:34.768917Z","shell.execute_reply.started":"2021-12-01T14:11:33.139583Z","shell.execute_reply":"2021-12-01T14:11:34.766472Z"}}
################################################################################################################
################################################################################################################

def unet_superres(input_shape=(128,128,64),initial_filters=4,stages=4,filter_scaling=4,enc_dec_ratio=3/4,l2=0,model_name='unet_superres'):        
    
    # INPUT
    input = tfk.Input(shape=(input_shape[0],input_shape[1],input_shape[2],1))
    x=input
    
    # PARALLEL PATH 
    xp_early = tfkl.Conv3DTranspose(
        filters=int(initial_filters*filter_scaling*0.25),
        kernel_size=(int(input_shape[0]*0.5+1), int(input_shape[0]*0.5+1), 1),
        strides=1, 
        padding='valid',
        kernel_initializer=tfk.initializers.GlorotUniform(),
        name=f"TrConv3D_parallel_1.5_early")(x)
    
    # ENCODING PATH
    c = []
    for e in range(stages):
        x = tfkl.Conv3D(
            filters=initial_filters*filter_scaling*(e+1),
            kernel_size=3, 
            strides=1,
            padding='same',
            kernel_initializer=tfk.initializers.GlorotUniform(),
            kernel_regularizer=tfk.regularizers.l2(l2),
            name=f"Conv3D1_enc_{(e+1)}")(x)
        x = tfkl.BatchNormalization()(x)
        x = tfkl.LeakyReLU(     
            alpha=1,
            name=f"LeakyReLU1_enc_{(e+1)}")(x)
        
        x = tfkl.Conv3D(
            filters=initial_filters*filter_scaling*(e+1),
            kernel_size=5, 
            strides=1,
            padding='same',
            kernel_initializer=tfk.initializers.GlorotUniform(),
            kernel_regularizer=tfk.regularizers.l2(l2),
            name=f"Conv3D2_enc_{(e+1)}")(x)
        x = tfkl.BatchNormalization()(x)
        x = tfkl.LeakyReLU(     
            alpha=1,
            name=f"LeakyReLU2_enc_{(e+1)}")(x)
        c.append(x)
        x = tfkl.MaxPool3D(
            pool_size=(2,2,1),
            name=f"MaxPool3D_enc_{(e+1)}")(x)
        
    # BOTTLENECK
    x = tfkl.Conv3D(
            filters=initial_filters*filter_scaling*(e+2),
            kernel_size=3, 
            strides=1,
            padding='same',
            kernel_initializer=tfk.initializers.GlorotUniform(),
            kernel_regularizer=tfk.regularizers.l2(l2),
            name=f"Conv3D1_bottleneck")(x)
    x = tfkl.BatchNormalization()(x)
    x = tfkl.LeakyReLU(     
            alpha=1,
            name=f"LeakyReLU1_bottleneck")(x)
    x = tfkl.Conv3D(
            filters=initial_filters*filter_scaling*(e+2),
            kernel_size=3, 
            strides=1,
            padding='same',
            kernel_initializer=tfk.initializers.GlorotUniform(),
            kernel_regularizer=tfk.regularizers.l2(l2),
            name=f"Conv3D2_bottleneck")(x)
    x = tfkl.BatchNormalization()(x)
    x = tfkl.LeakyReLU(     
            alpha=1,
            name=f"LeakyReLU2_bottleneck")(x)

    # DECODING PATH
    for d in range(stages):        
        x = tfkl.Conv3DTranspose(
                filters=int(initial_filters*filter_scaling*(e+2)-initial_filters*filter_scaling*(d+1)*enc_dec_ratio),
                kernel_size=5, 
                strides=1, 
                padding='same',
                kernel_initializer=tfk.initializers.GlorotUniform(),
                kernel_regularizer=tfk.regularizers.l2(l2),
                name=f"TrConv3D1_dec_{(stages-d)}")(x)
        x = tfkl.BatchNormalization()(x)
        x = tfkl.LeakyReLU(     
                alpha=1,
                name=f"LeakyReLU1_dec_{stages-d}")(x)

        x = tfkl.Conv3DTranspose(
                filters=int(initial_filters*filter_scaling*(e+2)-initial_filters*filter_scaling*(d+1)*enc_dec_ratio),
                kernel_size=3, 
                strides=1, 
                padding='same',
                kernel_initializer=tfk.initializers.GlorotUniform(),
                kernel_regularizer=tfk.regularizers.l2(l2),
                name=f"TrConv3D2_dec_{stages-d}")(x)
        x = tfkl.UpSampling3D(size=(2,2,1))(x)
        x = tfkl.LeakyReLU(     
                alpha=1,
                name=f"LeakyReLU2_dec_{stages-d}")(x)
        
        x = tfkl.Concatenate()([x,c[-d-1]])
    
    xp_late = tfkl.Conv3DTranspose(
            filters=int((initial_filters*filter_scaling*(e+2)-initial_filters*filter_scaling*(d+1+1)*enc_dec_ratio)*0.5),
            kernel_size=(int(input_shape[0]*0.5+1), int(input_shape[0]*0.5+1), 1),
            strides=1, 
            padding='valid',
            kernel_initializer=tfk.initializers.GlorotUniform(),
            name=f"TrConv3D_parallel_1.5_late")(x)
    xp_late = tfkl.LeakyReLU(     
            alpha=1,
            name=f"LeakyReLU1.5_dec_{stages-d}")(xp_late)
    x = tfkl.Concatenate()([xp_early,xp_late])
    
    x = tfkl.Conv3DTranspose(
            filters=1,
            kernel_size=(int(input_shape[0]*0.5+1), int(input_shape[0]*0.5+1), 1),
            strides=1, 
            padding='valid',
            kernel_initializer=tfk.initializers.GlorotUniform(),
            name=f"TrConv3D_final")(x)
    
    out = tfkl.LeakyReLU(     
            alpha=1,
            name=f"LeakyReLU2_final")(x)

    
    return tfk.Model(inputs=input, outputs=[out], name=model_name)

################################################################################################################
################################################################################################################

MODEL_NAME        = 'U-Net_superresolution'
INPUT_SHAPE       = (128,128,64)
INITAL_FILTERS    = 4
STAGES            = 4
FILTER_SCALING    = 4
ENC_DEC_RATIO     = 3/4
L2                = 1e-6

model = unet_superres(
    input_shape=INPUT_SHAPE,
    initial_filters=INITAL_FILTERS,
    stages=STAGES,
    filter_scaling=FILTER_SCALING,
    enc_dec_ratio=ENC_DEC_RATIO,
    l2=L2,
    model_name=MODEL_NAME,
)

print("TRAINABLE PARAMETERS: ","{:e}".format(count_params(model.trainable_weights)))

model.summary()
#!pip install visualkeras
#import visualkeras
#visualkeras.layered_view(model)
tfk.utils.plot_model(model,rankdir='LR',show_shapes=True)

# %% [markdown]
# # Training

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-12-01T15:46:24.008461Z","iopub.execute_input":"2021-12-01T15:46:24.009092Z","iopub.status.idle":"2021-12-01T15:48:19.353900Z","shell.execute_reply.started":"2021-12-01T15:46:24.009048Z","shell.execute_reply":"2021-12-01T15:48:19.352429Z"}}
# Custom metrics definition
def psnr_loss(target_data, ref_data):
    return -1*tf.image.psnr(target_data, ref_data, 1)

def psnr(target_data, ref_data):
    return tf.image.psnr(target_data, ref_data, 1)

def ssim(target_data, ref_data):
    return tf.image.ssim(target_data, ref_data, 1)

def msle(target_data,ref_data):
    return tf.keras.metrics.mean_squared_logarithmic_error()

my_loss = tfk.losses.MeanSquaredLogarithmicError()

my_metrics = []
my_metrics.append(psnr)
my_metrics.append(ssim)

my_callbacks = []
my_callbacks.append(show_activations())
my_callbacks.append(
    tf.keras.callbacks.EarlyStopping(
    monitor="val_psnr",
    patience=9,
    mode="max",
    restore_best_weights=True)
)

my_callbacks.append(
    tf.keras.callbacks.ModelCheckpoint(
        "checkpoint.h5",
        monitor="val_loss",
        save_best_only=False,
        save_weights_only=False,
        mode="min",
    )
)

my_callbacks.append(
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_psnr",
        factor=0.1,
        patience=5,
        verbose=1,
        mode="max"
    ),
)

LEARNING_RATE = 1e-1
BATCH_SIZE = 2
MAX_EPOCHS = 50

model.compile(
    optimizer = tf.keras.optimizers.Adam(LEARNING_RATE),
    loss      = my_loss,
    metrics   = my_metrics 
    )

# Run Model Training
monitoring = model.fit(
    x = trainVolumes, 
    y = trainLabels, 
    batch_size = BATCH_SIZE, 
    epochs = MAX_EPOCHS, 
    validation_data = (validationVolumes,
                     validationLabels),
    callbacks = my_callbacks) 

# Save the net
model.save("./model.h5")

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-12-01T14:06:55.169915Z","iopub.status.idle":"2021-12-01T14:06:55.170721Z","shell.execute_reply.started":"2021-12-01T14:06:55.170456Z","shell.execute_reply":"2021-12-01T14:06:55.170481Z"}}
# REDECLARING METRICS IF MODEL IS LOADED PRE-TRAINED
def psnr_loss(target_data, ref_data):
    return -1*tf.image.psnr(target_data, ref_data, 1)

def psnr(target_data, ref_data):
    return tf.image.psnr(target_data, ref_data, 1)

def ssim(target_data, ref_data):
    return tf.image.ssim(target_data, ref_data, 1)

my_loss = 'mse'

my_metrics = []
my_metrics.append(psnr)
my_metrics.append(ssim)

##############################################################################################

# If you want to load a pre-trained model
# Set this parameter 'True' only if you want to load another model, otherwise leave it 'False'
LOAD = True                    
MODEL_NAME = 'model_00.h5'      # The name of the model to load 
LEARNING_RATE = 1e-3
BATCH_SIZE = 1
MAX_EPOCHS = 50

if LOAD is True:
    #model_to_load = os.path.join(MODEL_PATH, MODEL_NAME)
    my_model = tf.keras.models.load_model('../input/modelokk/model.h5',compile=False)
    print("Model loaded!")

    my_model.compile(
    optimizer = tf.keras.optimizers.Adam(LEARNING_RATE),
    loss      = my_loss,
    metrics   = my_metrics 
    )
    
PLOT = True
          
# just visualizing slice by slice:
if PLOT:
    for case in range(len(testVolumes)):
        print(f"Evaluating Test case #{case+1}")
        prediction = my_model.predict(np.expand_dims(testVolumes[case],0))
        print(f"PSNR={psnr(testLabels[case, :, :, :].astype('float32'), prediction[0,:, :, :,0].astype('float32'))} - SSIM={ssim(testLabels[case, :, :, :].astype('float32'), prediction[0,:, :, :,0].astype('float32'))}")
        for i in range(0, N_SLICES_VOLUME):
            fig = plt.figure(figsize = (25,17))
            plt.subplot(1, 3, 1)
            plt.imshow(testVolumes[case, :, :, i, 0], cmap = 'gray')
            plt.title("Original 128x128x64")
            plt.subplot(1, 3, 2)
            plt.imshow(testLabels[case, :, :, i], cmap = 'gray')
            plt.title("Expected 256x256x64")
            plt.subplot(1, 3, 3)
            plt.imshow(prediction[0,:, :, i], cmap = 'gray')
            plt.title(f"Predicted 256x256x64")
            plt.show(fig)

# %% [code] {"execution":{"iopub.status.busy":"2021-12-01T14:06:55.171857Z","iopub.status.idle":"2021-12-01T14:06:55.17273Z","shell.execute_reply.started":"2021-12-01T14:06:55.172488Z","shell.execute_reply":"2021-12-01T14:06:55.172526Z"}}
prediction[0,:, :, :,0].astype('float32').shape

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-12-01T14:06:55.173973Z","iopub.status.idle":"2021-12-01T14:06:55.174397Z","shell.execute_reply.started":"2021-12-01T14:06:55.174156Z","shell.execute_reply":"2021-12-01T14:06:55.174178Z"}}
print(f"PSNR={psnr(testLabels[case, :, :, :].astype('float32'), prediction[0,:, :, :].astype('float32'))} - SSIM={ssim(testLabels[case, :, :, :], prediction[0,:, :, :])}")

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-12-01T14:06:55.175676Z","iopub.status.idle":"2021-12-01T14:06:55.176074Z","shell.execute_reply.started":"2021-12-01T14:06:55.175857Z","shell.execute_reply":"2021-12-01T14:06:55.17588Z"}}
print(my_model.evaluate(x = testVolumes, y = testLabels))