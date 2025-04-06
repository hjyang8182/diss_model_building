import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, BatchNormalization, Conv1D, Conv2D, Flatten, Dense, Concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1, l2
import os
from tensorflow.keras.callbacks import CSVLogger, ReduceLROnPlateau, ModelCheckpoint
from data import ApneaDataLoader

spo2_cnn = load_model(f'/home/hy381/rds/hpc-work/models/spo2_cnn_1d_model/spo2_cnn_1d_model.keras')
mel_spec_cnn = load_model(f'/home/hy381/rds/hpc-work/models/audio_ms_cnn/audio_ms_cnn.keras')

def interm_model(): 
    spo2_input = Input(shape=(20,), name="spo2_input")
    audio_input = Input(shape=(128, 1251), name="audio_input")
    
    # spo2_feature = spo2_cnn(spo2_input)
    # audio_feature = mel_spec_cnn(audio_input)

    spo2_model = Model(inputs=spo2_cnn.get_layer(index=0).input, outputs=spo2_cnn.layers[-3].output)  # Second last layer outputa
    audio_model = Model(inputs=mel_spec_cnn.get_layer(index=0).input, outputs=mel_spec_cnn.layers[-3].output)  # Second last layer output

    spo2_input = Input(shape=(20,), name="spo2_input")
    audio_input = Input(shape=(128, 1251), name="audio_input")

    spo2_feature = spo2_model(spo2_input)
    audio_feature = audio_model(audio_input)

    fused = Concatenate()([spo2_feature, audio_feature])

    x = BatchNormalization()(fused)
    x = Dense(16, activation = 'relu', kernel_regularizer = l1(0.001))(x)
    x = Dropout(0.3)
    x = BatchNormalization()(fused)

    output = Dense(3, activation = 'softmax')(x)
    fusion_model = Model(inputs=[spo2_input, audio_input], outputs=output)
    optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-4, clipnorm = 1.0)
    fusion_model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy', 'precision', 'recall'])
    fusion_model.summary()
    return fusion_model

data_loader = ApneaDataLoader(train_batch_size=64)
train_spo2_data, valid_spo2_data, test_spo2_data = spo2_data_full = data_loader.load_full_data(['spo2'], apply_delta = True, batch = False)
train_audio_data, valid_audio_data, test_audio_data = data_loader.prepare_data(['audio_mel_spec'], parse_type = 'audio_feature', batch = False)

models_dir = '/home/hy381/rds/hpc-work/models'
model_name = 'if_cnn_ms_cnn'
model_dir = os.path.join(models_dir, model_name)
try: 
    os.mkdir(model_dir)
except FileExistsError: 
        print(f"Directory {model_dir} already exists")
except PermissionError: 
    print(f"Permission Denied to create directory {model_dir}")
except Exception as e: 
    print(f"Error: {e} occurred making directory {model_dir}")
    
model_file = os.path.join('/home/hy381/rds/hpc-work/models', model_name, f"{model_name}.keras")
output_file = os.path.join(model_dir, f"{model_name}.txt")
model_file = os.path.join(model_dir, f"{model_name}.keras")
log_file = os.path.join(model_dir, f"{model_name}.log")

checkpoint = tf.keras.callbacks.ModelCheckpoint(model_file, monitor = 'val_loss', save_best_only=True, mode = "min", verbose =1)
csv_logger = CSVLogger(log_file)
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1, epsilon=1e-4, mode='min')

def merge_fn(spo2_data, audio_data):
    spo2_input, spo2_label, _ = spo2_data
    audio_input, audio_label , _= audio_data
    return (spo2_input, audio_input), spo2_label

train_dataset = tf.data.Dataset.zip((train_spo2_data, train_audio_data))
train_dataset = train_dataset.map(merge_fn)
train_dataset = train_dataset.shuffle(buffer_size=10000).prefetch(buffer_size=tf.data.experimental.AUTOTUNE).batch(32).repeat()

valid_dataset = tf.data.Dataset.zip((valid_spo2_data, train_audio_data))
valid_dataset = valid_dataset.map(merge_fn)
valid_dataset = valid_dataset.shuffle(buffer_size=10000).prefetch(buffer_size=tf.data.experimental.AUTOTUNE).batch(32).repeat()

fusion_model = interm_model()


history = fusion_model.fit(train_dataset, steps_per_epoch = data_loader.FULL_TRAIN_STEPS_PER_EPOCH, epochs = 50, verbose = 1, validation_data = valid_dataset, callbacks = [checkpoint, csv_logger, reduce_lr_loss], validation_steps = data_loader.FULL_VALID_STEPS_PER_EPOCH)
print(history)
# def train_fused_model()