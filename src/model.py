import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import time

class Model:
    def __init__(self, lr=0.0001, epochs=20):
        self._lr = lr
        self._epochs = epochs
        self._model = self._construct_model()

    def _construct_model(self):
        """Create and compile a cnn tensorflow model
            Args:
                None
            Returns:
                model (Sequential): tensorflow sequential model
        """
        # img_shape = (224, 224, 3)
        # base_model = tf.keras.applications.MobileNetV2(
        #     input_shape=img_shape,
        #     include_top=False,
        #     weights='imagenet' 
        # )
        # preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
        # global_avg_layer = GlobalAveragePooling2D()
        # dense_1 = Dense(512, activation='relu')
        # dense_2 = .Dense(128, activation='relu')
        # dense_3 = Dense(32, activation='relu')
        # dense_4 = Dense(16, activation='relu')
        # prediction_layer = Dense(7, activation='softmax')

        # inputs = tf.keras.Input(shape=img_shape)
        # x = preprocess_input(inputs)
        # x = base_model(x, training=False)
        # x = global_avg_layer(x)
        # x = tf.keras.layers.Dropout(0.2)(x)
        # x = dense_1(x)
        # x = dense_2(x)
        # x = dense_3(x)
        # x = dense_4(x)
        # outputs = prediction_layer(x)
        # self.model = tf.keras.Model(inputs, outputs)

        HEIGHT=48
        WIDTH=48

        model = tf.keras.Sequential(name='Emotion_Detector')

        model.add(Conv2D(filters=64, kernel_size=(5,5), input_shape=(HEIGHT, WIDTH, 3), activation='elu', padding='same', kernel_initializer='he_normal', name='conv2d_1'))
        model.add(BatchNormalization(name='batchnorm_1'))
        model.add(Conv2D(filters=64, kernel_size=(5,5), activation='elu', padding='same', kernel_initializer='he_normal', name='conv2d_2'))
        model.add(BatchNormalization(name='batchnorm_2'))        
        model.add(MaxPooling2D(pool_size=(2,2), name='maxpool2d_1'))
        model.add(Dropout(0.4, name='dropout_1'))
        model.add(Conv2D(filters=128, kernel_size=(3,3), activation='elu', padding='same', kernel_initializer='he_normal', name='conv2d_3'))
        model.add(BatchNormalization(name='batchnorm_3'))
        model.add(Conv2D(filters=128, kernel_size=(3,3), activation='elu', padding='same', kernel_initializer='he_normal', name='conv2d_4'))
        model.add(BatchNormalization(name='batchnorm_4'))        
        model.add(MaxPooling2D(pool_size=(2,2), name='maxpool2d_2'))
        model.add(Dropout(0.4, name='dropout_2'))
        model.add(Conv2D(filters=256, kernel_size=(3,3), activation='elu', padding='same', kernel_initializer='he_normal', name='conv2d_5'))
        model.add(BatchNormalization(name='batchnorm_5'))
        model.add(Conv2D(filters=256, kernel_size=(3,3), activation='elu', padding='same', kernel_initializer='he_normal', name='conv2d_6'))
        model.add(BatchNormalization(name='batchnorm_6'))        
        model.add(MaxPooling2D(pool_size=(2,2), name='maxpool2d_3'))
        model.add(Dropout(0.5, name='dropout_3'))
        model.add(Flatten(name='flatten'))            
        model.add(Dense(128,activation='elu', kernel_initializer='he_normal', name='dense_1'))
        model.add(BatchNormalization(name='batchnorm_7'))        
        model.add(Dropout(0.6, name='dropout_4'))        
        model.add(Dense(5, activation='softmax', name='out_layer'))

        optimizer=tf.keras.optimizers.Adam(learning_rate=self._lr)

        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        model.summary()
        self.model = model
        return self.model

    def train(self, train_data, val_data):
        """Training of model with train data and validate on val data
        Args:
            train_data (DirectoryIterator)
            val_data (DirectoryIterator)
        """
        print("[INFO] training model...")

        early_stopping = EarlyStopping(
            monitor='val_accuracy', 
            mode='max', 
            verbose=1, 
            patience=6, 
            min_delta=1e-4
        )

        checkpoint = ModelCheckpoint(
            filepath='model/emo.h5',
            monitor='val_accuracy',
            verbose=1,
            save_best_only=True,
            mode='max'
        )

        self.H = self._model.fit(
            train_data,
            steps_per_epoch=len(train_data),
            validation_data=val_data,
            validation_steps=len(val_data),
            epochs=self._epochs,
            callbacks=[early_stopping, checkpoint]
        )

    def predict(self, test_data):
        """Make prediction on the test data
        Args:
            test_data (DirectoryIterator)
        Returns:
            test_loss (float)
            test_accuracy (float)
        """
        print("[INFO] evaluating network...")
        test_loss, test_accuracy = self._model.evaluate(
            test_data, 
            steps=None, 
            callbacks=None, 
            max_queue_size=10, 
            workers=1,
            use_multiprocessing=False, 
            verbose=0
        )
        return test_loss, test_accuracy

    def error_evaluation(self, test_data, PLOT_HEATMAP=True, SAVE_CSV=False):
        """Identify wrongly classified images in test set and generate a metrics report.
        Option to plot a heatmap of the confusion matrix and 
        export the paths of the wronlgy classified images as a csv.
        Args:
            test_data (DirectoryIterator)
        Returns:
            df: pandas DataFrame of the wrongly classified images
        """
        pred = self._model.predict(test_data)
        pred_index = np.argmax(pred, axis=1)
        true_index = test_data.classes
        fnames = test_data.filenames

        # Get class labels
        class_labels = []
        for key in test_data.class_indices:
            class_labels.append(key)

        # Print a metrics report
        print(classification_report(true_index, pred_index, target_names=class_labels))

        # Plot a heatmap of the confusion matrix
        if PLOT_HEATMAP:
            cm = confusion_matrix(true_index, pred_index)
            plt.figure(figsize=(10, 8))
            ax = sns.heatmap(cm, annot=True, lw=2, fmt='.2f', cmap='coolwarm', annot_kws={"size": 16})
            ax.set_xticklabels(class_labels, rotation=45, ha="right", fontsize=16)
            ax.set_yticklabels(class_labels, rotation=0, fontsize=16)
            ax.set_xlabel('Predicted Label', fontsize=16)
            ax.set_ylabel('True Label', fontsize=16)

        # Get wrongly predicted images
        errors = np.where(pred_index != true_index)[0]
        wrongly_classified_images = []
        true_labels = []
        pred_labels = []
        for i in errors:
            wrongly_classified_images.append(fnames[i])
            true_labels.append(class_labels[true_index[i]])
            pred_labels.append(class_labels[pred_index[i]])

        df = pd.DataFrame(
            {'wrongly_classified_images': wrongly_classified_images, 
             'true_labels': true_labels,
             'pred_labels': pred_labels}
            )

        if SAVE_CSV:
            timestr = time.strftime("%Y%m%d-%H%M%S")
            df.to_csv(
                f'/wrongly_classified_images_{timestr}.csv', 
                index=False, 
                header=False
            )
        return df

    def plot_training_acc(self):
        # plot the training loss and accuracy
        N = self._epochs
        H = self.H
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
        plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
        plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
        plt.title("Training Loss and Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="lower left")
        plt.show()
        # plt.savefig(plot_path)

    def save_model(self, path):
            """Save model in a '.h5' format
            """
            self.model.save(path)
            

