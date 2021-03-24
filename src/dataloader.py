from tensorflow.keras.preprocessing.image import ImageDataGenerator



class DataLoader:
    def __init__(self, data_dir, batch_size):
        self.data_dir = data_dir
        self.batch_size = batch_size

    def init_generator(self):
        # Generate batches of images with augmentation for training set
        self.aug = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.15,
            height_shift_range=0.15,
            shear_range=0.15,
            zoom_range=0.15,
            horizontal_flip=True,
        )
        self.no_aug = ImageDataGenerator()
    
    def load_data(self): 
        img_size = (48, 48) 
        train_data = self.aug.flow_from_directory(
            directory=f'{self.data_dir}/train',
            color_mode='rgb',
            target_size=img_size,
            batch_size=self.batch_size, 
            shuffle=True,
            seed=18,
            class_mode='categorical'
        )
        val_data = self.no_aug.flow_from_directory(
            directory=f'{self.data_dir}/val',
            color_mode='rgb',
            target_size=img_size,
            batch_size=self.batch_size, 
            shuffle=True,
            seed=18,
            class_mode='categorical'
        )
        test_data = self.no_aug.flow_from_directory(
            directory=f'{self.data_dir}/test',
            color_mode='rgb',
            target_size=img_size,
            batch_size=self.batch_size, 
            shuffle=True,
            seed=18,
            class_mode='categorical'
        ) 
        return train_data, val_data, test_data