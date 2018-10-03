import numpy as np
import nibabel as nib
import keras
from keras.preprocessing import image
image.ImageDataGenerator

def loadDataGeneral(df,target_floder):
    """
    This function loads data stored in nifti format. Data should already be of
    appropriate shape.

    Inputs:
    - df: Pandas dataframe with two columns: image filenames and ground truth filenames.
    - path: Path to folder containing filenames from df.
    - append_coords: Whether to append coordinate channels or not.
    Returns:
    - X: Array of 3D images with 1 or 4 channels depending on `append_coords`.
    - y: Array of 3D masks with 1 channel.
    """

    X, y = [], []
    for i, item in df.iterrows():

        img_path = ''.join([str(target_floder), '/image/', item[0]])
        mask_path = ''.join([str(target_floder), '/label/', item[0]])
        nii_img = nib.load(img_path)
        img = nii_img.get_data()
        nii_mask = nib.load(mask_path)
        mask = nii_mask.get_data()
        mask = np.clip(mask, 0, 1)
        # cmask = (mask * 1. / 255)
        # out = cmask
        out = mask
        img = np.array(img, dtype=np.float64)
        brain = img > 0
        img -= img[brain].mean()
        img /= img[brain].std()
        X.append(img)
        y.append(out)
   # X = np.array(X, dtype=np.float64)
    X = np.array(X)
    #X -= X.mean()
    #X /= X.std()
    X = np.expand_dims(X, -1)  # X.shape  --> (4,128,128,64,1)
    y = np.expand_dims(y, -1)  # y.shape  --> (4,128,128,64,1)
    y_main = np.concatenate((1 - y, y), -1)  # y.shape  --> (4,128,128,64,2)
    y = np.array(y_main)
    # y_aux = np.array(y)



    print '### Dataset loaded'
    print '\t{}'.format(target_floder)
    print '\t{}\t{}'.format(X.shape, y.shape)
    print '\tX:{:.1f}-{:.1f}\ty:{:.1f}-{:.1f}\n'.format(X.min(), X.max(), y.min(), y.max())
    return X, y

class DataGenerator(keras.utils.Sequence):

    def __init__(self, df,target_floder, batch_size = 1,shuffle = False):
        self.df = df
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.target_floder = target_floder
        self.indexes = np.arange(self.df.size)
        self.temp_X = None
        self.temp_y_main = None
        self.temp_y_aux = None
    def __len__(self):
        """Number of batch in the Sequence.
        # Returns
            The number of batches in the Sequence.
        """
        return int(np.floor(self.df.size/self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_path_temp = [self.df.loc[i]["path"] for i in indexes]
        X, y_main, y_aux = self.__data_generation(list_path_temp,self.target_floder)
        self.temp_X = X
        self.temp_y_main = y_main
        self.temp_y_aux = y_aux
        return X, [y_main, y_aux]
    def getitem(self):
        return self.temp_X, [self.temp_y_main, self.temp_y_aux]

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        import random
        if self.shuffle == True:

            random.shuffle(self.indexes)

    def __data_generation(self,list_path_temp,target_floder):
        X, y = [], []
        for i, name in enumerate(list_path_temp):
            img_path = ''.join([str(target_floder), '/image/', name])
            mask_path = ''.join([str(target_floder), '/label/', name])
            nii_img = nib.load(img_path)
            img = nii_img.get_data()
            nii_mask = nib.load(mask_path)
            mask = nii_mask.get_data()
            mask = np.clip(mask, 0, 1)
            out = mask
            img = np.array(img, dtype=np.float64)
            brain = img > 0
            img -= img[brain].mean()
            img /= img[brain].std()
            X.append(img)
            y.append(out)
        # X = np.array(X, dtype=np.float64)
        X = np.array(X)
        # X -= X.mean()
        # X /= X.std()
        X = np.expand_dims(X, -1)  # X.shape  --> (4,128,128,64,1)
        y = np.expand_dims(y, -1)  # y.shape  --> (4,128,128,64,1)
        y_main = np.concatenate((1 - y, y), -1)  # y.shape  --> (4,128,128,64,2)
        y_main = np.array(y_main)
        y_aux = np.array(y)

        return X, y_main, y_aux

